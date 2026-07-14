# Absolute joint-motion hard-fail layer for the GE-Sim-V2 evaluators

**Status:** implemented, verified end-to-end. All changes live under `tools/`,
`tests/`, `configs/` (the `tools/` scripts are currently untracked/uncommitted).

## Why

Both evaluators of the world model's predicted `(T,16)` state trajectory
(`tools/eval_state_continuity.py`, `tools/eval_temporospatial_risk.py`) scored
frames **only** with a robust z-score computed from the *same rollout's* own
median/MAD. That is a relative outlier detector, not an absolute
physical-plausibility check. Two blind spots:

1. A **uniformly** too-fast rollout shifts the whole jerk distribution together,
   so median/MAD absorb it and z-scores stay near zero — nothing fires no matter
   how bad the trajectory is globally.
2. A mostly-static rollout with one legitimate fast motion makes that frame a
   huge *relative* outlier — a false fail on physically valid motion.

Tuning `fail_z`/`warn_z` can't fix either; it is a property of self-referential
normalization. The fix is an **absolute** threshold layer added alongside the
existing relative one (the relative system is unchanged).

## Key correction to the original plan — state is joint space, not quaternion

The original plan assumed each 7-dim per-arm block was an EEF pose
(`3 pos + 4 quat`) and specified a rotation-geodesic metric with a `q/-q`
double-cover fix. **That is wrong for this repo.** The `(T,16)` `state.npy` is
**joint space**: WM layout `[L7_arm(0:7), L_grip(7), R7_arm(8:15), R_grip(15)]`,
each 7-block being **7 revolute joint angles in radians**.

Confirmed from source — `src/gesim/types.py` (WM layout), `src/gesim/episode.py`
(`actions_0.npy` = "absolute-joint actions"; `state_joints_0.npy (T,20) =
[L7,R7,L_grip,R_grip,head2,waist2]`), `src/gesim/env.py` ("actions: (L,16)
joint-space") — and empirically: the alleged "quaternion" sub-block
`state[:,3:7]` has norm ≈ 2.0, not 1.0.

Consequences: no quaternion/6D anywhere in `state.npy`; the double-cover concern
is moot (joint angles are a plain vector, component-wise differencing is exact);
the absolute axis is **joint angular velocity/acceleration** (deg/frame,
deg/frame²), not a rotation geodesic. FK *is* available
(`gesim.conditioning.kinematics.CompiledKinematics.fk_action` → per-arm
`[xyz,qxyzw]`) if an EEF-orientation metric is ever wanted, but that grades a
derived quantity rather than the world model's raw output.

## What changed

### New files
- **`tools/joint_motion_metrics.py`** — single source of truth imported by both
  evaluators: `joint_step_deg` (deg/frame), `joint_acc_deg` (signed deg/frame²),
  `arm_joint_motion` (per-frame max-over-joints step & acc), and a `JointBounds`
  config object loaded from JSON (never inlined).
- **`tools/compute_joint_bounds.py`** — scans a directory of GT joint arrays
  (`state_joints_0.npy` `(T,20)` or WM `state.npy` `(T,16)`, auto-detected by
  width) and emits `configs/joint_bounds.json` at a high percentile (default
  p99.9). Keeps the bound external and fixed, not derived from the sample graded.
- **`configs/joint_bounds.json`** — generated from the 3 demo assets:
  `joint_step_deg_max ≈ 2.57 deg/frame`, `joint_acc_deg_max ≈ 1.34 deg/frame²`
  at fps 16 (p99.9).
- **`tests/test_joint_motion_metrics.py`** — 7 unit tests (static → 0; constant
  velocity → constant step, zero acc; rad→deg + max-over-joints; no sign-flip
  artifact; supra-threshold detection; signed acc; bounds round-trip).

### Modified (grafted onto existing code — relative system untouched)
- **`tools/eval_temporospatial_risk.py`**: `--joint-bounds` flag +
  `hard_fail.joint_step_deg_max/joint_acc_deg_max` config keys; per-frame
  absolute hard-fail checks tagged `hard_fail_joint_step_deg` /
  `hard_fail_joint_acc_deg` (distinct from relative reasons, same tier as
  `nan_or_inf_state`, **not** folded into the weighted `risk` score); new
  `joint_step_deg`/`joint_acc_deg` CSV columns; summary percentiles + hard-fail
  counts + `joint_bounds`; `plots/joint_step_deg.png`; report/console notes.
- **`tools/eval_state_continuity.py`**: `--joint-step-max-deg`,
  `--joint-acc-max-deg`, `--joint-bounds` args; a frame fails on the z threshold
  **or** an absolute threshold, each reported via a `flagged_by` column
  (`relative_z` / `absolute_joint_step` / `absolute_joint_acc`); new CSV columns;
  `most_alerting_frame.txt` reports which check flagged the worst frame plus the
  worst absolute-step frame; summary gains per-check fail counts and percentiles.

Event alignment: absolute signals are aligned to the same `event_frame = i+2`
convention by taking the worst value within the jerk window (`step[i:i+3]`,
`acc[i:i+2]`).

## How to run

```bash
# 1. (Re)compute bounds from GT (external, fixed reference).
python3 tools/compute_joint_bounds.py \
    --glob 'assets/*/state_joints_0.npy' \
    --percentile 99.9 --fps 16 \
    --out configs/joint_bounds.json

# 2. Temporospatial risk eval + top-dangerous clips into the replay tree.
python3 tools/eval_temporospatial_risk.py \
    --state outputs/replay/state.npy \
    --video outputs/replay/video.mp4 \
    --joint-bounds configs/joint_bounds.json \
    --out outputs/replay/eval_temporospatial_risk

# 3. Lightweight state-continuity eval.
python3 tools/eval_state_continuity.py \
    --state outputs/replay/state.npy \
    --joint-bounds configs/joint_bounds.json \
    --out outputs/eval_state_continuity

# Tests (no project install needed):
uv run --no-project --with pytest --with numpy pytest tests/test_joint_motion_metrics.py -q
```

Both evaluators run with **no new required args** — the absolute checks default
to disabled (`None`) unless `--joint-bounds` (or config/CLI values) is supplied.

## Verification (on `outputs/replay/state.npy`, 450 frames)

- The rollout is *uniformly* too fast (max 6.0 deg/frame vs GT bound 2.57): the
  relative z-score alone yields **0–1 fails**, the absolute bound flags **33**
  joint-velocity and **64** joint-accel frames — exactly blind spot (1).
- A GT trajectory scored against the same bound gets **0 absolute fails**; its 33
  "fails" are all `relative_z` false alarms on legit fast reaches — blind spot
  (2), correctly *not* flagged by the absolute axis.
- Top-10 risk clips extracted to
  `outputs/replay/eval_temporospatial_risk/clips/` (10-frame H.264/faststart mp4,
  all playable). Clips are still ranked by the existing `risk_score` (relative);
  ranking by absolute severity is a small, separate opt-in if desired.

## Non-goals honored
Relative risk system (`weights`, `thresholds`, z-score logic),
`near_chunk_boundary()`, and all existing CLI args / output filenames are
unchanged; every addition is additive.
