#!/usr/bin/env python3
"""Compute absolute joint-motion bounds from ground-truth trajectories.

Scans a directory of GT joint arrays and emits ``joint_bounds.json`` holding a
high-percentile absolute joint angular velocity (deg/frame) and acceleration
(deg/frame^2). Both evaluators load this JSON so the bound stays *external and
fixed* rather than derived from the sample under test.

Two GT array shapes are auto-detected by column count:
- (T, 20) ``state_joints_0.npy`` layout ``[L7, R7, L_grip, R_grip, head2, waist2]``
  -> arm joints are columns 0:14.
- (T, 16) WM ``state.npy`` layout ``[L7_arm, L_grip, R7_arm, R_grip]``
  -> arm joints are columns [0:7] + [8:15].

Override with ``--arm-dims`` if your files differ.

Example:
    python3 tools/compute_joint_bounds.py \
        --glob 'assets/*/state_joints_0.npy' \
        --percentile 99.9 --fps 16 \
        --out configs/joint_bounds.json
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from joint_motion_metrics import (
    DEFAULT_ARM_DIMS,
    arm_joint_motion,
)

# Arm-joint columns for the (T,20) state_joints layout [L7, R7, Lg, Rg, head2, waist2].
STATE_JOINTS_ARM_DIMS = list(range(0, 14))


def arm_dims_for(width: int, override) -> list:
    if override is not None:
        return override
    if width >= 20:
        return STATE_JOINTS_ARM_DIMS
    if width >= 16:
        return DEFAULT_ARM_DIMS
    raise ValueError(f"cannot infer arm dims for array width {width}; pass --arm-dims")


def main():
    ap = argparse.ArgumentParser(description="Compute absolute joint-motion bounds from GT.")
    ap.add_argument("--glob", action="append", required=True,
                    help="Glob for GT joint .npy files. Repeatable.")
    ap.add_argument("--percentile", type=float, default=99.9,
                    help="Percentile of the pooled deg/frame distribution to use as bound.")
    ap.add_argument("--fps", type=float, default=16.0, help="Eval fps (provenance only).")
    ap.add_argument("--arm-dims", type=int, nargs="+", default=None,
                    help="Explicit arm-joint column indices (overrides auto-detect).")
    ap.add_argument("--out", default="configs/joint_bounds.json", help="Output JSON path.")
    args = ap.parse_args()

    paths = []
    for g in args.glob:
        paths.extend(sorted(glob.glob(g)))
    paths = [p for p in paths if Path(p).is_file()]
    if not paths:
        raise SystemExit(f"no files matched: {args.glob}")

    all_step = []  # (N,) per-frame max joint deg/frame, pooled
    all_acc = []
    used = []
    for p in paths:
        arr = np.load(p)
        if arr.ndim != 2 or arr.shape[0] < 3:
            print(f"[skip] {p}: shape {arr.shape} too small")
            continue
        dims = arm_dims_for(arr.shape[1], args.arm_dims)
        if max(dims) >= arr.shape[1]:
            print(f"[skip] {p}: arm dims exceed width {arr.shape[1]}")
            continue
        arms = arr[:, dims]
        step_max, acc_max = arm_joint_motion(arms)
        all_step.append(step_max)
        all_acc.append(acc_max)
        used.append(p)
        print(f"[ok] {p}: T={arr.shape[0]} arm_dims={len(dims)} "
              f"step_p{args.percentile}={np.percentile(step_max, args.percentile):.4f} "
              f"step_max={step_max.max():.4f}")

    if not all_step:
        raise SystemExit("no usable files")

    step_pool = np.concatenate(all_step)
    acc_pool = np.concatenate(all_acc)

    bounds = {
        "joint_step_deg_max": float(np.percentile(step_pool, args.percentile)),
        "joint_acc_deg_max": float(np.percentile(acc_pool, args.percentile)),
        "fps": float(args.fps),
        "percentile": float(args.percentile),
        "n_frames_step": int(step_pool.size),
        "n_frames_acc": int(acc_pool.size),
        "source_files": used,
        "note": ("Absolute joint angular velocity/acceleration bounds from GT. "
                 "deg/frame at the stated fps. Loaded by the evaluators; not inlined."),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(bounds, f, indent=2)

    print("\n=== joint bounds ===")
    print(f"files: {len(used)}  pooled step frames: {step_pool.size}")
    print(f"joint_step_deg_max (p{args.percentile}): {bounds['joint_step_deg_max']:.4f} deg/frame")
    print(f"joint_acc_deg_max  (p{args.percentile}): {bounds['joint_acc_deg_max']:.4f} deg/frame^2")
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
