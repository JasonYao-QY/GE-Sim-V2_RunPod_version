#!/usr/bin/env python3
import argparse
import csv
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from joint_motion_metrics import JointBounds, arm_joint_motion


DEFAULT_CONFIG = {
    "dims": {
        "arm": list(range(0, 7)) + list(range(8, 15)),
        "grip": [7, 15],
    },
    "chunk": {
        "size": 25,
        "boundary_window": 2,
    },
    "robust": {
        "z_cap": 10.0,
        "eps": 1e-12,
    },
    "score": {
        "missing_action_unexplained_factor": 0.5,
        "weights": {
            "unexplained_state_jerk": 0.35,
            "video_jump": 0.25,
            "state_acc": 0.15,
            "boundary_state_jerk": 0.15,
            "state_jerk": 0.10,
        },
        "thresholds": {
            "warning": 0.30,
            "alert": 0.50,
            "fail": 0.70,
        },
    },
    "hard_fail": {
        "nan_or_inf_state": True,
        "video_decode_failure": True,
        "state_jerk_z": 15.0,
        "video_jump_z": 15.0,
        # Absolute (non-relative) joint-motion limits, deg/frame at eval fps.
        # None disables the check. Populate from configs/joint_bounds.json
        # (compute_joint_bounds.py) or a robot joint-speed spec; never inline a
        # guessed magic number in the scoring logic.
        "joint_step_deg_max": None,
        "joint_acc_deg_max": None,
    },
    "video": {
        "width": 128,
        "height": 64,
    },
    "clips": {
        "top_k": 10,
        "back": 5,
        "forward": 5,
        "fps": 16,
    },
}


def deep_update(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Optional[str]) -> dict:
    cfg = DEFAULT_CONFIG
    if path:
        with open(path, "r") as f:
            user_cfg = json.load(f)
        cfg = deep_update(cfg, user_cfg)
    return cfg


def rms(x: np.ndarray, axis=1) -> np.ndarray:
    return np.sqrt(np.mean(x ** 2, axis=axis))


def robust_z(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    mad = max(float(mad), eps)
    return (x - med) / mad


def pos_clip_z(z: np.ndarray, z_cap: float) -> np.ndarray:
    return np.clip(np.maximum(z, 0.0), 0.0, z_cap)


def norm01_from_z(z: np.ndarray, z_cap: float) -> np.ndarray:
    return pos_clip_z(z, z_cap) / z_cap


def finite_diffs(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    step = x[1:] - x[:-1]
    acc = x[2:] - 2 * x[1:-1] + x[:-2]
    jerk = x[3:] - 3 * x[2:-1] + 3 * x[1:-2] - x[:-3]
    return step, acc, jerk


def near_chunk_boundary(frame: int, chunk_size: int, window: int) -> bool:
    if chunk_size <= 0:
        return False

    # Chunks: 0-24, 25-49, 50-74...
    # Boundary is between 24|25, 49|50, 74|75...
    r = frame % chunk_size

    # Near right side of boundary: 25, 50, 75...
    near_right = min(r, chunk_size - r) <= window

    # Near left side of boundary: 24, 49, 74...
    near_left = abs(r - (chunk_size - 1)) <= window

    return bool(near_right or near_left)


def decode_video_gray(video_path: Path, width: int, height: int) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Returns frames as float32 array (T,H,W) in [0,1].
    Uses ffmpeg rawvideo for minimal Python dependencies.
    """
    if shutil.which("ffmpeg") is None:
        return None, "ffmpeg_not_found"

    cmd = [
        "ffmpeg",
        "-v", "error",
        "-i", str(video_path),
        "-vf", f"scale={width}:{height},format=gray",
        "-f", "rawvideo",
        "-pix_fmt", "gray",
        "-",
    ]

    try:
        proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        return None, f"ffmpeg_decode_failed: {e.stderr.decode(errors='ignore')[:500]}"

    raw = proc.stdout
    frame_size = width * height
    if len(raw) == 0:
        return None, "ffmpeg_decoded_zero_bytes"
    if len(raw) % frame_size != 0:
        return None, f"raw_size_not_divisible: bytes={len(raw)}, frame_size={frame_size}"

    n = len(raw) // frame_size
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(n, height, width)
    arr = arr.astype(np.float32) / 255.0
    return arr, None


def video_temporal_features(video_frames: np.ndarray) -> Dict[str, np.ndarray]:
    """
    video_step[i] = visual change from frame i to i+1
    video_acc[i]  = change in visual change around frame i+1
    video_jerk[i] = change in video acceleration around event_frame i+2
    """
    if video_frames.shape[0] < 4:
        raise ValueError(f"Need >=4 video frames, got {video_frames.shape[0]}")

    diff = np.mean(np.abs(video_frames[1:] - video_frames[:-1]), axis=(1, 2))
    acc = np.abs(diff[1:] - diff[:-1])
    jerk = np.abs(acc[1:] - acc[:-1])

    return {
        "video_step": diff,
        "video_acc": acc,
        "video_jerk": jerk,
    }


def extract_clip(
    video_path: Path,
    out_path: Path,
    start_frame: int,
    end_frame_exclusive: int,
    fps: float,
) -> bool:
    if shutil.which("ffmpeg") is None:
        return False

    end_inclusive = end_frame_exclusive - 1
    if end_inclusive < start_frame:
        return False

    vf = f"select='between(n\\,{start_frame}\\,{end_inclusive})',setpts=N/FRAME_RATE/TB"

    cmd = [
        "ffmpeg",
        "-y",
        "-v", "error",
        "-i", str(video_path),
        "-vf", vf,
        "-an",
        "-r", str(fps),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(out_path),
    ]

    subprocess.run(cmd, check=True)
    return True


def percentile_summary(x: np.ndarray) -> dict:
    if x is None or len(x) == 0:
        return {}
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p90": float(np.percentile(x, 90)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
        "max": float(np.max(x)),
        "argmax": int(np.argmax(x)),
    }


def verdict_from_risk(risk: float, thresholds: dict) -> str:
    if risk >= thresholds["fail"]:
        return "fail"
    if risk >= thresholds["alert"]:
        return "alert"
    if risk >= thresholds["warning"]:
        return "warning"
    return "normal"


def write_lines(path: Path, values: List[int]) -> None:
    with path.open("w") as f:
        for v in values:
            f.write(f"{v}\n")


def make_plots(out_dir: Path, event_frames: np.ndarray, rows: List[dict]) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    risk = np.array([r["risk_score"] for r in rows], dtype=float)
    state_jerk_z = np.array([r["state_jerk_z"] for r in rows], dtype=float)
    video_jump_z = np.array([r["video_jump_z"] for r in rows], dtype=float)
    joint_step_deg = np.array([r["joint_step_deg"] for r in rows], dtype=float)

    plt.figure()
    plt.plot(event_frames, risk)
    plt.xlabel("event frame")
    plt.ylabel("risk score")
    plt.title("Temporospatial risk score")
    plt.savefig(plots_dir / "risk_score.png", dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(event_frames, state_jerk_z)
    plt.xlabel("event frame")
    plt.ylabel("state jerk robust z")
    plt.title("State-space jerk z-score")
    plt.savefig(plots_dir / "state_jerk_z.png", dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(event_frames, video_jump_z)
    plt.xlabel("event frame")
    plt.ylabel("video jump robust z")
    plt.title("Video temporal jump z-score")
    plt.savefig(plots_dir / "video_jump_z.png", dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(event_frames, joint_step_deg)
    plt.xlabel("event frame")
    plt.ylabel("joint step (deg/frame)")
    plt.title("Absolute joint angular velocity")
    plt.savefig(plots_dir / "joint_step_deg.png", dpi=160, bbox_inches="tight")
    plt.close()

    return True


def main():
    parser = argparse.ArgumentParser(description="Automatic temporospatial risk evaluator.")
    parser.add_argument("--state", required=True, help="Path to state.npy, shape (T,16).")
    parser.add_argument("--video", default=None, help="Optional generated video mp4.")
    parser.add_argument("--actions", default=None, help="Optional actions.npy, shape (T,16).")
    parser.add_argument("--out", default="outputs/eval_temporospatial_risk", help="Output directory.")
    parser.add_argument("--config", default=None, help="Optional JSON config for weights/constants.")
    parser.add_argument("--joint-bounds", default=None,
                        help="Optional joint_bounds.json (compute_joint_bounds.py) providing absolute "
                             "joint_step_deg_max / joint_acc_deg_max. Overrides config hard_fail values.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Absolute joint-motion bounds (independent hard-fail axis, not a weighted signal).
    # Precedence: --joint-bounds file > config hard_fail values > disabled (None).
    joint_bounds = JointBounds(
        joint_step_deg_max=cfg["hard_fail"].get("joint_step_deg_max"),
        joint_acc_deg_max=cfg["hard_fail"].get("joint_acc_deg_max"),
    )
    if args.joint_bounds:
        jb = JointBounds.from_json(args.joint_bounds)
        if jb.joint_step_deg_max is not None:
            joint_bounds.joint_step_deg_max = jb.joint_step_deg_max
        if jb.joint_acc_deg_max is not None:
            joint_bounds.joint_acc_deg_max = jb.joint_acc_deg_max
        joint_bounds.fps = jb.fps
        joint_bounds.percentile = jb.percentile
        joint_bounds.source_files = jb.source_files

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    state_path = Path(args.state)
    state = np.load(state_path)

    summary = {
        "state_path": str(state_path),
        "video_path": args.video,
        "actions_path": args.actions,
        "config_path": args.config,
        "global_hard_fail": False,
        "global_hard_fail_reasons": [],
    }

    if state.ndim != 2 or state.shape[1] != 16:
        summary["global_hard_fail"] = True
        summary["global_hard_fail_reasons"].append(f"bad_state_shape:{state.shape}")
        with (out_dir / "summary.json").open("w") as f:
            json.dump(summary, f, indent=2)
        raise ValueError(f"Expected state shape (T,16), got {state.shape}")

    if cfg["hard_fail"].get("nan_or_inf_state", True) and not np.all(np.isfinite(state)):
        summary["global_hard_fail"] = True
        summary["global_hard_fail_reasons"].append("state_contains_nan_or_inf")

    T = state.shape[0]
    if T < 4:
        raise ValueError(f"Need at least 4 state frames for jerk, got {T}")

    arm_dims = cfg["dims"]["arm"]
    grip_dims = cfg["dims"]["grip"]
    chunk_size = int(cfg["chunk"]["size"])
    boundary_window = int(cfg["chunk"]["boundary_window"])
    z_cap = float(cfg["robust"]["z_cap"])
    eps = float(cfg["robust"]["eps"])

    arms = state[:, arm_dims]
    grips = state[:, grip_dims]

    state_step, state_acc, state_jerk = finite_diffs(arms)
    grip_step, grip_acc, grip_jerk = finite_diffs(grips)

    state_step_rms = rms(state_step)
    state_acc_rms = rms(state_acc)
    state_jerk_rms = rms(state_jerk)
    grip_jerk_rms = rms(grip_jerk)

    state_step_z = robust_z(state_step_rms, eps=eps)
    state_acc_z = robust_z(state_acc_rms, eps=eps)
    state_jerk_z = robust_z(state_jerk_rms, eps=eps)

    N = len(state_jerk_rms)
    event_frames = np.arange(N) + 2

    # Event-aligned state acceleration: jerk[i] uses acc[i] and acc[i+1].
    state_acc_event_z = np.maximum(state_acc_z[:N], state_acc_z[1:N + 1])

    # Absolute joint-motion signals (deg/frame, deg/frame^2). These are NOT
    # z-scored and NOT folded into the weighted risk; they are an independent
    # absolute hard-fail axis, same tier as nan_or_inf_state.
    # per-frame max over joints: step (T-1,), acc (T-2,).
    joint_step_max, joint_acc_max = arm_joint_motion(arms)
    # Align to event frame f=i+2: jerk[i] uses states i..i+3 -> step transitions
    # i,i+1,i+2 and acc frames i,i+1. Take the worst within that window.
    joint_step_event = np.array(
        [float(joint_step_max[i:i + 3].max()) for i in range(N)], dtype=np.float64
    )
    joint_acc_event = np.array(
        [float(joint_acc_max[i:i + 2].max()) for i in range(N)], dtype=np.float64
    )

    # Optional action features.
    action_available = False
    action_jerk_z_event = np.zeros(N, dtype=np.float64)
    if args.actions:
        action_path = Path(args.actions)
        actions = np.load(action_path)
        if actions.ndim != 2 or actions.shape[1] != 16:
            summary["global_hard_fail_reasons"].append(f"bad_actions_shape:{actions.shape}")
        else:
            action_available = True
            min_t = min(T, actions.shape[0])
            action_arms = actions[:min_t, arm_dims]
            if min_t >= 4:
                _, _, action_jerk = finite_diffs(action_arms)
                action_jerk_rms = rms(action_jerk)
                action_jerk_z = robust_z(action_jerk_rms, eps=eps)
                m = min(N, len(action_jerk_z))
                action_jerk_z_event[:m] = action_jerk_z[:m]
            else:
                summary["global_hard_fail_reasons"].append("actions_too_short_for_jerk")

    # Optional video features.
    video_available = False
    video_decode_error = None
    video_jump_z_event = np.zeros(N, dtype=np.float64)
    video_frame_count = None

    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            video_decode_error = f"video_not_found:{video_path}"
        else:
            vf, err = decode_video_gray(
                video_path,
                int(cfg["video"]["width"]),
                int(cfg["video"]["height"]),
            )
            if err:
                video_decode_error = err
            else:
                video_available = True
                video_frame_count = int(vf.shape[0])
                try:
                    vfeat = video_temporal_features(vf)
                    video_jerk_z = robust_z(vfeat["video_jerk"], eps=eps)
                    video_acc_z = robust_z(vfeat["video_acc"], eps=eps)

                    # Event frame f=i+2 aligns naturally with video_jerk[i].
                    m = min(N, len(video_jerk_z))
                    video_jump_z_event[:m] = video_jerk_z[:m]

                    # Use max(video jerk, neighboring video acc) as a simple visible jump factor.
                    for i in range(min(N, len(video_jump_z_event))):
                        candidates = [video_jump_z_event[i]]
                        if i < len(video_acc_z):
                            candidates.append(video_acc_z[i])
                        if i + 1 < len(video_acc_z):
                            candidates.append(video_acc_z[i + 1])
                        video_jump_z_event[i] = max(candidates)
                except Exception as e:
                    video_decode_error = f"video_feature_failed:{repr(e)}"

        if video_decode_error and cfg["hard_fail"].get("video_decode_failure", True):
            summary["global_hard_fail"] = True
            summary["global_hard_fail_reasons"].append(video_decode_error)

    # Score.
    weights = cfg["score"]["weights"]
    thresholds = cfg["score"]["thresholds"]
    missing_action_factor = float(cfg["score"].get("missing_action_unexplained_factor", 0.5))

    rows = []
    normal_frames, warning_frames, alert_frames, fail_frames = [], [], [], []

    hard_state_jerk_z = float(cfg["hard_fail"].get("state_jerk_z", math.inf))
    hard_video_jump_z = float(cfg["hard_fail"].get("video_jump_z", math.inf))
    hard_joint_step_deg = joint_bounds.joint_step_deg_max  # None => disabled
    hard_joint_acc_deg = joint_bounds.joint_acc_deg_max

    for i, frame in enumerate(event_frames):
        s_j = float(max(state_jerk_z[i], 0.0))
        s_a = float(max(state_acc_event_z[i], 0.0))
        a_j = float(max(action_jerk_z_event[i], 0.0))
        v_j = float(max(video_jump_z_event[i], 0.0))
        j_step = float(joint_step_event[i])
        j_acc = float(joint_acc_event[i])
        boundary = near_chunk_boundary(int(frame), chunk_size, boundary_window)

        if action_available:
            unexplained = max(0.0, s_j - a_j)
            action_status = "available"
        else:
            unexplained = missing_action_factor * s_j
            action_status = "missing"

        f_unexplained = min(unexplained, z_cap) / z_cap
        f_video = min(v_j, z_cap) / z_cap
        f_state_acc = min(s_a, z_cap) / z_cap
        f_boundary_state = (1.0 if boundary else 0.0) * min(s_j, z_cap) / z_cap
        f_state_jerk = min(s_j, z_cap) / z_cap

        risk = (
            weights["unexplained_state_jerk"] * f_unexplained
            + weights["video_jump"] * f_video
            + weights["state_acc"] * f_state_acc
            + weights["boundary_state_jerk"] * f_boundary_state
            + weights["state_jerk"] * f_state_jerk
        )

        verdict = verdict_from_risk(risk, thresholds)

        reasons = []
        if s_j >= 6:
            reasons.append("high_state_jerk")
        if s_a >= 6:
            reasons.append("high_state_acc")
        if v_j >= 6:
            reasons.append("high_video_jump")
        if action_available and unexplained >= 6:
            reasons.append("state_jerk_unexplained_by_action")
        if not action_available:
            reasons.append("action_missing")
        if not video_available:
            reasons.append("video_missing")
        if boundary:
            reasons.append("near_chunk_boundary")

        # Hard per-frame fail overrides.
        if s_j >= hard_state_jerk_z:
            verdict = "fail"
            reasons.append("hard_fail_state_jerk_z")
        if v_j >= hard_video_jump_z:
            verdict = "fail"
            reasons.append("hard_fail_video_jump_z")
        # Absolute joint-motion hard fails (raw degrees vs configured bound,
        # NOT a z-score). Distinct reason tags from the relative anomalies above.
        if hard_joint_step_deg is not None and j_step >= hard_joint_step_deg:
            verdict = "fail"
            reasons.append("hard_fail_joint_step_deg")
        if hard_joint_acc_deg is not None and j_acc >= hard_joint_acc_deg:
            verdict = "fail"
            reasons.append("hard_fail_joint_acc_deg")

        if verdict == "fail":
            fail_frames.append(int(frame))
        elif verdict == "alert":
            alert_frames.append(int(frame))
        elif verdict == "warning":
            warning_frames.append(int(frame))
        else:
            normal_frames.append(int(frame))

        rows.append({
            "jerk_index": int(i),
            "event_frame": int(frame),
            "verdict": verdict,
            "risk_score": float(risk),
            "state_jerk_rms": float(state_jerk_rms[i]),
            "state_jerk_z": float(s_j),
            "state_acc_z": float(s_a),
            "action_jerk_z": float(a_j),
            "unexplained_state_jerk_z": float(unexplained),
            "video_jump_z": float(v_j),
            "joint_step_deg": float(j_step),
            "joint_acc_deg": float(j_acc),
            "near_chunk_boundary": bool(boundary),
            "grip_jerk_rms": float(grip_jerk_rms[i]),
            "reasons": "|".join(reasons),
        })

    # Write CSV.
    csv_path = out_dir / "frame_scores.csv"
    with csv_path.open("w", newline="") as f:
        fieldnames = [
            "jerk_index",
            "event_frame",
            "verdict",
            "risk_score",
            "state_jerk_rms",
            "state_jerk_z",
            "state_acc_z",
            "action_jerk_z",
            "unexplained_state_jerk_z",
            "video_jump_z",
            "joint_step_deg",
            "joint_acc_deg",
            "near_chunk_boundary",
            "grip_jerk_rms",
            "reasons",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    write_lines(out_dir / "normal_frames.txt", normal_frames)
    write_lines(out_dir / "warning_frames.txt", warning_frames)
    write_lines(out_dir / "alert_frames.txt", alert_frames)
    write_lines(out_dir / "fail_frames.txt", fail_frames)

    # Top risk clips.
    clips_written = []
    if args.video and Path(args.video).exists():
        clips_dir = out_dir / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)

        fps = float(cfg["clips"]["fps"])
        back = int(cfg["clips"]["back"])
        forward = int(cfg["clips"]["forward"])
        top_k = int(cfg["clips"]["top_k"])

        top_rows = sorted(rows, key=lambda r: r["risk_score"], reverse=True)[:top_k]
        for rank, r in enumerate(top_rows, start=1):
            frame = int(r["event_frame"])
            start = max(0, frame - back)
            end = min(T, frame + forward)
            out_clip = clips_dir / f"rank{rank:02d}_{r['verdict']}_frame{frame:04d}.mp4"
            try:
                ok = extract_clip(Path(args.video), out_clip, start, end, fps)
                if ok:
                    clips_written.append({
                        "rank": rank,
                        "frame": frame,
                        "path": str(out_clip),
                        "start_frame": start,
                        "end_frame_exclusive": end,
                        "verdict": r["verdict"],
                        "risk_score": r["risk_score"],
                    })
            except subprocess.CalledProcessError:
                pass

    plots_written = make_plots(out_dir, event_frames, rows)

    # Summary.
    risk_values = np.array([r["risk_score"] for r in rows], dtype=float)
    verdict_counts = {
        "normal": len(normal_frames),
        "warning": len(warning_frames),
        "alert": len(alert_frames),
        "fail": len(fail_frames),
    }

    top10 = sorted(rows, key=lambda r: r["risk_score"], reverse=True)[:10]

    summary.update({
        "state_shape": list(state.shape),
        "action_available": action_available,
        "video_available": video_available,
        "video_decode_error": video_decode_error,
        "video_frame_count": video_frame_count,
        "event_frame_definition": "jerk[i] uses state[i:i+4], assigned to event_frame=i+2",
        "num_event_frames": int(N),
        "verdict_counts": verdict_counts,
        "risk_score": percentile_summary(risk_values),
        "state_step_rms": percentile_summary(state_step_rms),
        "state_acc_rms": percentile_summary(state_acc_rms),
        "state_jerk_rms": percentile_summary(state_jerk_rms),
        "state_jerk_z": percentile_summary(np.maximum(state_jerk_z, 0.0)),
        "grip_jerk_rms": percentile_summary(grip_jerk_rms),
        "joint_step_deg": percentile_summary(joint_step_event),
        "joint_acc_deg": percentile_summary(joint_acc_event),
        "joint_bounds": joint_bounds.to_dict(),
        "num_hard_fail_joint_step": sum(1 for r in rows if "hard_fail_joint_step_deg" in r["reasons"]),
        "num_hard_fail_joint_acc": sum(1 for r in rows if "hard_fail_joint_acc_deg" in r["reasons"]),
        "top10_by_risk": top10,
        "clips_written": clips_written,
        "plots_written": plots_written,
    })

    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    with (out_dir / "config_used.json").open("w") as f:
        json.dump(cfg, f, indent=2)

    # Human-readable report.
    with (out_dir / "report.md").open("w") as f:
        f.write("# Temporospatial Risk Evaluation Report\n\n")
        f.write(f"- State: `{state_path}`\n")
        f.write(f"- State shape: `{state.shape}`\n")
        f.write(f"- Video: `{args.video}`\n")
        f.write(f"- Actions: `{args.actions}`\n")
        f.write(f"- Action available: `{action_available}`\n")
        f.write(f"- Video available: `{video_available}`\n")
        f.write(f"- Global hard fail: `{summary['global_hard_fail']}`\n")
        f.write(f"- Global hard fail reasons: `{summary['global_hard_fail_reasons']}`\n\n")

        f.write("## Verdict counts\n\n")
        for k, v in verdict_counts.items():
            f.write(f"- {k}: {v}\n")

        f.write("\n## Top 10 by risk\n\n")
        f.write("| rank | frame | verdict | risk | state_jerk_z | action_jerk_z | video_jump_z | boundary | reasons |\n")
        f.write("|---:|---:|---|---:|---:|---:|---:|---|---|\n")
        for rank, r in enumerate(top10, start=1):
            f.write(
                f"| {rank} | {r['event_frame']} | {r['verdict']} | "
                f"{r['risk_score']:.4f} | {r['state_jerk_z']:.3f} | "
                f"{r['action_jerk_z']:.3f} | {r['video_jump_z']:.3f} | "
                f"{r['near_chunk_boundary']} | {r['reasons']} |\n"
            )

        f.write("\n## Notes\n\n")
        f.write("- `fail` is automatic only when weighted risk or hard-fail rules exceed configured thresholds.\n")
        f.write("- `action_missing` means unexplained jerk is approximated using the configured missing-action factor.\n")
        f.write("- `video_missing` means risk does not include visual discontinuity evidence.\n")
        f.write(
            "- `hard_fail_joint_step_deg` / `hard_fail_joint_acc_deg` are ABSOLUTE "
            "joint-motion limits (deg/frame, deg/frame^2), independent of the "
            f"relative z-score risk. Bounds: step_max={joint_bounds.joint_step_deg_max}, "
            f"acc_max={joint_bounds.joint_acc_deg_max} "
            "(None = disabled; supply via --joint-bounds or config hard_fail).\n"
        )

    print("\n=== Temporospatial Risk Eval ===")
    print(f"state: {state_path}")
    print(f"shape: {state.shape}")
    print(f"out: {out_dir}")
    print(f"action_available: {action_available}")
    print(f"video_available: {video_available}")
    print(f"verdict_counts: {verdict_counts}")
    print(f"joint_bounds: step_max={joint_bounds.joint_step_deg_max} acc_max={joint_bounds.joint_acc_deg_max}")
    print(f"hard_fail_joint_step frames: {summary['num_hard_fail_joint_step']}  "
          f"hard_fail_joint_acc frames: {summary['num_hard_fail_joint_acc']}")
    print(f"top_risk_frame: {top10[0]['event_frame'] if top10 else None}")
    print(f"top_risk_score: {top10[0]['risk_score'] if top10 else None}")
    print(f"top_verdict: {top10[0]['verdict'] if top10 else None}")
    print("\nWrote:")
    print(f"- {out_dir / 'summary.json'}")
    print(f"- {out_dir / 'report.md'}")
    print(f"- {out_dir / 'frame_scores.csv'}")
    print(f"- {out_dir / 'normal_frames.txt'}")
    print(f"- {out_dir / 'warning_frames.txt'}")
    print(f"- {out_dir / 'alert_frames.txt'}")
    print(f"- {out_dir / 'fail_frames.txt'}")
    print(f"- {out_dir / 'config_used.json'}")
    if clips_written:
        print(f"- {out_dir / 'clips'}")
    if plots_written:
        print(f"- {out_dir / 'plots'}")


if __name__ == "__main__":
    main()
