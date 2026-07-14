#!/usr/bin/env python3
import argparse
import csv
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np

from joint_motion_metrics import JointBounds, arm_joint_motion


ARM_DIMS = list(range(0, 7)) + list(range(8, 15))
GRIP_DIMS = [7, 15]


def rms(x: np.ndarray, axis=1) -> np.ndarray:
    return np.sqrt(np.mean(x ** 2, axis=axis))


def robust_z(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    mad = max(float(mad), 1e-12)
    return (x - med) / mad


def percentile_dict(x: np.ndarray) -> dict:
    if len(x) == 0:
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


def near_chunk_boundary(frame: int, chunk_size: int, window: int) -> bool:
    """
    frame is a state-frame index.
    Boundaries are between chunk_size-1 | chunk_size,
    2*chunk_size-1 | 2*chunk_size, etc.
    We mark frames within +/- window of either side.
    """
    if chunk_size <= 0:
        return False

    # nearest boundary frame on the right side: 25, 50, 75...
    r = frame % chunk_size
    dist_to_right_boundary = min(r, chunk_size - r)
    return dist_to_right_boundary <= window or abs(r - (chunk_size - 1)) <= window


def write_lines(path: Path, values) -> None:
    with path.open("w") as f:
        for v in values:
            f.write(f"{v}\n")


def extract_clip_ffmpeg(video_path: Path, out_path: Path, start_frame: int, end_frame_exclusive: int, fps: float) -> bool:
    """
    Extract exact frame interval [start_frame, end_frame_exclusive).
    Writes a browser-compatible H.264 mp4.

    Requires ffmpeg in PATH.
    """
    if shutil.which("ffmpeg") is None:
        print("[warn] ffmpeg not found; skipping clip extraction.")
        return False

    end_inclusive = end_frame_exclusive - 1
    if end_inclusive < start_frame:
        print("[warn] invalid clip frame range; skipping clip extraction.")
        return False

    select_expr = f"select='between(n\\,{start_frame}\\,{end_inclusive})',setpts=N/FRAME_RATE/TB"

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vf", select_expr,
        "-an",
        "-r", str(fps),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(out_path),
    ]

    print("[info] running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate temporal continuity of GE-Sim predicted (T,16) state trajectory."
    )
    parser.add_argument("--state", required=True, help="Path to state.npy, shape (T,16).")
    parser.add_argument("--video", default=None, help="Optional generated video mp4 for alert clip extraction.")
    parser.add_argument("--out", default="outputs/eval_state_continuity", help="Output directory.")
    parser.add_argument("--fps", type=float, default=16.0, help="Video FPS.")
    parser.add_argument("--chunk-size", type=int, default=25, help="Model chunk size.")
    parser.add_argument("--boundary-window", type=int, default=2, help="Frames near chunk boundary to mark.")
    parser.add_argument("--fail-z", type=float, default=8.0, help="Fail threshold on robust z-score of arm jerk RMS.")
    parser.add_argument("--warn-z", type=float, default=6.0, help="Warn threshold on robust z-score of arm jerk RMS.")
    parser.add_argument("--joint-step-max-deg", type=float, default=None,
                        help="Absolute per-joint angular velocity limit (deg/frame). None disables.")
    parser.add_argument("--joint-acc-max-deg", type=float, default=None,
                        help="Absolute per-joint angular acceleration limit (deg/frame^2). None disables.")
    parser.add_argument("--joint-bounds", default=None,
                        help="Optional joint_bounds.json; fills any --joint-*-max-deg left unset.")
    parser.add_argument("--clip-back", type=int, default=5, help="Frames before alert frame.")
    parser.add_argument("--clip-forward", type=int, default=5, help="Frames after alert frame; exclusive interval gives 10 frames by default.")
    args = parser.parse_args()

    state_path = Path(args.state)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    state = np.load(state_path)
    if state.ndim != 2 or state.shape[1] != 16:
        raise ValueError(f"Expected state shape (T,16), got {state.shape}")

    T = state.shape[0]
    if T < 4:
        raise ValueError(f"Need at least 4 frames for jerk; got T={T}")

    arms = state[:, ARM_DIMS]
    grippers = state[:, GRIP_DIMS]

    # Finite differences.
    arm_step = arms[1:] - arms[:-1]                                  # step[i] uses state i,i+1
    arm_acc = arms[2:] - 2 * arms[1:-1] + arms[:-2]                  # acc[i] uses state i,i+1,i+2
    arm_jerk = arms[3:] - 3 * arms[2:-1] + 3 * arms[1:-2] - arms[:-3] # jerk[i] uses state i..i+3

    grip_step = grippers[1:] - grippers[:-1]
    grip_acc = grippers[2:] - 2 * grippers[1:-1] + grippers[:-2]
    grip_jerk = grippers[3:] - 3 * grippers[2:-1] + 3 * grippers[1:-2] - grippers[:-3]

    arm_step_rms = rms(arm_step)
    arm_acc_rms = rms(arm_acc)
    arm_jerk_rms = rms(arm_jerk)

    grip_step_rms = rms(grip_step)
    grip_acc_rms = rms(grip_acc)
    grip_jerk_rms = rms(grip_jerk)

    jerk_z = robust_z(arm_jerk_rms)

    # Absolute joint-motion signals (deg/frame, deg/frame^2). Independent of the
    # relative z-score: a frame fails if it crosses the z threshold OR the
    # absolute degree threshold, each reported separately.
    # Precedence for the bounds: explicit --joint-*-max-deg > --joint-bounds file.
    jstep_max = args.joint_step_max_deg
    jacc_max = args.joint_acc_max_deg
    joint_bounds_meta = None
    if args.joint_bounds:
        jb = JointBounds.from_json(args.joint_bounds)
        joint_bounds_meta = jb.to_dict()
        if jstep_max is None:
            jstep_max = jb.joint_step_deg_max
        if jacc_max is None:
            jacc_max = jb.joint_acc_deg_max

    joint_step_pf, joint_acc_pf = arm_joint_motion(arms)  # (T-1,), (T-2,)

    N = len(arm_jerk_rms)
    # jerk[i] uses states i, i+1, i+2, i+3.
    # We assign it to event_frame = i + 2.
    # Example: jerk[48] covers states 48..51 and is centered around the 49|50 transition.
    event_frames = np.arange(N) + 2

    # Align absolute signals to event frame i+2 (worst within the jerk window).
    joint_step_event = np.array([float(joint_step_pf[i:i + 3].max()) for i in range(N)])
    joint_acc_event = np.array([float(joint_acc_pf[i:i + 2].max()) for i in range(N)])

    statuses = []
    flagged_by = []
    fail_frames = []
    warn_frames = []
    pass_frames = []

    for idx, frame in enumerate(event_frames):
        z = float(jerk_z[idx])
        js = float(joint_step_event[idx])
        ja = float(joint_acc_event[idx])

        rel_fail = z >= args.fail_z
        rel_warn = z >= args.warn_z
        abs_step_fail = jstep_max is not None and js >= jstep_max
        abs_acc_fail = jacc_max is not None and ja >= jacc_max

        flags = []
        if rel_fail:
            flags.append("relative_z")
        if abs_step_fail:
            flags.append("absolute_joint_step")
        if abs_acc_fail:
            flags.append("absolute_joint_acc")

        if rel_fail or abs_step_fail or abs_acc_fail:
            status = "fail"
            fail_frames.append(int(frame))
        elif rel_warn:
            status = "warn"
            warn_frames.append(int(frame))
        else:
            status = "pass"
            pass_frames.append(int(frame))
        statuses.append(status)
        flagged_by.append("|".join(flags) if flags else "none")

    worst_idx = int(np.argmax(arm_jerk_rms))
    worst_frame = int(event_frames[worst_idx])

    # Write per-frame CSV.
    csv_path = out_dir / "frame_scores.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "jerk_index",
                "event_frame",
                "status",
                "flagged_by",
                "near_chunk_boundary",
                "arm_jerk_rms",
                "arm_jerk_robust_z",
                "joint_step_deg",
                "joint_acc_deg",
                "arm_acc_rms_at_frame_minus_2_if_available",
                "arm_step_rms_at_frame_minus_2_if_available",
                "grip_jerk_rms",
            ],
        )
        writer.writeheader()
        for idx, frame in enumerate(event_frames):
            acc_idx = frame - 2
            step_idx = frame - 2
            writer.writerow({
                "jerk_index": idx,
                "event_frame": int(frame),
                "status": statuses[idx],
                "flagged_by": flagged_by[idx],
                "near_chunk_boundary": near_chunk_boundary(int(frame), args.chunk_size, args.boundary_window),
                "arm_jerk_rms": float(arm_jerk_rms[idx]),
                "arm_jerk_robust_z": float(jerk_z[idx]),
                "joint_step_deg": float(joint_step_event[idx]),
                "joint_acc_deg": float(joint_acc_event[idx]),
                "arm_acc_rms_at_frame_minus_2_if_available": float(arm_acc_rms[acc_idx]) if 0 <= acc_idx < len(arm_acc_rms) else "",
                "arm_step_rms_at_frame_minus_2_if_available": float(arm_step_rms[step_idx]) if 0 <= step_idx < len(arm_step_rms) else "",
                "grip_jerk_rms": float(grip_jerk_rms[idx]),
            })

    write_lines(out_dir / "pass_frames.txt", pass_frames)
    write_lines(out_dir / "warn_frames.txt", warn_frames)
    write_lines(out_dir / "fail_frames.txt", fail_frames)

    # Worst absolute joint-step frame (may differ from the worst-jerk frame).
    worst_abs_idx = int(np.argmax(joint_step_event))
    worst_abs_frame = int(event_frames[worst_abs_idx])

    with (out_dir / "most_alerting_frame.txt").open("w") as f:
        f.write(f"most_alerting_frame={worst_frame}\n")
        f.write(f"jerk_index={worst_idx}\n")
        f.write(f"arm_jerk_rms={float(arm_jerk_rms[worst_idx])}\n")
        f.write(f"arm_jerk_robust_z={float(jerk_z[worst_idx])}\n")
        f.write(f"joint_step_deg={float(joint_step_event[worst_idx])}\n")
        f.write(f"joint_acc_deg={float(joint_acc_event[worst_idx])}\n")
        f.write(f"flagged_by={flagged_by[worst_idx]}\n")
        f.write(f"near_chunk_boundary={near_chunk_boundary(worst_frame, args.chunk_size, args.boundary_window)}\n")
        f.write("# --- worst absolute joint-step frame (independent of relative jerk) ---\n")
        f.write(f"worst_abs_joint_step_frame={worst_abs_frame}\n")
        f.write(f"worst_abs_joint_step_deg={float(joint_step_event[worst_abs_idx])}\n")
        f.write(f"worst_abs_flagged_by={flagged_by[worst_abs_idx]}\n")
        f.write(f"joint_step_max_deg={jstep_max}\n")
        f.write(f"joint_acc_max_deg={jacc_max}\n")

    boundary_mask = np.array([
        near_chunk_boundary(int(frame), args.chunk_size, args.boundary_window)
        for frame in event_frames
    ])

    summary = {
        "state_path": str(state_path),
        "state_shape": list(state.shape),
        "tested_event_frames": int(len(event_frames)),
        "event_frame_definition": "jerk[i] uses state[i:i+4], assigned to event_frame=i+2",
        "arm_dims": ARM_DIMS,
        "grip_dims": GRIP_DIMS,
        "fail_z": args.fail_z,
        "warn_z": args.warn_z,
        "joint_step_max_deg": jstep_max,
        "joint_acc_max_deg": jacc_max,
        "joint_bounds": joint_bounds_meta,
        "num_pass": len(pass_frames),
        "num_warn": len(warn_frames),
        "num_fail": len(fail_frames),
        "num_fail_relative_z": sum(1 for fb in flagged_by if "relative_z" in fb),
        "num_fail_absolute_joint_step": sum(1 for fb in flagged_by if "absolute_joint_step" in fb),
        "num_fail_absolute_joint_acc": sum(1 for fb in flagged_by if "absolute_joint_acc" in fb),
        "most_alerting_frame": worst_frame,
        "most_alerting_jerk_index": worst_idx,
        "most_alerting_arm_jerk_rms": float(arm_jerk_rms[worst_idx]),
        "most_alerting_arm_jerk_robust_z": float(jerk_z[worst_idx]),
        "most_alerting_near_chunk_boundary": near_chunk_boundary(worst_frame, args.chunk_size, args.boundary_window),
        "arm_step_rms": percentile_dict(arm_step_rms),
        "arm_acc_rms": percentile_dict(arm_acc_rms),
        "arm_jerk_rms": percentile_dict(arm_jerk_rms),
        "joint_step_deg": percentile_dict(joint_step_event),
        "joint_acc_deg": percentile_dict(joint_acc_event),
        "grip_step_rms": percentile_dict(grip_step_rms),
        "grip_acc_rms": percentile_dict(grip_acc_rms),
        "grip_jerk_rms": percentile_dict(grip_jerk_rms),
        "boundary_analysis": {
            "chunk_size": args.chunk_size,
            "boundary_window": args.boundary_window,
            "boundary_event_count": int(boundary_mask.sum()),
            "nonboundary_event_count": int((~boundary_mask).sum()),
            "boundary_arm_jerk_mean": float(arm_jerk_rms[boundary_mask].mean()) if boundary_mask.any() else None,
            "nonboundary_arm_jerk_mean": float(arm_jerk_rms[~boundary_mask].mean()) if (~boundary_mask).any() else None,
            "boundary_arm_jerk_p95": float(np.percentile(arm_jerk_rms[boundary_mask], 95)) if boundary_mask.any() else None,
            "nonboundary_arm_jerk_p95": float(np.percentile(arm_jerk_rms[~boundary_mask], 95)) if (~boundary_mask).any() else None,
        },
        "top10_arm_jerk_frames": [
            {
                "rank": rank + 1,
                "event_frame": int(event_frames[i]),
                "jerk_index": int(i),
                "arm_jerk_rms": float(arm_jerk_rms[i]),
                "arm_jerk_robust_z": float(jerk_z[i]),
                "near_chunk_boundary": near_chunk_boundary(int(event_frames[i]), args.chunk_size, args.boundary_window),
            }
            for rank, i in enumerate(np.argsort(arm_jerk_rms)[-10:][::-1])
        ],
    }

    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    # Extract 10-frame clip around worst frame.
    clip_written = False
    if args.video is not None:
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"[warn] video path not found: {video_path}")
        else:
            start_frame = max(0, worst_frame - args.clip_back)
            end_frame = min(T, worst_frame + args.clip_forward)  # exclusive; default gives 10 frames
            clip_path = out_dir / "alert_clip.mp4"
            try:
                clip_written = extract_clip_ffmpeg(
                    video_path=video_path,
                    out_path=clip_path,
                    start_frame=start_frame,
                    end_frame_exclusive=end_frame,
                    fps=args.fps,
                )
                summary["alert_clip"] = {
                    "path": str(clip_path),
                    "start_frame": int(start_frame),
                    "end_frame_exclusive": int(end_frame),
                    "fps": args.fps,
                    "written": bool(clip_written),
                }
                with (out_dir / "summary.json").open("w") as f:
                    json.dump(summary, f, indent=2)
            except subprocess.CalledProcessError as e:
                print(f"[warn] ffmpeg clip extraction failed: {e}")

    print("\n=== State Continuity Eval ===")
    print(f"state: {state_path}")
    print(f"shape: {state.shape}")
    print(f"out: {out_dir}")
    print(f"pass/warn/fail: {len(pass_frames)} / {len(warn_frames)} / {len(fail_frames)}")
    print(f"joint bounds: step_max={jstep_max} acc_max={jacc_max} (deg/frame, deg/frame^2; None=disabled)")
    print(f"fails by check: relative_z={summary['num_fail_relative_z']} "
          f"absolute_joint_step={summary['num_fail_absolute_joint_step']} "
          f"absolute_joint_acc={summary['num_fail_absolute_joint_acc']}")
    print(f"most_alerting_frame: {worst_frame}")
    print(f"most_alerting_jerk_index: {worst_idx}")
    print(f"arm_jerk_rms: {float(arm_jerk_rms[worst_idx])}")
    print(f"arm_jerk_robust_z: {float(jerk_z[worst_idx])}")
    print(f"near_chunk_boundary: {near_chunk_boundary(worst_frame, args.chunk_size, args.boundary_window)}")
    if args.video is not None:
        print(f"alert_clip_written: {clip_written}")
    print("\nWrote:")
    print(f"- {out_dir / 'summary.json'}")
    print(f"- {out_dir / 'frame_scores.csv'}")
    print(f"- {out_dir / 'pass_frames.txt'}")
    print(f"- {out_dir / 'warn_frames.txt'}")
    print(f"- {out_dir / 'fail_frames.txt'}")
    print(f"- {out_dir / 'most_alerting_frame.txt'}")
    if clip_written:
        print(f"- {out_dir / 'alert_clip.mp4'}")


if __name__ == "__main__":
    main()
