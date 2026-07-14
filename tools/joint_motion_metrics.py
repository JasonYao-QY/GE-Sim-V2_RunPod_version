"""Absolute joint-motion metrics for GE-Sim-V2 (T,16) joint-space states.

Single source of truth, imported by both evaluators.

The (T,16) state the evaluators grade is *joint space*, not an EEF pose:
WM layout ``[L7_arm, L_grip, R7_arm, R_grip]``, each 7-block being seven
revolute joint angles in **radians** (confirmed in gesim.types / gesim.episode
and empirically: the dims[3:7] "quaternion" sub-block is not unit-norm). So the
absolute physical-plausibility axis here is joint angular velocity /
acceleration, not a rotation geodesic. Component-wise differencing is correct
for joint angles (each dim is an independent joint; no q/-q double cover).

Caveat: this assumes the arm joints do not wrap at +/-pi within the trajectory
(true for the bundled Genie-01 arm's limited revolute joints). Continuous joints
that wrap would need angle unwrapping first.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

# Default WM-layout arm/grip indices (matches both evaluators' defaults).
DEFAULT_ARM_DIMS: list[int] = list(range(0, 7)) + list(range(8, 15))
DEFAULT_GRIP_DIMS: list[int] = [7, 15]


def joint_step_deg(joint_traj: np.ndarray) -> np.ndarray:
    """joint_traj: (T, J) radians -> (T-1, J) absolute deg/frame per joint."""
    joint_traj = np.asarray(joint_traj, dtype=np.float64)
    return np.degrees(np.abs(joint_traj[1:] - joint_traj[:-1]))


def joint_acc_deg(step_deg: np.ndarray) -> np.ndarray:
    """step_deg: (T-1, J) -> (T-2, J) signed deg/frame^2 (first diff of step)."""
    step_deg = np.asarray(step_deg, dtype=np.float64)
    return step_deg[1:] - step_deg[:-1]


def per_frame_max(x: np.ndarray) -> np.ndarray:
    """Reduce a (N, J) per-joint array to (N,) taking the max magnitude joint."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        return np.abs(x)
    return np.max(np.abs(x), axis=1)


@dataclass
class JointBounds:
    """Absolute joint-motion limits, loaded from config/JSON (never inlined).

    A value of ``None`` disables that check. ``fps`` and ``percentile`` are
    provenance only; the scoring logic uses the two *_max fields.
    """

    joint_step_deg_max: float | None = None
    joint_acc_deg_max: float | None = None
    fps: float | None = None
    percentile: float | None = None
    source_files: list[str] = field(default_factory=list)

    @classmethod
    def from_json(cls, path: str) -> JointBounds:
        with open(path) as f:
            d = json.load(f)
        return cls.from_dict(d)

    @classmethod
    def from_dict(cls, d: dict) -> JointBounds:
        def _f(v):
            return None if v is None else float(v)

        return cls(
            joint_step_deg_max=_f(d.get("joint_step_deg_max")),
            joint_acc_deg_max=_f(d.get("joint_acc_deg_max")),
            fps=_f(d.get("fps")),
            percentile=_f(d.get("percentile")),
            source_files=list(d.get("source_files", [])),
        )

    def to_dict(self) -> dict:
        return {
            "joint_step_deg_max": self.joint_step_deg_max,
            "joint_acc_deg_max": self.joint_acc_deg_max,
            "fps": self.fps,
            "percentile": self.percentile,
            "source_files": self.source_files,
        }


def arm_joint_motion(arms: np.ndarray):
    """Convenience: (T, J) arm joints -> (per-frame step deg, per-frame acc deg).

    - step_max_per_frame: (T-1,)  max-over-joints absolute deg/frame
    - acc_max_per_frame:  (T-2,)  max-over-joints absolute deg/frame^2
    """
    step = joint_step_deg(arms)          # (T-1, J)
    acc = joint_acc_deg(step)            # (T-2, J)
    return per_frame_max(step), per_frame_max(acc)
