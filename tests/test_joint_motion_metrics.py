"""Tests for the absolute joint-motion metrics (tools/joint_motion_metrics.py).

The GE-Sim-V2 (T,16) state is joint space (radians), so the absolute axis is
joint angular velocity/acceleration, not a rotation geodesic. These tests pin
the sanity properties the acceptance criteria call for.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# tools/ is not an installed package; add it to the path like the scripts do.
TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from joint_motion_metrics import (  # noqa: E402
    JointBounds,
    arm_joint_motion,
    joint_acc_deg,
    joint_step_deg,
    per_frame_max,
)


def test_static_trajectory_reads_zero():
    """A rollout with no motion should read ~0 deg/frame."""
    arms = np.tile(np.linspace(-1.0, 1.0, 14), (30, 1)).astype(np.float32)  # (30,14) constant
    step, acc = arm_joint_motion(arms)
    assert step.shape == (29,)
    assert acc.shape == (28,)
    assert np.allclose(step, 0.0)
    assert np.allclose(acc, 0.0)


def test_constant_velocity_step_and_zero_acc():
    """Uniform 0.01 rad/frame ramp on one joint -> constant deg/frame, zero acc."""
    T, J = 20, 14
    arms = np.zeros((T, J), dtype=np.float64)
    arms[:, 3] = 0.01 * np.arange(T)  # 0.01 rad/frame on joint 3
    step, acc = arm_joint_motion(arms)
    expected_deg = np.degrees(0.01)
    assert np.allclose(step, expected_deg)
    assert np.allclose(acc, 0.0, atol=1e-9)


def test_degrees_conversion_and_max_over_joints():
    """per-frame value is the max-magnitude joint, converted rad->deg."""
    arms = np.zeros((3, 14), dtype=np.float64)
    arms[1, 0] = np.radians(2.0)   # joint 0 moves 2 deg on frame 0->1
    arms[1, 5] = np.radians(5.0)   # joint 5 moves 5 deg on frame 0->1 (the max)
    step = joint_step_deg(arms)
    assert step.shape == (2, 14)
    assert pytest.approx(per_frame_max(step)[0], abs=1e-6) == 5.0


def test_no_double_cover_or_wrap_artifact_for_joints():
    """Joint angles are a plain vector: component-wise diff is exact, unlike a
    quaternion where q and -q would need special handling. A joint returning to
    the same value yields zero step (no sign-flip artifact)."""
    arms = np.zeros((4, 14), dtype=np.float64)
    arms[1, 2] = 0.5
    arms[2, 2] = 0.0   # back to origin
    arms[3, 2] = 0.0
    step = per_frame_max(joint_step_deg(arms))
    assert pytest.approx(step[0]) == np.degrees(0.5)
    assert pytest.approx(step[1]) == np.degrees(0.5)
    assert pytest.approx(step[2]) == 0.0


def test_injected_supra_threshold_step_is_detectable():
    """A single joint jump above a bound must exceed it while a quiet baseline
    stays under -- the absolute check the relative z-score would miss when the
    whole trajectory is quiet except one legit-looking frame is exactly this."""
    arms = np.zeros((10, 14), dtype=np.float64)
    arms[5:, 4] = np.radians(20.0)  # 20 deg jump at frame 4->5 on joint 4
    step, _ = arm_joint_motion(arms)
    bound = 10.0
    assert step.max() >= bound
    assert (step >= bound).sum() == 1


def test_acc_is_signed_first_difference():
    step = np.array([[1.0], [3.0], [2.0]])  # (3,1)
    acc = joint_acc_deg(step)
    assert acc.shape == (2, 1)
    assert acc[0, 0] == pytest.approx(2.0)
    assert acc[1, 0] == pytest.approx(-1.0)


def test_bounds_roundtrip_and_disabled_default():
    jb = JointBounds()
    assert jb.joint_step_deg_max is None and jb.joint_acc_deg_max is None
    d = {"joint_step_deg_max": 2.5, "joint_acc_deg_max": 1.3, "fps": 16, "percentile": 99.9}
    jb2 = JointBounds.from_dict(d)
    assert jb2.joint_step_deg_max == 2.5
    assert jb2.to_dict()["joint_acc_deg_max"] == 1.3
