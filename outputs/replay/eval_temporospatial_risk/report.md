# Temporospatial Risk Evaluation Report

- State: `outputs/replay/state.npy`
- State shape: `(450, 16)`
- Video: `outputs/replay/video.mp4`
- Actions: `None`
- Action available: `False`
- Video available: `True`
- Global hard fail: `False`
- Global hard fail reasons: `[]`

## Verdict counts

- normal: 380
- warning: 3
- alert: 0
- fail: 64

## Top 10 by risk

| rank | frame | verdict | risk | state_jerk_z | action_jerk_z | video_jump_z | boundary | reasons |
|---:|---:|---|---:|---:|---:|---:|---|---|
| 1 | 250 | fail | 0.8446 | 11.121 | 0.000 | 11.323 | True | high_state_jerk|high_state_acc|high_video_jump|action_missing|near_chunk_boundary|hard_fail_joint_step_deg|hard_fail_joint_acc_deg |
| 2 | 50 | fail | 0.6861 | 7.756 | 0.000 | 11.031 | True | high_state_jerk|high_state_acc|high_video_jump|action_missing|near_chunk_boundary|hard_fail_joint_step_deg|hard_fail_joint_acc_deg |
| 3 | 76 | fail | 0.6849 | 7.959 | 0.000 | 7.865 | True | high_state_jerk|high_state_acc|high_video_jump|action_missing|near_chunk_boundary|hard_fail_joint_step_deg|hard_fail_joint_acc_deg |
| 4 | 275 | fail | 0.6549 | 6.639 | 0.000 | 10.871 | True | high_state_jerk|high_state_acc|high_video_jump|action_missing|near_chunk_boundary|hard_fail_joint_step_deg|hard_fail_joint_acc_deg |
| 5 | 75 | fail | 0.6493 | 7.123 | 0.000 | 7.865 | True | high_state_jerk|high_state_acc|high_video_jump|action_missing|near_chunk_boundary|hard_fail_joint_step_deg|hard_fail_joint_acc_deg |
| 6 | 225 | fail | 0.6374 | 6.938 | 0.000 | 7.745 | True | high_state_jerk|high_state_acc|high_video_jump|action_missing|near_chunk_boundary|hard_fail_joint_step_deg|hard_fail_joint_acc_deg |
| 7 | 251 | fail | 0.6338 | 5.502 | 0.000 | 10.257 | True | high_state_acc|high_video_jump|action_missing|near_chunk_boundary|hard_fail_joint_step_deg|hard_fail_joint_acc_deg |
| 8 | 249 | fail | 0.6190 | 5.153 | 0.000 | 12.029 | True | high_state_acc|high_video_jump|action_missing|near_chunk_boundary|hard_fail_joint_step_deg|hard_fail_joint_acc_deg |
| 9 | 100 | fail | 0.6039 | 5.832 | 0.000 | 14.516 | True | high_state_acc|high_video_jump|action_missing|near_chunk_boundary|hard_fail_joint_step_deg|hard_fail_joint_acc_deg |
| 10 | 276 | fail | 0.5806 | 4.891 | 0.000 | 10.871 | True | high_state_acc|high_video_jump|action_missing|near_chunk_boundary|hard_fail_joint_step_deg|hard_fail_joint_acc_deg |

## Notes

- `fail` is automatic only when weighted risk or hard-fail rules exceed configured thresholds.
- `action_missing` means unexplained jerk is approximated using the configured missing-action factor.
- `video_missing` means risk does not include visual discontinuity evidence.
- `hard_fail_joint_step_deg` / `hard_fail_joint_acc_deg` are ABSOLUTE joint-motion limits (deg/frame, deg/frame^2), independent of the relative z-score risk. Bounds: step_max=2.568942145792971, acc_max=1.3430251610848785 (None = disabled; supply via --joint-bounds or config hard_fail).
