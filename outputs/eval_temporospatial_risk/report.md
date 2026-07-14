# Temporospatial Risk Evaluation Report

- State: `outputs/replay/state.npy`
- State shape: `(450, 16)`
- Video: `outputs/replay/video_browser.mp4`
- Actions: `None`
- Action available: `False`
- Video available: `True`
- Global hard fail: `False`
- Global hard fail reasons: `[]`

## Verdict counts

- normal: 413
- warning: 15
- alert: 7
- fail: 12

## Top 10 by risk

| rank | frame | verdict | risk | state_jerk_z | action_jerk_z | video_jump_z | boundary | reasons |
|---:|---:|---|---:|---:|---:|---:|---|---|
| 1 | 250 | fail | 0.8446 | 11.121 | 0.000 | 14.650 | True | high_state_jerk|high_state_acc|high_video_jump|action_missing|near_chunk_boundary |
| 2 | 76 | fail | 0.7124 | 7.959 | 0.000 | 8.965 | True | high_state_jerk|high_state_acc|high_video_jump|action_missing|near_chunk_boundary |
| 3 | 50 | alert | 0.6861 | 7.756 | 0.000 | 14.076 | True | high_state_jerk|high_state_acc|high_video_jump|action_missing|near_chunk_boundary |
| 4 | 225 | alert | 0.6834 | 6.938 | 0.000 | 9.583 | True | high_state_jerk|high_state_acc|high_video_jump|action_missing|near_chunk_boundary |
| 5 | 75 | alert | 0.6768 | 7.123 | 0.000 | 8.965 | True | high_state_jerk|high_state_acc|high_video_jump|action_missing|near_chunk_boundary |
| 6 | 275 | alert | 0.6549 | 6.639 | 0.000 | 12.732 | True | high_state_jerk|high_state_acc|high_video_jump|action_missing|near_chunk_boundary |
| 7 | 251 | alert | 0.6338 | 5.502 | 0.000 | 13.541 | True | high_state_acc|high_video_jump|action_missing|near_chunk_boundary |
| 8 | 249 | fail | 0.6190 | 5.153 | 0.000 | 15.194 | True | high_state_acc|high_video_jump|action_missing|near_chunk_boundary|hard_fail_video_jump_z |
| 9 | 100 | fail | 0.6039 | 5.832 | 0.000 | 15.624 | True | high_state_acc|high_video_jump|action_missing|near_chunk_boundary|hard_fail_video_jump_z |
| 10 | 276 | alert | 0.5806 | 4.891 | 0.000 | 12.732 | True | high_state_acc|high_video_jump|action_missing|near_chunk_boundary |

## Notes

- `fail` is automatic only when weighted risk or hard-fail rules exceed configured thresholds.
- `action_missing` means unexplained jerk is approximated using the configured missing-action factor.
- `video_missing` means risk does not include visual discontinuity evidence.
