# RoboCasa identifiability launch recovery

- Date: `2026-08-29`
- Experiment: `CWAM-EMPIRICAL-IDENTIFIABILITY-R5-20260828`
- Status: `in_progress`

The private-group attempt exited before evaluation because the readonly staged
DeMiAn path was reassigned as a child-process environment variable. Rename only
the parent-shell path, preserve the fixed S1 checkpoint and protocol, and add a
launch-contract regression test before resubmission.
