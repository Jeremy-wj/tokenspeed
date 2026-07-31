# MI350X profile-guided follow-up

Date: 2026-07-24

Legacy pre-rebase GPT-OSS-120B TP=4 evidence from the older serving profile.
It is retained because it corrected the trace comparison and established that
prefill used two-shot calls, but it is not deployment evidence for profile v4
or any post-`3f88dcc2` runtime.

Key artifacts:

- `corrected_profile_comparison.json` — corrected kernel counts, target medians,
  and GPU-window spans;
- `e2e_summary.json` — two-seed serving comparison and rejected block override;
- `graph_replay_summary.json` — isolated graph evidence;
- remaining JSON/CSV files — supporting width, block, and segmentation screens.

The live deployment decision is
[GPT-OSS-120B status](../../../docs/gpt-oss-120b-status.md).
