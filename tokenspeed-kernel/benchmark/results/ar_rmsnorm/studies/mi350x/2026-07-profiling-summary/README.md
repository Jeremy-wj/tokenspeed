# MI350X Proton profiling summary

Date: 2026-07-23

`proton_summary.json` preserves pre-profile-v4 eager Proton aggregates for the
old triton-shmem and native-all-reduce runtime. The traces show substantial
rank skew and include two migration-orphaned sources whose aggregate metrics
survived but whose raw traces are absent.

Use this study only for historical profiler bring-up and skew diagnosis. It is
not a post-rebase performance baseline or promotion input.
