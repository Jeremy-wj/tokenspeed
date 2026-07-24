#!/usr/bin/env bash
# Compatibility wrapper for the validated gpt-oss-120B MI350X profile.
set -euo pipefail

source "$(dirname "$0")/profiles/ar_rmsnorm/gpt_oss_120b_mi350x.env"
exec "$(dirname "$0")/e2e_arnorm_bench.sh" "$@"
