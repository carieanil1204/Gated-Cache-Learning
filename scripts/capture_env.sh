#!/usr/bin/env bash
# Usage: scripts/capture_env.sh <run_id>
# Writes env.txt and commit.txt into results/<run_id>/, per the
# reproducibility checklist (environment capture + commit versioning).
set -euo pipefail

run_id="${1:?usage: capture_env.sh <run_id>}"
out_dir="results/${run_id}"
mkdir -p "${out_dir}"

pip freeze > "${out_dir}/env.txt"

commit="$(git rev-parse HEAD)"
if ! git diff --quiet || ! git diff --cached --quiet; then
    commit="${commit} (dirty tree — uncommitted changes present, results not trustworthy for reporting)"
fi
echo "${commit}" > "${out_dir}/commit.txt"

echo "Captured environment and commit for run ${run_id} -> ${out_dir}"
