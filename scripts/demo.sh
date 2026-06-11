#!/usr/bin/env bash
# End-to-end demo: baseline -> regressed candidate -> CI gate catches it -> drift.
set -e
cd "$(dirname "$0")/.."
D="examples/datasets/support_triage.yaml"
RUN="${REGRESSLM_BIN:-regresslm}"

rm -rf .regresslm

echo "### 1. Baseline run (good classifier v1) ###"
$RUN run --dataset "$D" --target examples.targets:triage_v1 --scorers label_match --model v1 --run-id run-v1
$RUN baseline support_triage run-v1

echo; echo "### 2. Candidate run (v2 — a prompt change broke billing) ###"
$RUN run --dataset "$D" --target examples.targets:triage_v2_regressed --scorers label_match --model v2 --run-id run-v2

echo; echo "### 3. CI gate (exits non-zero -> fails the build) ###"
$RUN gate --dataset support_triage --candidate run-v2 || echo ">> gate failed as expected (exit $?)"

echo; echo "### 4. Drift across runs/models ###"
$RUN drift support_triage
