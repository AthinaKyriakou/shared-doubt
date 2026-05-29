#!/bin/bash
# 10_verbal_confidence_baseline.sh — compute verbalised uncertainty for a
# specific (model, dataset) across all datasplits and languages defined
# in constants.py.
#
# Usage:
#   bash 10_verbal_confidence_baseline.sh --model llama_3.1_8B --dataset global_mmlu
#   bash 10_verbal_confidence_baseline.sh --model qwen3_8B --dataset mkqa --dry-run
#
# Logs: each run appends to logs/verb_unc_<model>_<dataset>_<split>_<lang>.log

set -euo pipefail

DRY_RUN=0
MODEL=""
DATASET=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)   DRY_RUN=1; shift ;;
        --model)     MODEL="$2"; shift 2 ;;
        --dataset)   DATASET="$2"; shift 2 ;;
        *)           echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$MODEL" || -z "$DATASET" ]]; then
    echo "Usage: bash 10_verbal_confidence_baseline.sh --model <model> --dataset <dataset> [--dry-run]"
    exit 1
fi

mkdir -p logs

# ── Read splits and languages from constants.py for the given dataset ────────

JOBS=$(python3 - "$MODEL" "$DATASET" << 'PYEOF'
import sys
sys.path.insert(0, "../src")
from constants import (
    MODELS, DATASETS, LANGUAGES,
    DATASPLITS_GMMLU, DATASPLITS_MKQA,
)

model, dataset = sys.argv[1], sys.argv[2]

if model not in MODELS:
    print(f"ERROR: unknown model '{model}'. Valid: {MODELS}", file=sys.stderr)
    sys.exit(1)
if dataset not in DATASETS:
    print(f"ERROR: unknown dataset '{dataset}'. Valid: {DATASETS}", file=sys.stderr)
    sys.exit(1)

dl = dataset.lower()
if "mmlu" in dl:
    splits = DATASPLITS_GMMLU
elif "mkqa" in dl:
    splits = DATASPLITS_MKQA
else:
    print(f"ERROR: cannot infer splits for dataset '{dataset}'", file=sys.stderr)
    sys.exit(1)

for split in splits:
    for lang in LANGUAGES:
        print(f"{model} {dataset} {split} {lang}")
PYEOF
)

if [[ -z "$JOBS" ]]; then
    echo "No jobs generated — check model/dataset names and that constants.py is importable."
    exit 1
fi

TOTAL=$(echo "$JOBS" | wc -l)
COUNT=0
FAILED=0

echo "=== Verbalised Uncertainty runner ==="
echo "Model      : $MODEL"
echo "Dataset    : $DATASET"
echo "Total jobs : $TOTAL"
echo "Dry run    : $DRY_RUN"
echo ""

while IFS=' ' read -r model dataset split lang; do
    COUNT=$(( COUNT + 1 ))
    LOG="logs/verb_unc_${model}_${dataset}_${split}_${lang}.log"

    CMD="python3 ../src/verbal_confidence_baseline.py compute_verb_unc \
        --model-name     $model \
        --dataset-name   $dataset \
        --datasplit-name $split \
        --lang           $lang"

    echo "[${COUNT}/${TOTAL}] $model | $dataset | $split | $lang"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "  DRY-RUN: $CMD"
        continue
    fi

    if $CMD >> "$LOG" 2>&1; then
        echo "  OK  → $LOG"
    else
        echo "  FAILED (exit $?) → $LOG"
        FAILED=$(( FAILED + 1 ))
    fi

done <<< "$JOBS"

echo ""
echo "=== Done: $COUNT jobs, $FAILED failed ==="
if [[ "$FAILED" -gt 0 ]]; then
    exit 1
fi