#!/usr/bin/env bash

DATASET='mkqa'
MODEL='llama_3.1_8B'

echo "========================================"
echo "Merging hidden states extraction for: $DATASET"
echo "========================================"

python ../src/hidden_states.py \
  merge_hidden_states \
  --model-name "$MODEL" \
  --dataset-name "$DATASET"