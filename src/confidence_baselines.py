"""
Compute generation-time UQ baselines using LM-Polygraph.

Baselines computed:
  1. Length-Normalised Sequence Likelihood (NSL):
       NSL(x) = (1/L) * sum_{t=1}^{L} log P(y_t | y_{<t}, x)
       Also saved in exponentiated form (geometric mean of token probs, in [0,1]).

  2. P(True) (Kadavath et al., 2022):
     Prompts the model with its own question and answer and measures the
     log-probability assigned to the token "True". Uses the prompt template
     from LM-Polygraph's PromptCalculator.

  3. Attention Score (Vazhentsev et al., 2024):
     Runs a forward pass on the prompt+generation, extracts attention maps at
     the middle layer, and computes the sum of log-diagonal attention weights
     averaged across heads.

For the Mass-Mean Probe baseline see mass_mean_probe.py.

Usage:
    python confidence_baselines.py \
        --model-name llama_3.1_8B \
        --dataset-name mkqa \
        --datasplit-name mkqa_answerable \
        --lang en \
        --baselines nsl p_true attention_score
"""

import argparse
import json
import logging
import math
import sys
import gc
import time
from pathlib import Path
from typing import List

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from lm_polygraph.utils.model import WhiteboxModel
from lm_polygraph.stat_calculators.greedy_probs import GreedyProbsCalculator
from lm_polygraph.stat_calculators.prompt import PromptCalculator
from lm_polygraph.stat_calculators.attention_forward_pass import AttentionForwardPassCalculator
from lm_polygraph.estimators.attention_score import AttentionScore

from constants import (
    DATASETS, MODELS, LANGUAGES,
    LLAMA_MODEL_PATH, QWEN_MODEL_PATH,
    SYS_PROMPT_FILE, DATA_DIR, LOCAL_RESULTS_DIR,
    BATCH_SIZE, MAX_NEW_TOKENS,
    DATASPLITS_GMMLU, DATASPLITS_MKQA,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)

GENERATION_BASELINES = {"nsl", "p_true", "attention_score"}


def compute_baselines(
    local_model_path: str,
    model_name: str,
    sys_prompt_file: str,
    input_file: str,
    results_dir: str,
    dataset: str,
    datasplit: str,
    lang: str,
    batch_size: int,
    max_new_tokens: int,
    baselines: List[str],
):
    """Compute one or more generation-time UQ baselines in a single pass.

    NSL, P(True), and Attention Score all share the same greedy-decoded
    answers, so running them together means:
      - One greedy generation pass per example instead of three
      - All three scores are guaranteed to be computed over the same answer
      - Answer correctness only needs to be evaluated once

    Each baseline is self-contained with respect to the probe and mass-mean
    baselines (which use sampled generations from hidden_states.py).

    Args:
        baselines: list of baselines to compute, any subset of
            ["nsl", "p_true", "attention_score"]
    """

    # ── OUTPUT FILE ───────────────────────────────────────────────────────────
    # Name encodes which baselines are in the file, in sorted order
    baselines_tag = "_".join(sorted(baselines))
    out_file = Path(results_dir) / dataset / model_name / datasplit / f"{lang}_{baselines_tag}_scores.jsonl"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # ── MODEL SETUP ───────────────────────────────────────────────────────────
    logging.info(f"Loading model from {local_model_path}")

    base_model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
        trust_remote_code=True,
    )
    base_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        local_model_path,
        local_files_only=True,
        use_fast=True,
        padding_side="left",
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Wrap in LM-Polygraph's WhiteboxModel
    model = WhiteboxModel(base_model, tokenizer, model_path=local_model_path)

    # ── LM-POLYGRAPH STAT CALCULATORS ─────────────────────────────────────────
    # GreedyProbsCalculator: always needed — generates answers and per-token log-probs
    # output_attentions=False to save GPU memory
    greedy_calc = GreedyProbsCalculator(output_attentions=False)

    # PromptCalculator: P(True) — only initialised when needed
    # BasePromptCalculator reads questions from dependencies["input_texts"] and
    # answers from dependencies["greedy_texts"]. texts= is API-only, not used internally.
    ptrue_calc = PromptCalculator() if "p_true" in baselines else None

    # AttentionForwardPassCalculator + AttentionScore — only when needed
    # Does a per-example forward pass on prompt+generation to extract attention maps.
    # When output_attentions=True is requested with attn_implementation="sdpa",
    # HuggingFace automatically falls back to eager attention with a warning —
    # attention weights are returned correctly.
    attn_fwd_calc = AttentionForwardPassCalculator() if "attention_score" in baselines else None
    attn_estimator = AttentionScore() if "attention_score" in baselines else None

    logging.info(f"Computing baselines: {baselines}")

    # ── LOAD SYSTEM PROMPT ────────────────────────────────────────────────────
    with open(sys_prompt_file, "r", encoding="utf-8") as pf:
        prompts_cfg = json.load(pf)

    if dataset == "mkqa":
        system_instruction = prompts_cfg[dataset][datasplit][lang]
    else:
        system_instruction = prompts_cfg[dataset][lang]

    # ── LOAD INPUT EXAMPLES ───────────────────────────────────────────────────
    examples = []
    with open(input_file, "r", encoding="utf-8") as fin:
        for line in fin:
            item = json.loads(line)
            if dataset == "mkqa":
                examples.append((item["example_id"], item["queries"][lang]))
            else:
                examples.append((item["sample_id"], item["question"]))

    # ── SORT BY PROMPT LENGTH (when attention_score is requested) ─────────────
    # AttentionForwardPassCalculator tokenises the full batch with padding=True,
    # then slices batch["input_ids"][i] per example — so pad tokens are present
    # in each per-example forward pass. Sorting by question length ensures
    # examples of similar length are batched together, minimising padding tokens
    # in the attention diagonal. The system prompt is fixed so question length
    # determines relative prompt length ordering.
    if "attention_score" in baselines:
        examples.sort(key=lambda x: len(tokenizer.encode(x[1], add_special_tokens=False)))
        logging.info("Sorted examples by question length to minimise within-batch padding.")

    total_batches = (len(examples) + batch_size - 1) // batch_size
    logging.info(
        f"Total examples: {len(examples)} | "
        f"Batch size: {batch_size} | "
        f"Total batches: {total_batches}"
    )
    start_total = time.time()

    # ── BATCHED SCORING ───────────────────────────────────────────────────────
    with torch.no_grad():
        for batch_num, i in enumerate(
            tqdm(
                range(0, len(examples), batch_size),
                total=total_batches,
                desc=f"[{dataset}/{datasplit}/{lang}]",
                unit="batch",
            ),
            start=1,
        ):
            t_batch_start = time.time()
            batch = examples[i : i + batch_size]
            ids, queries = zip(*batch)

            # (1) Format each query with the chat template (text only, not tokenised)
            formatted_texts = []
            for q in queries:
                msgs = [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": q},
                ]
                formatted_texts.append(
                    tokenizer.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True
                    )
                )

            # (2) Run greedy generation → per-token log-probs + generated texts
            # greedy_log_likelihoods contains true log-probs: _ScoresProcessor
            # applies log_softmax before storing scores in WhiteboxModel.generate
            batch_stats = greedy_calc(
                dependencies={},
                texts=formatted_texts,
                model=model,
                max_new_tokens=max_new_tokens,
            )

            # (3a) P(True): score the greedily generated answer
            if ptrue_calc is not None:
                ptrue_stats = ptrue_calc(
                    dependencies={
                        "greedy_texts": batch_stats["greedy_texts"],
                        "input_texts": list(queries),
                    },
                    texts=list(queries),  # kept for API compatibility, not used internally
                    model=model,
                )
                batch_stats.update(ptrue_stats)

            # (3b) Attention score: per-example forward pass on prompt + generation
            attn_scores = None
            if attn_fwd_calc is not None:
                attn_stats = attn_fwd_calc(
                    dependencies=batch_stats,
                    texts=formatted_texts,
                    model=model,
                )
                batch_stats.update(attn_stats)
                # AttentionScore reads stats["model"] to determine the middle layer
                batch_stats["model"] = model
                attn_scores = attn_estimator(batch_stats)

            # (4) Compute scores for each example and write combined record
            for idx, ex_id in enumerate(ids):
                ll = batch_stats["greedy_log_likelihoods"][idx]  # List[float]

                if len(ll) == 0:
                    logging.warning(f"Example {ex_id}: empty generation, skipping.")
                    continue

                rec = {
                    "example_id": ex_id,
                    "query": queries[idx],
                    "answer": batch_stats["greedy_texts"][idx],
                    "num_tokens": len(ll),
                }

                if "nsl" in baselines:
                    nsl = sum(ll) / len(ll)
                    rec["nsl_log"] = float(nsl)
                    rec["nsl_prob"] = float(math.exp(nsl))

                if "p_true" in baselines:
                    p_true_logprob = float(batch_stats["p_true"][idx])
                    rec["p_true_log"] = p_true_logprob
                    rec["p_true_prob"] = float(math.exp(p_true_logprob))

                if "attention_score" in baselines:
                    rec["attention_score"] = float(attn_scores[idx])

                with open(out_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            # ── PER-BATCH TIMING ──────────────────────────────────────────────
            t_batch_elapsed = time.time() - t_batch_start
            t_total_elapsed = time.time() - start_total
            avg_per_batch = t_total_elapsed / batch_num
            remaining_batches = total_batches - batch_num
            eta_seconds = avg_per_batch * remaining_batches

            logging.info(
                f"Batch {batch_num}/{total_batches} | "
                f"This batch: {t_batch_elapsed:.1f}s | "
                f"Avg/batch: {avg_per_batch:.1f}s | "
                f"ETA: {eta_seconds / 60:.1f} min"
            )

            # ── MEMORY CLEANUP ────────────────────────────────────────────────
            del batch_stats
            torch.cuda.empty_cache()
            gc.collect()

    # ── FINAL SUMMARY ─────────────────────────────────────────────────────────
    total_time = time.time() - start_total
    logging.info(
        f"\nDone. {len(examples)} examples in {total_batches} batches | "
        f"Total time: {total_time / 60:.1f} min ({total_time / 3600:.2f} h)"
    )
    logging.info(f"✅ Scores saved to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute UQ baselines via LM-Polygraph"
    )
    parser.add_argument("--model-name", type=str, choices=MODELS, required=True)
    parser.add_argument("--dataset-name", type=str, choices=DATASETS, required=True)
    parser.add_argument(
        "--datasplit-name", type=str,
        choices=DATASPLITS_GMMLU + DATASPLITS_MKQA, required=True,
    )
    parser.add_argument("--lang", type=str, choices=LANGUAGES, required=True)
    parser.add_argument(
        "--baselines", type=str, nargs="+",
        choices=["nsl", "p_true", "attention_score"], required=True,
        help=(
            "One or more generation-time baselines to compute in a single pass. "
            "All share the same greedy generation, producing one combined output file. "
            "For the Mass-Mean Probe see mass_mean_probe.py."
        ),
    )
    args = parser.parse_args()

    dataset = args.dataset_name
    datasplit = args.datasplit_name
    model_name = args.model_name
    lang = args.lang
    baselines = args.baselines

    # ── Generation-time baselines: need model + input file ────────────────────
    if dataset == "mkqa":
        input_file = f"{DATA_DIR}/{dataset}/{datasplit}.jsonl"
        max_new_tokens = (
            2 if datasplit == "mkqa_answerable_binary" and lang == "de"
            else (1 if datasplit == "mkqa_answerable_binary" else MAX_NEW_TOKENS)
        )
    elif dataset == "global_mmlu":
        input_file = f"{DATA_DIR}/{dataset}/{lang}_final/{datasplit}.jsonl"
        max_new_tokens = MAX_NEW_TOKENS
    else:
        logging.error(f"Dataset '{dataset}' not integrated")
        sys.exit(1)

    if model_name == "llama_3.1_8B":
        local_model_path = LLAMA_MODEL_PATH
    elif model_name == "qwen3_8B":
        local_model_path = QWEN_MODEL_PATH
    else:
        logging.error(f"Model '{model_name}' not integrated")
        sys.exit(1)

    compute_baselines(
        local_model_path=local_model_path,
        model_name=model_name,
        sys_prompt_file=SYS_PROMPT_FILE,
        input_file=input_file,
        results_dir=LOCAL_RESULTS_DIR,
        dataset=dataset,
        datasplit=datasplit,
        lang=lang,
        batch_size=BATCH_SIZE,
        max_new_tokens=max_new_tokens,
        baselines=baselines,
    )