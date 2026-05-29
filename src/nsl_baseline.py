"""
nsl_baseline.py — Length-Normalised Sequence Likelihood (NSL) baseline.

For each (question, answer) pair already produced by hidden_states.py, this
script computes:

    NSL(a | q) = (1/T) * Σ_{t=1}^{T} log P(a_t | a_1, …, a_{t-1}, q)

where T is the number of answer tokens.

Reference: Malinin & Gales, 2020 (sequence log-likelihood for uncertainty)
           https://github.com/IINemo/lm-polygraph — used as evaluation suite

Usage (mirrors ptrue_baseline.py):
    python nsl_baseline.py compute_nsl \
        --model-name  llama_3.1_8B \
        --dataset-name global_mmlu \
        --datasplit-name  <split> \
        --lang en

    python nsl_baseline.py compute_nsl \
        --model-name  qwen3_8B \
        --dataset-name mkqa \
        --datasplit-name  <split> \
        --lang ja \
        --track-emissions
"""

import argparse
import gc
import json
import logging
import math
import os
import sys
import time

import torch
from codecarbon import OfflineEmissionsTracker
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import find_subsequence

from constants import (
    BATCH_SIZE,
    DATASETS,
    DATASPLITS_GMMLU,
    DATASPLITS_MKQA,
    EMISSIONS_DIR,
    LANGUAGES,
    LLAMA_MODEL_PATH,
    RESULTS_DIR,
    MODELS,
    QWEN_MODEL_PATH,
    SYS_PROMPT_FILE,
)

# ── SPLIT CONFIG ──────────────────────────────────────────────────────────────
TRAIN_LANG = 'fr'
STRATIFIED = True

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)


# ── HELPERS ───────────────────────────────────────────────────────────────────

def load_test_ids(results_dir: str, dataset: str, model_name: str,
                  datasplit: str) -> set:
    """
    Load the test-split example IDs from the splits JSON file — same path
    logic as ptrue_baseline.py to guarantee identical test subsets.
    """
    exclude_time_sensitive = (dataset == "mkqa")
    if STRATIFIED:
        fname = (
            f"train_lang_{TRAIN_LANG}_splits_stratified_without_time_sensitive.json"
            if exclude_time_sensitive
            else f"train_lang_{TRAIN_LANG}_splits_stratified.json"
        )
    else:
        fname = f"train_lang_{TRAIN_LANG}_splits.json"

    splits_path = os.path.join(results_dir, dataset, model_name, datasplit, fname)
    if not os.path.exists(splits_path):
        logging.error(f"Splits file not found: {splits_path}")
        sys.exit(1)

    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    test_ids = set(str(eid) for eid in splits["test"])
    logging.info(f"Loaded {len(test_ids)} test IDs from {splits_path}")
    return test_ids


def load_answers(answers_file: str, test_ids: set = None) -> list:
    """
    Load (example_id, query, answer) triples from {lang}_answer.jsonl.
    If test_ids is provided, only examples in the test split are returned.
    """
    if not os.path.exists(answers_file):
        logging.error(f"Answers file not found: {answers_file}")
        logging.error("Run hidden_states.py extract_hidden_states first.")
        sys.exit(1)

    examples = []
    skipped  = 0
    with open(answers_file, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if test_ids is not None and str(rec["example_id"]) not in test_ids:
                skipped += 1
                continue
            examples.append((str(rec["example_id"]), rec["query"], rec["answer"]))

    if test_ids is not None:
        missing = test_ids - {eid for eid, _, _ in examples}
        if missing:
            logging.warning(
                f"{len(missing)} test-split examples are absent from the answers "
                f"file and will be excluded: {answers_file}"
            )
        logging.info(
            f"Loaded {len(examples)} test-split examples "
            f"({skipped} skipped — not in test split, "
            f"{len(missing)} missing from answers file)"
        )
    else:
        logging.info(f"Loaded {len(examples)} examples from {answers_file}")
    return examples


def load_processed_ids(nsl_file: str) -> set:
    """Return IDs already written to the output file (for resume support)."""
    seen = set()
    if not os.path.exists(nsl_file):
        return seen
    with open(nsl_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                seen.add(json.loads(line)["example_id"])
            except Exception:
                pass
    return seen


def load_system_instruction(dataset: str, datasplit: str, lang: str) -> str:
    """
    Load the system prompt for the given (dataset, datasplit, lang) combination,
    mirroring the exact loading logic in hidden_states.py so NSL is computed
    with the same context the model saw during generation.
    """
    with open(SYS_PROMPT_FILE, "r", encoding="utf-8") as pf:
        prompts_cfg = json.load(pf)
    if dataset == "mkqa":
        return prompts_cfg[dataset][datasplit][lang]
    else:
        return prompts_cfg[dataset][lang]


# ── SHARED INFERENCE LOOP ────────────────────────────────────────────────────

def _run_nsl_loop(
    model,
    tokenizer,
    examples: list,
    nsl_file: str,
    batch_size: int,
    model_type: str,
    system_instruction: str,
    start_total: float,
):
    """
    Batched NSL inference loop.

    For each (query, answer) pair:
      1. Build the prompt via the chat template (left-padded across the batch).
      2. Encode the answer with tokenizer.encode(add_special_tokens=False),
         matching hidden_states.py's own tokenization of gen_text exactly.
      3. Locate query tokens within the prompt using find_subsequence.
      4. Append answer tokens (right-padded) to the prompt.
      5. Run ONE forward pass: logits (B, L+max_T, V).
         Because of causal attention, prompt-position logits are unaffected
         by the answer tokens appended to the right — both NSL values come
         from this single pass.
      6. NSL_query:  mean log P over query token positions in the prompt.
         NSL_answer: mean log P over answer token positions (L-1+j).

    Output record per example:
        {
          "example_id":    ...,
          "query":         ...,
          "answer":        ...,
          "nsl_query":     <float>,   # mean log P of query tokens
          "nsl_answer":    <float>,   # mean log P of answer tokens
          "query_length":  <int>,
          "answer_length": <int>,
          "uncertainty":   <float>,   # -nsl_answer
        }
    """
    seen_ids   = load_processed_ids(nsl_file)
    to_process = [(eid, q, a) for eid, q, a in examples if eid not in seen_ids]
    logging.info(
        f"To process: {len(to_process)} | Already done (skipped): {len(seen_ids)}"
    )

    if not to_process:
        logging.info("All examples already processed. Nothing to do.")
        return

    total_batches = (len(to_process) + batch_size - 1) // batch_size

    with open(nsl_file, "a", encoding="utf-8") as out_f, torch.no_grad():
        for batch_num, i in enumerate(
            tqdm(
                range(0, len(to_process), batch_size),
                total=total_batches,
                desc="NSL",
                unit="batch",
            ),
            start=1,
        ):
            t_batch_start = time.time()
            batch          = to_process[i : i + batch_size]
            ids, queries, answers = zip(*batch)

            # ── TOKENISE PROMPTS AND ANSWERS ──────────────────────────────
            prompt_ids_list  = []
            answer_ids_list  = []

            for q, a in zip(queries, answers):
                msgs = [
                    {"role": "system", "content": system_instruction},
                    {"role": "user",   "content": q},
                ]
                if model_type == "qwen":
                    p_ids = tokenizer.apply_chat_template(
                        msgs, tokenize=True, add_generation_prompt=True,
                        enable_thinking=False, return_dict=False,
                    )
                else:
                    p_ids = tokenizer.apply_chat_template(
                        msgs, tokenize=True, add_generation_prompt=True, return_dict=False,
                    )
                # Encode the answer exactly as hidden_states.py does:
                # tokenizer.encode(gen_text, add_special_tokens=False).
                # Using the chat template instead risks adding spurious
                # tokens (e.g. a double '\n' for Qwen3 when gen_text
                # already ends with '\n' before <|im_end|>).
                a_ids = tokenizer.encode(a, add_special_tokens=False)
                prompt_ids_list.append(p_ids)
                answer_ids_list.append(a_ids)

            # ── LOCATE QUERY TOKENS WITHIN EACH PROMPT ───────────────────
            # find_subsequence mirrors hidden_states.py: encode the query
            # with add_special_tokens=False and search for it in the prompt.
            query_info = []   # list of (q_token_ids, q_start, q_end) | None
            for idx_q, (q, p_ids) in enumerate(zip(queries, prompt_ids_list)):
                q_toks = tokenizer.encode(q, add_special_tokens=False)
                q_start, q_end = find_subsequence(q_toks, p_ids)
                if q_start == -1:
                    logging.warning(
                        f"  ex {ids[idx_q]}: query tokens not found in prompt — "
                        "nsl_query set to NaN."
                    )
                    query_info.append(None)
                else:
                    query_info.append((q_toks, q_start, q_end))

            # ── PAD PROMPTS (LEFT) ────────────────────────────────────────
            tok = tokenizer.pad(
                {"input_ids": prompt_ids_list}, padding=True, return_tensors="pt"
            )
            tok = {k: v.to(model.device) for k, v in tok.items()}

            B, L   = tok["input_ids"].shape
            max_T  = max((len(a) for a in answer_ids_list), default=0)

            if max_T == 0:
                logging.warning("All answers in this batch are empty. Skipping.")
                continue

            # ── PAD ANSWERS (RIGHT) and build masks ───────────────────────
            # Right-padding is correct here: causal attention means the
            # padded zero tokens at positions > real answer length never
            # influence the logits at the real answer positions.
            dev = model.device
            answer_tensor = torch.zeros(B, max_T, dtype=torch.long,  device=dev)
            answer_mask   = torch.zeros(B, max_T, dtype=torch.long,  device=dev)
            for idx, a_ids in enumerate(answer_ids_list):
                T_i = len(a_ids)
                if T_i > 0:
                    answer_tensor[idx, :T_i] = torch.tensor(
                        a_ids, dtype=torch.long, device=dev
                    )
                    answer_mask[idx, :T_i] = 1

            # ── SINGLE FORWARD PASS ───────────────────────────────────────
            full_input = torch.cat([tok["input_ids"],      answer_tensor], dim=1)
            full_mask  = torch.cat([tok["attention_mask"], answer_mask  ], dim=1)

            logits = model(
                input_ids=full_input, attention_mask=full_mask
            ).logits  # (B, L+max_T, V)

            log_sf = torch.log_softmax(logits, dim=-1)  # (B, L+max_T, V)

            # ── EXTRACT NSL_QUERY AND NSL_ANSWER PER EXAMPLE ─────────────
            for idx, ex_id in enumerate(ids):

                # ── NSL_answer ────────────────────────────────────────────
                a_ids = answer_ids_list[idx]
                T_a   = len(a_ids)
                if T_a == 0:
                    logging.warning(
                        f"  ex {ex_id}: empty answer after stripping special "
                        "tokens — nsl_answer set to NaN."
                    )
                    nsl_answer = float("nan")
                else:
                    # With left-padding, last real prompt token is always at
                    # L-1; logit at L-1+j predicts answer token j.
                    nsl_answer = sum(
                        log_sf[idx, L - 1 + j, tok_id].item()
                        for j, tok_id in enumerate(a_ids)
                    ) / T_a

                # ── NSL_query ─────────────────────────────────────────────
                qi = query_info[idx]
                if qi is None:
                    nsl_query = float("nan")
                    T_q = 0
                else:
                    q_toks, q_start, q_end = qi
                    T_q = q_end - q_start
                    # pad_len = how many padding tokens precede the real prompt
                    pad_len = L - len(prompt_ids_list[idx])
                    # Absolute position of query token j in the padded sequence:
                    #   pad_len + q_start + j
                    # Logit predicting it is one position earlier:
                    #   pad_len + q_start + j - 1
                    nsl_query = sum(
                        log_sf[idx, pad_len + q_start + j - 1, tok_id].item()
                        for j, tok_id in enumerate(q_toks)
                    ) / T_q

                uncertainty = (
                    -nsl_answer if not math.isnan(nsl_answer) else float("nan")
                )

                rec = {
                    "example_id":       ex_id,
                    "query":            queries[idx],
                    "answer":           answers[idx],
                    "nsl_query":        nsl_query,
                    "nsl_query_prob":   float(math.exp(nsl_query)) if not math.isnan(nsl_query) else float("nan"),
                    "nsl_answer":       nsl_answer,
                    "nsl_answer_prob":  float(math.exp(nsl_answer)) if not math.isnan(nsl_answer) else float("nan"),
                    "query_length":     T_q,
                    "answer_length":    T_a,
                    "uncertainty":      uncertainty,
                }
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                logging.info(
                    f"  ex {ex_id}: nsl_query={nsl_query:.4f}  "
                    f"nsl_answer={nsl_answer:.4f}  "
                    f"uncertainty={uncertainty:.4f}"
                )

            # flush once per batch for crash-recovery
            out_f.flush()

            # ── BATCH TIMING ──────────────────────────────────────────────
            t_elapsed     = time.time() - t_batch_start
            avg_per_batch = (time.time() - start_total) / batch_num
            eta_seconds   = avg_per_batch * (total_batches - batch_num)
            logging.info(
                f"Batch {batch_num}/{total_batches} | "
                f"this={t_elapsed:.1f}s | avg={avg_per_batch:.1f}s | "
                f"ETA={eta_seconds / 60:.1f} min"
            )

            # ── MEMORY CLEANUP ────────────────────────────────────────────
            try:
                del logits, log_sf, full_input, full_mask
                del answer_tensor, answer_mask, tok
            except Exception:
                pass
            torch.cuda.empty_cache()
            gc.collect()


# ── MODEL-SPECIFIC ENTRY POINTS ───────────────────────────────────────────────

def compute_nsl_qwen(
    local_model_path: str,
    results_dir: str,
    dataset: str,
    datasplit: str,
    lang: str,
    batch_size: int,
):
    logging.info("Computing NSL with Qwen3 8B")

    answers_file = f"{results_dir}/{dataset}/qwen3_8B/{datasplit}/{lang}_answer.jsonl"
    nsl_file     = f"{results_dir}/{dataset}/qwen3_8B/{datasplit}/{lang}_nsl.jsonl"
    os.makedirs(os.path.dirname(nsl_file), exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        local_model_path, local_files_only=True, use_fast=True,
        padding_side="left", trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        local_model_path, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map="auto", trust_remote_code=True,
    )
    model.eval()

    if tokenizer.pad_token_id is None:
        logging.warning("No padding token identified. Exiting.")
        sys.exit(1)

    test_ids    = load_test_ids(results_dir, dataset, "qwen3_8B", datasplit)
    examples    = load_answers(answers_file, test_ids=test_ids)
    start_total = time.time()

    system_instruction = load_system_instruction(dataset, datasplit, lang)
    _run_nsl_loop(model, tokenizer, examples, nsl_file, batch_size,
                  model_type="qwen", system_instruction=system_instruction,
                  start_total=start_total)

    total_time = time.time() - start_total
    logging.info(f"\nDone. Total time: {total_time / 60:.1f} min ({total_time / 3600:.2f} h)")
    logging.info(f"✅ NSL scores saved to {nsl_file}")


def compute_nsl_llama(
    local_model_path: str,
    results_dir: str,
    dataset: str,
    datasplit: str,
    lang: str,
    batch_size: int,
):
    logging.info("Computing NSL with Llama 3.1 8B Instruct")

    answers_file = f"{results_dir}/{dataset}/llama_3.1_8B/{datasplit}/{lang}_answer.jsonl"
    nsl_file     = f"{results_dir}/{dataset}/llama_3.1_8B/{datasplit}/{lang}_nsl.jsonl"
    os.makedirs(os.path.dirname(nsl_file), exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        local_model_path, local_files_only=True, use_fast=True,
        padding_side="left",
    )
    model = AutoModelForCausalLM.from_pretrained(
        local_model_path, local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map="auto",
    )
    model.eval()
    tokenizer.pad_token_id = tokenizer.eos_token_id  # Llama has no explicit pad token

    test_ids    = load_test_ids(results_dir, dataset, "llama_3.1_8B", datasplit)
    examples    = load_answers(answers_file, test_ids=test_ids)
    start_total = time.time()

    system_instruction = load_system_instruction(dataset, datasplit, lang)
    _run_nsl_loop(model, tokenizer, examples, nsl_file, batch_size,
                  model_type="llama", system_instruction=system_instruction,
                  start_total=start_total)

    total_time = time.time() - start_total
    logging.info(f"\nDone. Total time: {total_time / 60:.1f} min ({total_time / 3600:.2f} h)")
    logging.info(f"✅ NSL scores saved to {nsl_file}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser():
    parser = argparse.ArgumentParser(
        description="Compute NSL confidence scores for pre-generated LLM answers."
    )
    parser.add_argument("--track-emissions", action="store_true", default=False)

    sub = parser.add_subparsers(dest="operation", required=True)
    p_compute = sub.add_parser(
        "compute_nsl",
        help="Run NSL estimation for one model / dataset / datasplit / language.",
    )
    p_compute.add_argument("--model-name",     type=str, choices=MODELS,                             required=True)
    p_compute.add_argument("--dataset-name",   type=str, choices=DATASETS,                           required=True)
    p_compute.add_argument("--datasplit-name", type=str, choices=DATASPLITS_GMMLU + DATASPLITS_MKQA, required=True)
    p_compute.add_argument("--lang",           type=str, choices=LANGUAGES,                          required=True)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()

    tracker = None
    if args.track_emissions:
        tracker = OfflineEmissionsTracker(
            output_dir=EMISSIONS_DIR,
            output_file="nsl.csv",
        )
        tracker.start()

    try:
        if args.model_name == "llama_3.1_8B":
            compute_nsl_llama(
                LLAMA_MODEL_PATH, RESULTS_DIR,
                args.dataset_name, args.datasplit_name, args.lang, BATCH_SIZE,
            )
        elif args.model_name == "qwen3_8B":
            compute_nsl_qwen(
                QWEN_MODEL_PATH, RESULTS_DIR,
                args.dataset_name, args.datasplit_name, args.lang, BATCH_SIZE,
            )
        else:
            logging.error(f"Unknown model: {args.model_name}")
            sys.exit(1)
    finally:
        if tracker is not None:
            tracker.stop()