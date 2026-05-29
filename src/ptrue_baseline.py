"""
p_true.py — P(True) confidence estimation baseline.

For each (question, answer) pair already produced by hidden_states.py, this script:
  1. Builds a language-specific P(True) verification prompt (see LANG_CONFIGS)
     following the structure of the LM-Polygraph PTrue estimator:
         <Question label>: {q}
         <Answer label>: {a}
         <Is-it-correct question>
         (A) <True word>
         (B) <False word>
         <Conclusion line>
  2. Wraps it in the model's chat template and runs a teacher-forced
     forward pass for each True/False surface form.
  3. Computes the full sequence probability P(surface | prompt) for each
     surface form and sums over True / False candidates respectively.
  4. Writes one JSON record per example to {lang}_p_true.jsonl.

Reference: https://arxiv.org/abs/2207.05221
           https://github.com/IINemo/lm-polygraph/blob/main/src/lm_polygraph/estimators/p_true.py

Usage (mirrors hidden_states.py):
    python p_true.py compute_p_true \
        --model-name  llama_3.1_8B \
        --dataset-name global_mmlu \
        --datasplit-name  <split> \
        --lang en

    python p_true.py compute_p_true \
        --model-name  qwen3_8B \
        --dataset-name mkqa \
        --datasplit-name  <split> \
        --lang de \
        --track-emissions
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable, List

import torch
from codecarbon import OfflineEmissionsTracker
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

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
)

# ── SPLIT CONFIG ─────────────────────────────────────────────────────────────
# P(True) is evaluated only on the test split defined by the splits JSON,
# matching exactly the examples seen by eval_probe.py / eval_p_true.py.
TRAIN_LANG  = 'fr'   # training language that defines the split
STRATIFIED  = True   # must match the splits file used during probe training

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)

# ── LANGUAGE CONFIGURATIONS ──────────────────────────────────────────────────
# ⚠ Known deviation from LM-Polygraph:
#   LM-Polygraph feeds the prompt as a raw completion to a WhiteboxModel.
#   Because we use instruct-tuned checkpoints we must wrap it in each
#   model's chat template, which is an irremovable deviation that should
#   be noted when reporting results against the LM-Polygraph baseline.
#
# Prompt construction uses f-strings (not str.format()) so that curly
# braces inside LLM-generated answers do not raise KeyError/IndexError.

@dataclass
class LangConfig:
    """
    All language-specific P(True) configuration in one place.

    Attributes:
        build_prompt   — callable (question, answer) → verification prompt string
        true_surfaces  — surface forms to collect token IDs for (True side)
        false_surfaces — surface forms to collect token IDs for (False side)
        anchor         — last line of the prompt, used as context for token-ID
                         resolution via prefix subtraction
    """
    build_prompt:   Callable[[str, str], str]
    true_surfaces:  List[str]
    false_surfaces: List[str]
    anchor:         str


# For each language, true_surfaces / false_surfaces include both the
# space-prefixed form (most likely first generated token in BPE models)
# and the bare form, plus lowercase variants where applicable.
# Japanese has no case variants; space-prefixed and bare are sufficient.
LANG_CONFIGS: dict = {
    "en": LangConfig(
        build_prompt=lambda q, a: (
            f"Question: {q}\n"
            f"Possible answer: {a}\n"
            "Is the possible answer:\n"
            " (A) True\n"
            " (B) False\n"
            "The possible answer is:"
        ),
        true_surfaces=[" True", "True", " true", "true"],
        false_surfaces=[" False", "False", " false", "false"],
        anchor="The possible answer is:",
    ),
    "es": LangConfig(
        build_prompt=lambda q, a: (
            f"Pregunta: {q}\n"
            f"Posible respuesta: {a}\n"
            "Es la posible respuesta:\n"
            "(A) Verdadero\n"
            "(B) Falso\n"
            "La posible respuesta es:"
        ),
        true_surfaces=[" Verdadero", "Verdadero", " verdadero", "verdadero"],
        false_surfaces=[" Falso", "Falso", " falso", "falso"],
        anchor="La posible respuesta es:",
    ),
    "fr": LangConfig(
        build_prompt=lambda q, a: (
            f"Question: {q}\n"
            f"Réponse possible: {a}\n"
            "La réponse possible est-elle:\n"
            " (A) Vrai\n"
            " (B) Faux\n"
            "La réponse possible est:"
        ),
        true_surfaces=[" Vrai", "Vrai", " vrai", "vrai"],
        false_surfaces=[" Faux", "Faux", " faux", "faux"],
        anchor="La réponse possible est:",
    ),
    "pl": LangConfig(
        build_prompt=lambda q, a: (
            f"Pytanie: {q}\n"
            f"Możliwa odpowiedź: {a}\n"
            "Czy możliwa odpowiedź to:\n"
            " (A) Prawda\n"
            " (B) Fałsz\n"
            "Możliwa odpowiedź to:"
        ),
        true_surfaces=[" Prawda", "Prawda", " prawda", "prawda"],
        false_surfaces=[" Fałsz", "Fałsz", " fałsz", "fałsz"],
        anchor="Możliwa odpowiedź to:",
    ),
    "ru": LangConfig(
        build_prompt=lambda q, a: (
            f"Вопрос: {q}\n"
            f"Возможный ответ: {a}\n"
            "Является ли возможный ответ:\n"
            " (A) Верным\n"
            " (B) Неверным\n"
            "Возможный ответ является:"
        ),
        true_surfaces=[" Верным", "Верным", " верным", "верным"],
        false_surfaces=[" Неверным", "Неверным", " неверным", "неверным"],
        anchor="Возможный ответ является:",
    ),
    "ja": LangConfig(
        build_prompt=lambda q, a: (
            f"質問: {q}\n"
            f"考えられる答え: {a}\n"
            "その答えは可能だろうか:\n"
            " (A) 真\n"
            " (B) 偽\n"
            "考えられる回答の正誤は:"
        ),
        # Japanese has no case variants; space-prefixed and bare suffice.
        true_surfaces=[" 真", "真"],
        false_surfaces=[" 偽", "偽"],
        anchor="考えられる回答の正誤は:",
    ),
}


# ── HELPERS ───────────────────────────────────────────────────────────────────

def get_true_false_token_ids(tokenizer, lang_config: LangConfig):
    """
    Resolve the full token sequences for every True/False surface form,
    using contextual prefix subtraction (encode anchor+surface, subtract
    anchor prefix).  Returns ALL sequences — single-token and multi-token —
    deduplicated by full tuple.

    Multi-token sequences are kept (unlike the earlier first-token approach)
    because sequence probability is computed via teacher forcing: the full
    joint P(t1) × P(t2|t1) × … is computed in one forward pass, so there
    is no approximation error from multi-token words.

    Returns:
        true_seqs  (List[List[int]])  — deduplicated token sequences for True
        false_seqs (List[List[int]])  — deduplicated token sequences for False
    """
    ANCHOR = lang_config.anchor
    prefix_ids = tokenizer.encode(ANCHOR, add_special_tokens=False)
    n = len(prefix_ids)

    def _contextual_tokens(surface):
        """Return the full contextual token sequence for surface, or None."""
        full_ids = tokenizer.encode(ANCHOR + surface, add_special_tokens=False)
        if full_ids[:n] != prefix_ids:
            logging.warning(
                f"Prefix tokens shifted for surface {surface!r} "
                f"(prefix_ids={prefix_ids}, full_ids={full_ids}). Skipping."
            )
            return None
        suffix = full_ids[n:]
        if not suffix:
            logging.warning(f"Empty suffix for surface {surface!r}. Skipping.")
            return None
        return suffix  # full sequence, not just the first token

    def _resolve_surface_forms(surfaces, label):
        seen = {}   # tuple(token_ids) → first surface that produced it (for logging)
        for surface in surfaces:
            seq = _contextual_tokens(surface)
            if seq is not None:
                key = tuple(seq)
                if key not in seen:
                    seen[key] = surface
                    logging.info(
                        f"  '{label}' <- surface {surface!r} -> "
                        f"tokens {list(key)} ({tokenizer.decode(list(key))!r})"
                    )
        if not seen:
            logging.error(
                f"Could not resolve any token sequence for '{label}' "
                f"from surfaces {surfaces}. Exiting."
            )
            sys.exit(1)
        return [list(k) for k in seen.keys()]

    logging.info(f"Resolving True/False token sequences for lang={lang_config.anchor!r}:")
    true_seqs  = _resolve_surface_forms(lang_config.true_surfaces,  "True")
    false_seqs = _resolve_surface_forms(lang_config.false_surfaces, "False")
    logging.info(f"  True  sequences : {true_seqs}")
    logging.info(f"  False sequences : {false_seqs}")
    return true_seqs, false_seqs

def load_test_ids(results_dir: str, dataset: str, model_name: str,
                  datasplit: str) -> set:
    """
    Load the test-split example IDs from the splits JSON file produced by
    hidden_states.py.  The path mirrors the logic in eval_p_true.py and
    classification_baselines.py so the exact same subset is evaluated.

    Path pattern:
        {results_dir}/{dataset}/{model_name}/{datasplit}/
            train_lang_{TRAIN_LANG}_splits_stratified[_without_time_sensitive].json
    """
    exclude_time_sensitive = (dataset == "mkqa")
    if STRATIFIED:
        if exclude_time_sensitive:
            fname = f"train_lang_{TRAIN_LANG}_splits_stratified_without_time_sensitive.json"
        else:
            fname = f"train_lang_{TRAIN_LANG}_splits_stratified.json"
    else:
        fname = f"train_lang_{TRAIN_LANG}_splits.json"

    splits_path = os.path.join(results_dir, dataset, model_name, datasplit, fname)
    if not os.path.exists(splits_path):
        logging.error(f"Splits file not found: {splits_path}")
        logging.error(
            "Ensure hidden_states.py has been run with the same TRAIN_LANG "
            "and STRATIFIED settings."
        )
        sys.exit(1)

    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    test_ids = set(str(eid) for eid in splits["test"])  # normalise to str — answers file stores IDs as strings
    sample = list(test_ids)[:3]
    logging.info(
        f"Loaded {len(test_ids)} test IDs from {splits_path} | "
        f"sample (type={type(list(test_ids)[0]).__name__}): {sample}"
    )
    return test_ids


def load_answers(answers_file: str, test_ids: set = None) -> list:
    """
    Load pre-computed (example_id, query, answer) triples from
    the {lang}_answer.jsonl written by hidden_states.py.

    If test_ids is provided, only examples whose example_id is in
    that set are returned — restricting inference to the same subset
    used by eval_probe.py / eval_p_true.py for a fair comparison.

    Returns:
        list of (example_id, query, answer)
    """
    if not os.path.exists(answers_file):
        logging.error(f"Answers file not found: {answers_file}")
        logging.error("Run hidden_states.py extract_hidden_states first.")
        sys.exit(1)

    # ── Diagnostic: sample raw IDs from answers file before filtering ──────
    if test_ids is not None:
        with open(answers_file, "r", encoding="utf-8") as f:
            raw_sample = [json.loads(f.readline())["example_id"] for _ in range(3)
                          if True]  # read up to 3 lines
        logging.info(
            f"Answer file ID sample (type={type(raw_sample[0]).__name__ if raw_sample else 'n/a'}): "
            f"{[(eid, type(eid).__name__) for eid in raw_sample]}"
        )

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
        # Diagnostic: show a few IDs from each source so mismatches are visible
        if examples:
            sample_ans = [(eid, type(eid).__name__) for eid, _, _ in examples[:3]]
            logging.info(f"Answer file ID sample: {sample_ans}")
        sample_test = [(eid, type(eid).__name__) for eid in list(test_ids)[:3]]
        logging.info(f"Splits file ID sample:  {sample_test}")
        missing = test_ids - {eid for eid, _, _ in examples}
        if missing:
            logging.warning(
                f"{len(missing)} test-split examples are absent from the answers "
                f"file and will be excluded from P(True) evaluation: {answers_file}. "
                "Re-run hidden_states.py if this is unexpected."
            )
        logging.info(
            f"Loaded {len(examples)} test-split examples from {answers_file} "
            f"({skipped} skipped — not in test split, {len(missing) if test_ids else 0} missing)"
        )
    else:
        logging.info(f"Loaded {len(examples)} examples from {answers_file}")
    return examples


def load_processed_ids(p_true_file: str) -> set:
    """Return IDs already written to the output file (for resume support)."""
    seen = set()
    if not os.path.exists(p_true_file):
        return seen
    with open(p_true_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                seen.add(json.loads(line)["example_id"])
            except Exception:
                pass
    return seen


# ── SHARED INFERENCE LOOP ────────────────────────────────────────────────────


def _compute_seq_probs(model, input_ids_base, attention_mask_base, surface_seqs):
    """
    Compute the teacher-forced sequence probability P(surface | prompt) for
    every surface sequence in surface_seqs, for every example in the batch.

    Strategy:
      For each surface token sequence [t0, t1, …, tN-1]:
        1. Append the surface tokens to the (left-padded) prompt:
               input  = [pad…, p0…pL-1, t0, t1, …, tN-1]
               mask   = [0…,   1…1,     1,  1,  …, 1    ]
        2. Run one forward pass to get logits everywhere.
        3. With left-padding, the last real prompt token is always at
           position L-1 (regardless of example-specific prompt length),
           so the logit at position (L-1+j) predicts surface token j.
        4. log P(surface | prompt_i) = Σ_j log_softmax(logits[i, L-1+j])[t_j]
        5. P(surface | prompt_i)     = exp( log P )

    Args:
        model                — the CausalLM model in eval mode
        input_ids_base       — (B, L) left-padded prompt token IDs
        attention_mask_base  — (B, L) corresponding attention mask
        surface_seqs         — list of token-ID lists (deduplicated surface forms)

    Returns:
        probs  (B, len(surface_seqs))  — sequence probability per example per surface
    """
    B, L   = input_ids_base.shape
    device = input_ids_base.device
    results = []

    with torch.no_grad():
        for surface_ids in surface_seqs:
            N              = len(surface_ids)
            surface_tensor = torch.tensor(surface_ids, dtype=torch.long, device=device)
            surface_exp    = surface_tensor.unsqueeze(0).expand(B, -1)        # (B, N)
            surface_mask   = torch.ones(B, N, dtype=torch.long, device=device)

            inp  = torch.cat([input_ids_base,      surface_exp ], dim=1)      # (B, L+N)
            mask = torch.cat([attention_mask_base,  surface_mask], dim=1)     # (B, L+N)

            logits = model(input_ids=inp, attention_mask=mask).logits         # (B, L+N, V)
            log_sf = torch.log_softmax(logits, dim=-1)

            seq_lp = torch.zeros(B, device=device)
            for j, tok_id in enumerate(surface_ids):
                # Position L-1+j predicts surface token j (left-padding guarantee)
                seq_lp += log_sf[:, L - 1 + j, tok_id]

            results.append(torch.exp(seq_lp))   # (B,)

            del inp, mask, logits, log_sf, seq_lp
            torch.cuda.empty_cache()

    return torch.stack(results, dim=1)   # (B, len(surface_seqs))


def _run_p_true_loop(
    model,
    tokenizer,
    examples: list,
    p_true_file: str,
    batch_size: int,
    true_seqs: list,
    false_seqs: list,
    lang_config: LangConfig,
    model_type: str,     # "qwen" | "llama"
    start_total: float,
):
    """
    Batched P(True) inference loop shared by both model functions.

    For each (query, answer) pair:
      - Formats the P(True) verification prompt.
      - Wraps it in the chat template (user turn only; no system prompt
        to keep the verification context clean and model-agnostic).
      - Runs a teacher-forced forward pass for each True/False surface form.
      - Computes teacher-forced sequence probability P(surface | prompt) for
        each True/False surface form and sums across forms.
      - Appends a JSON record to p_true_file.

    Alignment with LM-Polygraph PTrue:
      - stats["p_true"]  = Σ P(surface | prompt) over True  surface forms
      - stats["p_false"] = Σ P(surface | prompt) over False surface forms
      - uncertainty      = -p_true  (reproduced by LM-Polygraph __call__)

    Output record per example:
        {
          "example_id":  ...,
          "query":       ...,
          "answer":      ...,
          "p_true":      <float>,   # Σ P(surface | prompt) over True  surface forms
          "p_false":     <float>,   # Σ P(surface | prompt) over False surface forms
          "predicted":   "A" | "B", # higher-probability option
          "uncertainty": <float>,   # -p_true, matching LM-Polygraph PTrue.__call__
        }
    """
    seen_ids  = load_processed_ids(p_true_file)
    to_process = [(eid, q, a) for eid, q, a in examples if eid not in seen_ids]
    logging.info(
        f"To process: {len(to_process)} | Already done (skipped): {len(seen_ids)}"
    )

    if not to_process:
        logging.info("All examples already processed. Nothing to do.")
        return

    total_batches = (len(to_process) + batch_size - 1) // batch_size

    # Open output file in append mode so we can resume after interruption
    with open(p_true_file, "a", encoding="utf-8") as out_f, torch.no_grad():
        for batch_num, i in enumerate(
            tqdm(
                range(0, len(to_process), batch_size),
                total=total_batches,
                desc="P(True)",
                unit="batch",
            ),
            start=1,
        ):
            t_batch_start = time.time()
            batch          = to_process[i : i + batch_size]
            ids, queries, answers = zip(*batch)

            # ── BUILD VERIFICATION PROMPTS ─────────────────────────────────
            input_ids_list = []
            for q, a in zip(queries, answers):
                verification_prompt = lang_config.build_prompt(q, a)
                msgs = [{"role": "user", "content": verification_prompt}]

                if model_type == "qwen":
                    out = tokenizer.apply_chat_template(
                        msgs,
                        tokenize=True,
                        add_generation_prompt=True,
                        enable_thinking=False,   # keep non-thinking mode for Qwen3
                        return_dict=True,
                    )
                    input_ids_list.append(out["input_ids"])
                else:  # llama
                    tids = tokenizer.apply_chat_template(
                        msgs,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_dict=False,
                    )
                    input_ids_list.append(tids)

            tok = tokenizer.pad(
                {"input_ids": input_ids_list}, padding=True, return_tensors="pt"
            )
            tok = {k: v.to(model.device) for k, v in tok.items()}

            torch.cuda.empty_cache()

            # ── TEACHER-FORCED SEQUENCE PROBABILITY ───────────────────────
            # For each surface form, append its tokens to the prompt and do
            # one forward pass.  The logits at positions L-1…L-1+N-1 give
            # P(t0|prompt) × P(t1|prompt,t0) × … = P(surface|prompt).
            # This is correct for single-token AND multi-token words.
            true_probs  = _compute_seq_probs(
                model, tok["input_ids"], tok["attention_mask"], true_seqs
            )  # (B, len(true_seqs))
            false_probs = _compute_seq_probs(
                model, tok["input_ids"], tok["attention_mask"], false_seqs
            )  # (B, len(false_seqs))

            # ── EXTRACT AND WRITE SCORES ───────────────────────────────────
            for idx, ex_id in enumerate(ids):
                p_true_val  = true_probs[idx].sum().item()
                p_false_val = false_probs[idx].sum().item()
                predicted   = "A" if p_true_val >= p_false_val else "B"
                uncertainty = -p_true_val  # matches LM-Polygraph: return -np.array(ptrue)

                rec = {
                    "example_id": ex_id,
                    "query":      queries[idx],
                    "answer":     answers[idx],
                    "p_true":     p_true_val,
                    "p_false":    p_false_val,
                    "predicted":  predicted,
                    "uncertainty": uncertainty,
                }
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                logging.info(
                    f"  ex {ex_id}: p_true={p_true_val:.4f}  "
                    f"p_false={p_false_val:.4f}  predicted={predicted}  "
                    f"uncertainty={uncertainty:.4f}"
                )

            # flush once per batch — sufficient for crash recovery since
            # resume logic skips at example granularity from the output file
            out_f.flush()

            # ── BATCH TIMING ───────────────────────────────────────────────
            t_elapsed = time.time() - t_batch_start
            avg_per_batch  = (time.time() - start_total) / batch_num
            eta_seconds    = avg_per_batch * (total_batches - batch_num)
            logging.info(
                f"Batch {batch_num}/{total_batches} | "
                f"this={t_elapsed:.1f}s | avg={avg_per_batch:.1f}s | "
                f"ETA={eta_seconds / 60:.1f} min"
            )

            # ── MEMORY CLEANUP ─────────────────────────────────────────────
            try:
                del true_probs, false_probs, tok
            except Exception:
                pass
            torch.cuda.empty_cache()
            gc.collect()


# ── MODEL-SPECIFIC ENTRY POINTS ───────────────────────────────────────────────

def compute_p_true_qwen(
    local_model_path: str,
    results_dir: str,
    dataset: str,
    datasplit: str,
    lang: str,
    batch_size: int,
):
    logging.info("Computing P(True) with Qwen3 8B")

    answers_file = f"{results_dir}/{dataset}/qwen3_8B/{datasplit}/{lang}_answer.jsonl"
    p_true_file  = f"{results_dir}/{dataset}/qwen3_8B/{datasplit}/{lang}_p_true.jsonl"
    os.makedirs(os.path.dirname(p_true_file), exist_ok=True)

    # ── TOKENIZER & MODEL ─────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        local_model_path,
        local_files_only=True,
        use_fast=True,
        padding_side="left",
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    if tokenizer.pad_token_id is None:
        logging.warning("No padding token identified. Exiting.")
        sys.exit(1)

    lang_config = LANG_CONFIGS.get(lang)
    if lang_config is None:
        logging.error(f"No P(True) prompt config for language '{lang}'. "
                      f"Available: {list(LANG_CONFIGS.keys())}")
        sys.exit(1)

    true_seqs, false_seqs = get_true_false_token_ids(tokenizer, lang_config)
    test_ids    = load_test_ids(results_dir, dataset, "qwen3_8B", datasplit)
    examples    = load_answers(answers_file, test_ids=test_ids)
    start_total = time.time()

    _run_p_true_loop(
        model, tokenizer, examples, p_true_file, batch_size,
        true_seqs, false_seqs, lang_config, model_type="qwen", start_total=start_total,
    )

    total_time = time.time() - start_total
    logging.info(
        f"\nDone. Total time: {total_time / 60:.1f} min ({total_time / 3600:.2f} h)"
    )
    logging.info(f"✅ P(True) scores saved to {p_true_file}")


def compute_p_true_llama(
    local_model_path: str,
    results_dir: str,
    dataset: str,
    datasplit: str,
    lang: str,
    batch_size: int,
):
    logging.info("Computing P(True) with Llama 3.1 8B Instruct")

    answers_file = f"{results_dir}/{dataset}/llama_3.1_8B/{datasplit}/{lang}_answer.jsonl"
    p_true_file  = f"{results_dir}/{dataset}/llama_3.1_8B/{datasplit}/{lang}_p_true.jsonl"
    os.makedirs(os.path.dirname(p_true_file), exist_ok=True)

    # ── TOKENIZER & MODEL ─────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        local_model_path,
        local_files_only=True,
        use_fast=True,
        padding_side="left",
    )
    model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    )
    model.eval()
    # Llama has no explicit pad token; reuse EOS (standard practice)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    lang_config = LANG_CONFIGS.get(lang)
    if lang_config is None:
        logging.error(f"No P(True) prompt config for language '{lang}'. "
                      f"Available: {list(LANG_CONFIGS.keys())}")
        sys.exit(1)

    true_seqs, false_seqs = get_true_false_token_ids(tokenizer, lang_config)
    test_ids    = load_test_ids(results_dir, dataset, "llama_3.1_8B", datasplit)
    examples    = load_answers(answers_file, test_ids=test_ids)
    start_total = time.time()

    _run_p_true_loop(
        model, tokenizer, examples, p_true_file, batch_size,
        true_seqs, false_seqs, lang_config, model_type="llama", start_total=start_total,
    )

    total_time = time.time() - start_total
    logging.info(
        f"\nDone. Total time: {total_time / 60:.1f} min ({total_time / 3600:.2f} h)"
    )
    logging.info(f"✅ P(True) scores saved to {p_true_file}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser():
    parser = argparse.ArgumentParser(
        description="Compute P(True) confidence scores for pre-generated LLM answers."
    )
    parser.add_argument("--track-emissions", action="store_true", default=False)

    sub = parser.add_subparsers(dest="operation", required=True)

    p_compute = sub.add_parser(
        "compute_p_true",
        help="Run P(True) estimation for one model / dataset / datasplit / language.",
    )
    p_compute.add_argument("--model-name",      type=str, choices=MODELS,                             required=True)
    p_compute.add_argument("--dataset-name",    type=str, choices=DATASETS,                           required=True)
    p_compute.add_argument("--datasplit-name",  type=str, choices=DATASPLITS_GMMLU + DATASPLITS_MKQA, required=True)
    p_compute.add_argument("--lang",            type=str, choices=LANGUAGES,                          required=True)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()

    tracker = None
    if args.track_emissions:
        tracker = OfflineEmissionsTracker(
            output_dir=EMISSIONS_DIR,
            output_file="p_true.csv",
        )
        tracker.start()

    try:
        dataset   = args.dataset_name
        datasplit = args.datasplit_name
        lang      = args.lang
        model     = args.model_name

        if model == "llama_3.1_8B":
            compute_p_true_llama(
                LLAMA_MODEL_PATH, RESULTS_DIR,
                dataset, datasplit, lang, BATCH_SIZE,
            )
        elif model == "qwen3_8B":
            compute_p_true_qwen(
                QWEN_MODEL_PATH, RESULTS_DIR,
                dataset, datasplit, lang, BATCH_SIZE,
            )
        else:
            logging.error(f"Unknown model: {model}")
            sys.exit(1)

    finally:
        if tracker is not None:
            tracker.stop()