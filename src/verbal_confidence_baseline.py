"""
verb_uncertainty_baseline.py — Verbalised Uncertainty confidence estimation baseline.

For each (question, answer) pair already produced by hidden_states.py, this script:
  1. Builds a language-specific verification prompt asking the model to rate
     its confidence from 0 to 100.
  2. Wraps it in the model's chat template and generates a short response
     (greedy decoding, max 30 new tokens).
  3. Parses the numeric confidence value from the generated text.
  4. Writes one JSON record per example to {lang}_verb_unc.jsonl.

Reference: Tian et al., 2023 — "Just Ask for Calibration"
           Lin et al., 2022 — "Teaching Models to Express Their Uncertainty in Words"

Usage (mirrors ptrue_baseline.py):
    python verb_uncertainty_baseline.py compute_verb_unc \\
        --model-name  llama_3.1_8B \\
        --dataset-name global_mmlu \\
        --datasplit-name  <split> \\
        --lang en

    python verb_uncertainty_baseline.py compute_verb_unc \\
        --model-name  qwen3_8B \\
        --dataset-name mkqa \\
        --datasplit-name  <split> \\
        --lang ja \\
        --track-emissions
"""

import argparse
import gc
import json
import logging
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Callable

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
# Evaluated only on the test split defined by the splits JSON, matching
# exactly the examples seen by eval_probe.py / eval_p_true.py.
TRAIN_LANG  = 'fr'
STRATIFIED  = True
MAX_NEW_TOKENS = 10   # expected output is 1-3 tokens (bare number)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)

# ── LANGUAGE CONFIGURATIONS ──────────────────────────────────────────────────
# Each prompt presents the question and answer, then asks the model to
# self-assess its confidence as a percentage from 0 to 100.
#
# Prompt construction uses f-strings (not str.format()) so that curly
# braces inside LLM-generated answers do not raise KeyError/IndexError.

@dataclass
class LangConfig:
    """
    Language-specific verbalised-uncertainty configuration.

    Attributes:
        build_prompt — callable (question, answer) → verification prompt string
    """
    build_prompt: Callable[[str, str], str]


LANG_CONFIGS: dict = {
    "en": LangConfig(
        build_prompt=lambda q, a: (
            f"Question: {q}\n"
            f"Answer: {a}\n\n"
            "Considering the question and your provided answer, estimate the "
            "probability the answer is correct as a percentage from 0 to 100, "
            "where 0 = certainly incorrect and 100 = absolutely correct.\n"
            "Return only the number, nothing else."
        ),
    ),
    "fr": LangConfig(
        build_prompt=lambda q, a: (
            f"Question : {q}\n"
            f"Réponse : {a}\n\n"
            "Compte tenu de la question et de votre réponse, estimez la "
            "probabilité que la réponse soit correcte, en pourcentage, entre "
            "0 et 100, où 0 signifie « certainement incorrect » et 100 "
            "« absolument correct ».\n"
            "Répondez uniquement avec le nombre, rien d'autre."
        ),
    ),
    "es": LangConfig(
        build_prompt=lambda q, a: (
            f"Pregunta: {q}\n"
            f"Respuesta: {a}\n\n"
            "Considerando la pregunta y la respuesta proporcionada, estime la "
            "probabilidad de que la respuesta sea correcta en forma de "
            "porcentaje de 0 a 100, donde 0 = definitivamente incorrecta y "
            "100 = absolutamente correcta.\n"
            "Devuelva solo el número, nada más."
        ),
    ),
    "pl": LangConfig(
        build_prompt=lambda q, a: (
            f"Pytanie: {q}\n"
            f"Odpowiedź: {a}\n\n"
            "Biorąc pod uwagę pytanie i udzieloną odpowiedź, oszacuj "
            "prawdopodobieństwo, że odpowiedź jest poprawna, w procentach "
            "od 0 do 100, gdzie 0 oznacza odpowiedź zdecydowanie błędną, "
            "a 100 \u2013 odpowiedź całkowicie poprawną.\n"
            "Zwróć tylko liczbę, nic więcej."
        ),
    ),
    "ru": LangConfig(
        build_prompt=lambda q, a: (
            f"Вопрос: {q}\n"
            f"Ответ: {a}\n\n"
            "Учитывая вопрос и предоставленный вами ответ, оцените "
            "вероятность правильности ответа в процентах от 0 до 100, "
            "где 0 = заведомо неверный ответ, а 100 = абсолютно верный "
            "ответ.\n"
            "Верните только число, ничего больше."
        ),
    ),
    "ja": LangConfig(
        build_prompt=lambda q, a: (
            f"質問: {q}\n"
            f"回答: {a}\n\n"
            "質問とあなたの回答を考慮し、回答が正しい確率を0～100の範囲で"
            "推定してください。0は確実に間違っている、100は絶対に正しいと"
            "します。\n"
            "数字のみを返してください。"
        ),
    ),
}


# ── PARSING ──────────────────────────────────────────────────────────────────

def parse_confidence(text: str) -> float:
    """
    Extract a confidence value (0–100) from generated text.

    Tries, in order:
      1. Number immediately after a colon  — "Confidence: 85", "confiance : 75"
      2. The entire response is just a number — "85", "85%"
      3. Any number in [0, 100] found anywhere in the text (first match)

    Returns the parsed value, or NaN if nothing valid is found.
    """
    text = text.strip()

    # 1. Number after a colon (handles all languages)
    m = re.search(r':\s*(\d+(?:\.\d+)?)\s*%?', text)
    if m:
        val = float(m.group(1))
        if 0 <= val <= 100:
            return val

    # 2. Entire response is just a number
    m = re.fullmatch(r'(\d+(?:\.\d+)?)\s*%?', text)
    if m:
        val = float(m.group(1))
        if 0 <= val <= 100:
            return val

    # 3. Any number in [0, 100] in the text
    for m in re.finditer(r'\b(\d+(?:\.\d+)?)\b', text):
        val = float(m.group(1))
        if 0 <= val <= 100:
            return val

    return float("nan")


# ── HELPERS ──────────────────────────────────────────────────────────────────

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


def load_processed_ids(out_file: str) -> set:
    """Return IDs already written to the output file (for resume support)."""
    seen = set()
    if not os.path.exists(out_file):
        return seen
    with open(out_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                seen.add(json.loads(line)["example_id"])
            except Exception:
                pass
    return seen


# ── SHARED INFERENCE LOOP ───────────────────────────────────────────────────

def _run_verb_unc_loop(
    model,
    tokenizer,
    examples: list,
    out_file: str,
    batch_size: int,
    lang_config: LangConfig,
    model_type: str,        # "qwen" | "llama"
    terminators: list,
    pad_id: int,
    start_total: float,
):
    """
    Batched verbalised-uncertainty inference loop shared by both model functions.

    For each (query, answer) pair:
      - Formats the verification prompt.
      - Wraps it in the chat template (user turn only; no system prompt).
      - Generates a response with greedy decoding.
      - Parses the numeric confidence value (0–100) from the response.
      - Appends a JSON record to out_file.

    Output record per example:
        {
          "example_id":      ...,
          "query":           ...,
          "answer":          ...,
          "raw_response":    <str>,    # full generated text
          "confidence":      <float>,  # parsed 0–100 value (NaN on failure)
          "confidence_prob": <float>,  # confidence / 100
          "uncertainty":     <float>,  # −confidence_prob
        }
    """
    seen_ids   = load_processed_ids(out_file)
    to_process = [(eid, q, a) for eid, q, a in examples if eid not in seen_ids]
    logging.info(
        f"To process: {len(to_process)} | Already done (skipped): {len(seen_ids)}"
    )

    if not to_process:
        logging.info("All examples already processed. Nothing to do.")
        return

    total_batches = (len(to_process) + batch_size - 1) // batch_size
    n_parse_fail  = 0

    with open(out_file, "a", encoding="utf-8") as out_f, torch.no_grad():
        for batch_num, i in enumerate(
            tqdm(
                range(0, len(to_process), batch_size),
                total=total_batches,
                desc="VerbUnc",
                unit="batch",
            ),
            start=1,
        ):
            t_batch_start = time.time()
            batch          = to_process[i : i + batch_size]
            ids, queries, answers = zip(*batch)

            # ── BUILD VERIFICATION PROMPTS ────────────────────────────────
            input_ids_list = []
            for q, a in zip(queries, answers):
                verification_prompt = lang_config.build_prompt(q, a)
                msgs = [{"role": "user", "content": verification_prompt}]

                if model_type == "qwen":
                    out = tokenizer.apply_chat_template(
                        msgs,
                        tokenize=True,
                        add_generation_prompt=True,
                        enable_thinking=False,
                        return_dict=True,
                    )
                    input_ids_list.append(out["input_ids"])
                else:   # llama
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

            # ── GENERATE RESPONSES ────────────────────────────────────────
            prompt_len = tok["input_ids"].shape[1]

            gen_out = model.generate(
                **tok,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                eos_token_id=terminators,
                pad_token_id=pad_id,
            )

            # ── PARSE AND WRITE SCORES ────────────────────────────────────
            for idx, ex_id in enumerate(ids):
                gen_ids  = gen_out[idx, prompt_len:]
                raw_text = tokenizer.decode(
                    gen_ids, skip_special_tokens=True
                ).strip()

                confidence = parse_confidence(raw_text)

                if math.isnan(confidence):
                    n_parse_fail += 1
                    logging.warning(
                        f"  ex {ex_id}: could not parse confidence from "
                        f"response: {raw_text!r}"
                    )

                confidence_prob = (
                    confidence / 100.0
                    if not math.isnan(confidence)
                    else float("nan")
                )
                uncertainty = (
                    -confidence_prob
                    if not math.isnan(confidence_prob)
                    else float("nan")
                )

                rec = {
                    "example_id":      ex_id,
                    "query":           queries[idx],
                    "answer":          answers[idx],
                    "raw_response":    raw_text,
                    "confidence":      confidence,
                    "confidence_prob": confidence_prob,
                    "uncertainty":     uncertainty,
                }
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                logging.info(
                    f"  ex {ex_id}: confidence={confidence}  "
                    f"prob={confidence_prob:.4f}  "
                    f"response={raw_text!r}"
                )

            # flush once per batch for crash-recovery
            out_f.flush()

            # ── BATCH TIMING ─────────────────────────────────────────────
            t_elapsed      = time.time() - t_batch_start
            avg_per_batch  = (time.time() - start_total) / batch_num
            eta_seconds    = avg_per_batch * (total_batches - batch_num)
            logging.info(
                f"Batch {batch_num}/{total_batches} | "
                f"this={t_elapsed:.1f}s | avg={avg_per_batch:.1f}s | "
                f"ETA={eta_seconds / 60:.1f} min"
            )

            # ── MEMORY CLEANUP ───────────────────────────────────────────
            try:
                del gen_out, tok
            except Exception:
                pass
            torch.cuda.empty_cache()
            gc.collect()

    if n_parse_fail:
        logging.warning(
            f"⚠ {n_parse_fail}/{len(to_process)} examples failed confidence "
            "parsing — inspect raw_response in the output file."
        )


# ── MODEL-SPECIFIC ENTRY POINTS ─────────────────────────────────────────────

def compute_verb_unc_qwen(
    local_model_path: str,
    results_dir: str,
    dataset: str,
    datasplit: str,
    lang: str,
    batch_size: int,
):
    logging.info("Computing verbalised uncertainty with Qwen3 8B")

    answers_file = f"{results_dir}/{dataset}/qwen3_8B/{datasplit}/{lang}_answer.jsonl"
    out_file     = f"{results_dir}/{dataset}/qwen3_8B/{datasplit}/{lang}_verb_unc.jsonl"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    # ── TOKENIZER & MODEL ────────────────────────────────────────────────
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

    pad_id      = tokenizer.pad_token_id
    terminators = [tokenizer.eos_token_id]

    lang_config = LANG_CONFIGS.get(lang)
    if lang_config is None:
        logging.error(
            f"No verbalised-uncertainty prompt config for language '{lang}'. "
            f"Available: {list(LANG_CONFIGS.keys())}"
        )
        sys.exit(1)

    test_ids    = load_test_ids(results_dir, dataset, "qwen3_8B", datasplit)
    examples    = load_answers(answers_file, test_ids=test_ids)
    start_total = time.time()

    _run_verb_unc_loop(
        model, tokenizer, examples, out_file, batch_size,
        lang_config, model_type="qwen", terminators=terminators,
        pad_id=pad_id, start_total=start_total,
    )

    total_time = time.time() - start_total
    logging.info(
        f"\nDone. Total time: {total_time / 60:.1f} min ({total_time / 3600:.2f} h)"
    )
    logging.info(f"✅ Verbalised uncertainty scores saved to {out_file}")


def compute_verb_unc_llama(
    local_model_path: str,
    results_dir: str,
    dataset: str,
    datasplit: str,
    lang: str,
    batch_size: int,
):
    logging.info("Computing verbalised uncertainty with Llama 3.1 8B Instruct")

    answers_file = f"{results_dir}/{dataset}/llama_3.1_8B/{datasplit}/{lang}_answer.jsonl"
    out_file     = f"{results_dir}/{dataset}/llama_3.1_8B/{datasplit}/{lang}_verb_unc.jsonl"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    # ── TOKENIZER & MODEL ────────────────────────────────────────────────
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
    pad_id      = tokenizer.eos_token_id
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    lang_config = LANG_CONFIGS.get(lang)
    if lang_config is None:
        logging.error(
            f"No verbalised-uncertainty prompt config for language '{lang}'. "
            f"Available: {list(LANG_CONFIGS.keys())}"
        )
        sys.exit(1)

    test_ids    = load_test_ids(results_dir, dataset, "llama_3.1_8B", datasplit)
    examples    = load_answers(answers_file, test_ids=test_ids)
    start_total = time.time()

    _run_verb_unc_loop(
        model, tokenizer, examples, out_file, batch_size,
        lang_config, model_type="llama", terminators=terminators,
        pad_id=pad_id, start_total=start_total,
    )

    total_time = time.time() - start_total
    logging.info(
        f"\nDone. Total time: {total_time / 60:.1f} min ({total_time / 3600:.2f} h)"
    )
    logging.info(f"✅ Verbalised uncertainty scores saved to {out_file}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def build_parser():
    parser = argparse.ArgumentParser(
        description="Compute verbalised-uncertainty confidence scores "
                    "for pre-generated LLM answers."
    )
    parser.add_argument("--track-emissions", action="store_true", default=False)

    sub = parser.add_subparsers(dest="operation", required=True)

    p_compute = sub.add_parser(
        "compute_verb_unc",
        help="Run verbalised-uncertainty estimation for one "
             "model / dataset / datasplit / language.",
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
            output_file="verb_unc.csv",
        )
        tracker.start()

    try:
        dataset   = args.dataset_name
        datasplit = args.datasplit_name
        lang      = args.lang
        model     = args.model_name

        if model == "llama_3.1_8B":
            compute_verb_unc_llama(
                LLAMA_MODEL_PATH, RESULTS_DIR,
                dataset, datasplit, lang, BATCH_SIZE,
            )
        elif model == "qwen3_8B":
            compute_verb_unc_qwen(
                QWEN_MODEL_PATH, RESULTS_DIR,
                dataset, datasplit, lang, BATCH_SIZE,
            )
        else:
            logging.error(f"Unknown model: {model}")
            sys.exit(1)

    finally:
        if tracker is not None:
            tracker.stop()