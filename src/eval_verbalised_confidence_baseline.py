#!/usr/bin/env python
"""
eval_verbalised_confidence_baseline.py — Evaluate the verbalised confidence baseline.

Evaluates all (dataset × LLM) combinations in one run, writing every
per-language row to a single CSV so results are directly comparable with
eval_probe.py, eval_ptrue_baseline.py, and classification_baselines.py.
"""

import json
import logging
import math
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import auc, brier_score_loss, precision_recall_curve, roc_auc_score

from utils import compute_ece
from constants import LOCAL_RESULTS_DIR, DATASPLITS_MKQA, DATASPLITS_GMMLU, LANGUAGES

# ── CONFIG ────────────────────────────────────────────────────────────────────
TRAIN_LANGS  = ['fr']
TEST_LANGS   = LANGUAGES
STRATIFIED   = True

ALL_DATASETS = ["mkqa", "global_mmlu"]
ALL_LLMS     = ["llama_3.1_8B", "qwen3_8B"]

LOCAL_RESULTS_DIR = Path(LOCAL_RESULTS_DIR)


# ── DATA LOADING ──────────────────────────────────────────────────────────────

def _splits_path(results_root: Path, ds: str, exclude_time_sensitive: bool) -> Path:
    lang = TRAIN_LANGS[0]
    if STRATIFIED:
        fname = (
            f"train_lang_{lang}_splits_stratified_without_time_sensitive.json"
            if exclude_time_sensitive
            else f"train_lang_{lang}_splits_stratified.json"
        )
    else:
        fname = f"train_lang_{lang}_splits.json"
    return results_root / ds / fname


def load_test_ids(results_root: Path, datasplits: list,
                  exclude_time_sensitive: bool) -> set:
    test_ids = set()
    for ds in datasplits:
        sp = _splits_path(results_root, ds, exclude_time_sensitive)
        if not sp.exists():
            logging.warning(f"  [SKIP] Splits file not found: {sp}")
            continue
        with sp.open("r", encoding="utf-8") as f:
            splits_dict = json.load(f)
        test_ids.update(str(eid) for eid in splits_dict["test"])
    logging.info(f"  Loaded {len(test_ids)} test IDs")
    return test_ids


def load_correctness(results_root: Path, datasplits: list, lang: str) -> dict:
    correctness = {}
    for ds in datasplits:
        corr_path = results_root / ds / f"{lang}_correctness.jsonl"
        if not corr_path.exists():
            logging.warning(f"  [SKIP] Correctness file not found: {corr_path}")
            continue
        with corr_path.open("r", encoding="utf-8") as fin:
            correctness.update(json.loads(fin.readline().strip()))
    return correctness


def load_verb_unc_scores(results_root: Path, datasplits: list, lang: str) -> dict:
    """
    Load verbalised-confidence scores from {lang}_verb_unc.jsonl files.

    Returns:
        dict mapping example_id → confidence_prob (float in [0, 1], or NaN).
    """
    scores = {}
    for ds in datasplits:
        jsonl_path = results_root / ds / f"{lang}_verb_unc.jsonl"
        if not jsonl_path.exists():
            logging.warning(f"  [SKIP] Verb-unc file not found: {jsonl_path}")
            continue
        with jsonl_path.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                scores[rec["example_id"]] = rec["confidence_prob"]
    logging.info(f"  Loaded {len(scores)} verb-unc scores (lang={lang})")
    return scores


# ── METRICS ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    auroc = (
        roc_auc_score(y_true, y_score)
        if len(np.unique(y_true)) > 1
        else float("nan")
    )
    brier = brier_score_loss(y_true, y_score)
    ece   = compute_ece(y_true, y_score, n_bins=10)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    aupr  = auc(recall, precision)
    return {"auroc": auroc, "aupr": aupr, "brier": brier, "ece": ece}


# ── EVALUATION ────────────────────────────────────────────────────────────────

def evaluate_lang(test_lang: str, test_ids: set,
                  results_root: Path, datasplits: list) -> dict | None:
    correctness = load_correctness(results_root, datasplits, test_lang)
    verb_scores = load_verb_unc_scores(results_root, datasplits, test_lang)

    common_ids    = test_ids & correctness.keys() & verb_scores.keys()
    missing_verb  = test_ids & correctness.keys() - verb_scores.keys()
    if missing_verb:
        logging.warning(
            f"  {len(missing_verb)} test examples have no verb-unc score "
            "(run verbal_confidence_baseline.py for this lang/dataset)."
        )
    if not common_ids:
        logging.error(f"  No aligned examples for lang={test_lang}. Skipping.")
        return None

    # Drop examples where parsing failed (confidence_prob is NaN)
    valid_ids = sorted(
        eid for eid in common_ids
        if not (isinstance(verb_scores[eid], float) and math.isnan(verb_scores[eid]))
    )
    n_dropped = len(common_ids) - len(valid_ids)
    if n_dropped:
        logging.warning(f"  Dropped {n_dropped} examples with NaN confidence.")
    if not valid_ids:
        logging.error(f"  No valid examples for lang={test_lang}. Skipping.")
        return None

    y_arr  = np.array([int(correctness[eid]) for eid in valid_ids])
    sc_arr = np.array([verb_scores[eid]      for eid in valid_ids])

    n_pos = int((y_arr == 1).sum())
    n_neg = int((y_arr == 0).sum())
    ratio = n_pos / len(y_arr)
    logging.info(f"  N={len(y_arr)}  pos={n_pos}  neg={n_neg}  ratio={ratio:.4f}")

    if ratio == 0.0 or ratio == 1.0:
        logging.warning(f"  Skipping: single-class labels (ratio={ratio:.2f})")
        return None

    metrics = compute_metrics(y_arr, sc_arr)
    logging.info(
        f"  AUROC={metrics['auroc']:.4f}  AUPR={metrics['aupr']:.4f}  "
        f"Brier={metrics['brier']:.4f}  ECE={metrics['ece']:.4f}"
    )
    return metrics


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )

    output_path = LOCAL_RESULTS_DIR / f"verb_unc_baseline_train_{'_'.join(TRAIN_LANGS)}_eval.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows = []
    combos   = list(product(ALL_DATASETS, ALL_LLMS))

    for i, (dataset, llm) in enumerate(combos, 1):
        exclude_ts   = dataset == "mkqa"
        datasplits   = DATASPLITS_MKQA if dataset == "mkqa" else DATASPLITS_GMMLU
        results_root = LOCAL_RESULTS_DIR / dataset / llm

        logging.info("=" * 60)
        logging.info(f"[{i}/{len(combos)}] dataset={dataset}  llm={llm}")
        logging.info("=" * 60)

        test_ids = load_test_ids(results_root, datasplits, exclude_ts)

        for test_lang in TEST_LANGS:
            logging.info(f"── lang={test_lang} ──")
            metrics = evaluate_lang(test_lang, test_ids, results_root, datasplits)
            if metrics is None:
                continue
            all_rows.append({
                "lang":        test_lang,
                "test_lang":   test_lang,
                "train_langs": "-".join(TRAIN_LANGS),
                "dataset":     dataset,
                "llm":         llm,
                "method":      "verb_unc",
                "auroc":       round(metrics["auroc"], 4),
                "aupr":        round(metrics["aupr"],  4),
                "brier":       round(metrics["brier"], 4),
                "ece":         round(metrics["ece"],   4),
                "timestamp":   datetime.now(timezone.utc).isoformat(),
            })

    if not all_rows:
        logging.error("No successful evaluations. Nothing written.")
        sys.exit(1)

    eval_df = pd.DataFrame(all_rows)

    # ── Per-language results ──────────────────────────────────────────────────
    write_header = not output_path.exists()
    eval_df.to_csv(output_path, mode="a", index=False, header=write_header)
    logging.info(f"Wrote {len(eval_df)} rows → {output_path.resolve()}")

    # ── Summary: mean±std across languages per (dataset, llm) ────────────────
    summary = (
        eval_df
        .groupby(["dataset", "llm", "train_langs", "method"], observed=False)
        .agg(
            n_langs    = ("test_lang", "count"),
            auroc_mean = ("auroc", "mean"), auroc_std  = ("auroc", "std"),
            aupr_mean  = ("aupr",  "mean"), aupr_std   = ("aupr",  "std"),
            brier_mean = ("brier", "mean"), brier_std  = ("brier", "std"),
            ece_mean   = ("ece",   "mean"), ece_std    = ("ece",   "std"),
        )
        .reset_index()
    )
    for m in ["auroc", "aupr", "brier", "ece"]:
        summary[m] = summary.apply(
            lambda r, m=m: (
                f"{round(r[f'{m}_mean'], 2)}±{round(r[f'{m}_std'], 2)}"
                if not pd.isna(r[f"{m}_std"])
                else f"{round(r[f'{m}_mean'], 2)}"
            ),
            axis=1,
        )
    summary_path = output_path.with_name(output_path.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)
    logging.info(f"Wrote summary → {summary_path.resolve()}")


if __name__ == "__main__":
    main()