#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
classification_baselines.py — Majority-class and prior-probability baselines.

Simplification over the original:
  * No HDF5 / hidden-state loading — only positive rates are needed.
    Labels are read from {datasplit}/{lang}_correctness.jsonl filtered to the
    test split defined by the same splits JSON as eval_probe.py / eval_p_true.py.
  * No seeds — constant predictions are deterministic; shuffling labels with
    any seed produces identical metric values.

Output schema is identical to the original so results are directly comparable.
"""

import json
import logging
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
TRAIN_LANGS = ['fr']
TEST_LANGS  = LANGUAGES

BASELINE = "prior-prob"   # "majority" or "prior-prob"

ALL_DATASETS   = ["mkqa", "global_mmlu"]
ALL_LLMS       = ["llama_3.1_8B", "qwen3_8B"]
ALL_QUERY_MODES = [True, False]

# Kept for schema alignment with eval_probe.py / original baselines CSV
L1_W   = 0
L2_V   = 0.0
EPOCHS = 0

STRATIFIED   = True
RESULTS_ROOT = Path(LOCAL_RESULTS_DIR)


# ── DATA LOADING (lightweight — no HDF5) ─────────────────────────────────────

def _splits_path(results_root: Path, ds: str, stratified: bool,
                 exclude_time_sensitive: bool) -> Path:
    """Reproduce the splits-file path from load_test_unflattened_from_hdf5."""
    lang = TRAIN_LANGS[0]
    if stratified:
        if exclude_time_sensitive:
            fname = f"train_lang_{lang}_splits_stratified_without_time_sensitive.json"
        else:
            fname = f"train_lang_{lang}_splits_stratified.json"
    else:
        fname = f"train_lang_{lang}_splits.json"
    return results_root / ds / fname


def load_split_ids(results_root: Path, datasplits: list, stratified: bool,
                   exclude_time_sensitive: bool) -> tuple[set, set]:
    """Return (train_ids, test_ids) across all valid datasplits."""
    train_ids, test_ids = set(), set()
    for ds in datasplits:
        sp = _splits_path(results_root, ds, stratified, exclude_time_sensitive)
        if not sp.exists():
            logging.warning(f"  [SKIP] Splits file not found: {sp}")
            continue
        splits = json.loads(sp.read_text())
        train_ids.update(str(eid) for eid in splits.get("train", []))
        test_ids.update(str(eid) for eid in splits.get("test", []))
    return train_ids, test_ids


def load_correctness(results_root: Path, datasplits: list, lang: str) -> dict:
    """
    Load {example_id: 0|1} correctness labels from JSONL files.
    Each file is a single-line JSON dict — same format as eval_p_true.py.
    """
    correctness = {}
    for ds in datasplits:
        path = results_root / ds / f"{lang}_correctness.jsonl"
        if not path.exists():
            logging.warning(f"  [SKIP] Correctness file not found: {path}")
            continue
        correctness.update(json.loads(path.read_text().strip().splitlines()[0]))
    return correctness


def positive_rate(correctness: dict, ids: set) -> float:
    """Fraction of examples in `ids` that have label 1."""
    labels = [int(correctness[eid]) for eid in ids if eid in correctness]
    return float(np.mean(labels)) if labels else 0.0


# ── METRICS ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, p_const: float) -> dict:
    """
    Compute AUROC, AUPR, Brier, ECE for constant predictions.

    With constant predictions every threshold gives the same operating point,
    so AUROC = 0.5 regardless of class balance.  AUPR equals the positive rate
    (no-skill baseline). Brier and ECE are computed numerically for exactness.
    """
    n = len(y_true)
    y_pred = np.full(n, p_const, dtype=np.float64)

    auroc = (
        roc_auc_score(y_true, y_pred)
        if len(np.unique(y_true)) > 1
        else float("nan")
    )
    brier = brier_score_loss(y_true, y_pred)
    ece   = compute_ece(y_true, y_pred, n_bins=10)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    aupr  = auc(recall, precision)

    return {"auroc": auroc, "aupr": aupr, "brier": brier, "ece": ece}


# ── EVALUATION ────────────────────────────────────────────────────────────────

def evaluate_combo(dataset: str, llm: str, query: bool) -> list[dict]:
    """Run both baselines for all test languages for one dataset/LLM/mode combo."""
    exclude_ts  = dataset == "mkqa"
    datasplits  = DATASPLITS_MKQA if dataset == "mkqa" else DATASPLITS_GMMLU
    results_root = RESULTS_ROOT / dataset / llm
    query_label  = "query" if query else "answer"
    probe_name   = (
        "MajorityClassBaseline" if BASELINE.lower().startswith("majority")
        else "PriorProbBaseline"
    )

    # ── training positive rate (defines majority class / prior) ───────────────
    train_ids, test_ids = load_split_ids(
        results_root, datasplits, STRATIFIED, exclude_ts
    )
    train_correctness = load_correctness(results_root, datasplits, TRAIN_LANGS[0])
    p_train = positive_rate(train_correctness, train_ids)
    maj_class = int(p_train >= 0.5)
    logging.info(
        f"  p_train={p_train:.4f}  majority_class={maj_class}"
    )

    p_const = (
        float(maj_class)                                   # majority baseline
        if BASELINE.lower().startswith("majority")
        else float(p_train)                                # prior-prob baseline
    )

    ratio = p_train  # used for skip check (mirrors original)
    if ratio == 0.0 or ratio == 1.0:
        logging.warning("  Skipping: single-class training labels.")
        return []

    rows = []
    for test_lang in sorted(TEST_LANGS):
        test_correctness = load_correctness(results_root, datasplits, test_lang)
        common_ids = test_ids & test_correctness.keys()
        if not common_ids:
            logging.warning(f"  No test examples for lang={test_lang}. Skipping.")
            continue

        y_test = np.array([int(test_correctness[eid]) for eid in sorted(common_ids)])
        n_pos  = int((y_test == 1).sum())
        n_neg  = len(y_test) - n_pos
        logging.info(
            f"  lang={test_lang}  N={len(y_test)}  pos={n_pos}  neg={n_neg}"
        )

        metrics = compute_metrics(y_test, p_const)

        rows.append({
            "seed":           0,           # deterministic; kept for schema compat
            "lang":           test_lang,
            "test_lang":      test_lang,
            "train_langs":    "-".join(sorted(TRAIN_LANGS)),
            "l1_w":           L1_W,
            "l2_v":           L2_V,
            "auroc":          metrics["auroc"],
            "brier":          metrics["brier"],
            "ece":            metrics["ece"],
            "aupr":           metrics["aupr"],
            "probe":          probe_name,
            "dataset":        dataset,
            "llm":            llm,
            "mode":           query_label,
            "p_train":        p_train,
            "majority_class": maj_class,
            "baseline_type":  BASELINE,
            "timestamp":      datetime.now(timezone.utc).isoformat(),
        })

    return rows


# ── OUTPUT PATH ───────────────────────────────────────────────────────────────

def get_output_csv_path() -> Path:
    probe_name = (
        "MajorityClassBaseline" if BASELINE.lower().startswith("majority")
        else "PriorProbBaseline"
    )
    train_tag = "_".join(sorted(TRAIN_LANGS))
    return RESULTS_ROOT / f"{probe_name}_train_{train_tag}_eval.csv"


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )

    TRAIN_LANGS.sort()
    TEST_LANGS.sort()

    combos = list(product(ALL_DATASETS, ALL_LLMS, ALL_QUERY_MODES))
    logging.info(
        f"Running {len(combos)} combinations: "
        f"{ALL_DATASETS} x {ALL_LLMS} x query={ALL_QUERY_MODES}"
    )

    all_rows = []
    for i, (dataset, llm, query) in enumerate(combos, 1):
        logging.info(
            f"{'='*60}\n[{i}/{len(combos)}] "
            f"dataset={dataset}  llm={llm}  mode={'query' if query else 'answer'}  "
            f"baseline={BASELINE}\n{'='*60}"
        )
        try:
            all_rows.extend(evaluate_combo(dataset, llm, query))
        except Exception as e:
            logging.exception(f"Evaluation failed for {dataset}/{llm}: {e}")
            sys.exit(1)

    if not all_rows:
        logging.error("No successful evaluations; nothing written.")
        sys.exit(1)

    combined = pd.DataFrame(all_rows)
    output_path = get_output_csv_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Per-evaluation results ────────────────────────────────────────────────
    to_write = combined.copy()
    for col in ["auroc", "brier", "ece", "aupr"]:
        to_write[col] = to_write[col].round(4)
    write_header = not output_path.exists()
    to_write.to_csv(output_path, mode="a", index=False, header=write_header)
    logging.info(f"Wrote {len(to_write)} rows → {output_path.resolve()}")

    # ── Summary (mean±std across languages, matching eval_probe.py style) ─────
    summary = (
        combined
        .groupby(
            ["probe", "dataset", "llm", "mode", "train_langs",
             "test_lang", "lang", "l1_w", "l2_v"],
            dropna=False, observed=False,
        )
        .agg(
            auroc_mean=("auroc", "mean"), auroc_std=("auroc", "std"),
            brier_mean=("brier", "mean"), brier_std=("brier", "std"),
            ece_mean  =("ece",   "mean"), ece_std  =("ece",   "std"),
            aupr_mean =("aupr",  "mean"), aupr_std =("aupr",  "std"),
        )
        .reset_index()
    )
    for m in ["auroc", "brier", "ece", "aupr"]:
        for col in [f"{m}_mean", f"{m}_std"]:
            summary[col] = summary[col].round(2)
        summary[m] = summary.apply(
            lambda r, m=m: (
                f"{r[f'{m}_mean']}±{r[f'{m}_std']}"
                if not pd.isna(r[f"{m}_std"])
                else f"{r[f'{m}_mean']}"
            ),
            axis=1,
        )

    summary_path = output_path.with_name(output_path.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)
    logging.info(f"Wrote summary → {summary_path.resolve()}")
    logging.info("All combinations complete.")


if __name__ == "__main__":
    main()