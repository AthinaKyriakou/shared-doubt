"""
Mass-Mean Probe baseline (Marks & Tegmark, 2024) 
The implementation is aligned with the SoftmaxLayerProbe pipeline 
so that the ONLY difference is the probe itself.

Key alignment guarantees vs. train_probe.py / eval_probe.py:
  1. Same data source:  merged HDF5 last-token hidden states
  2. Same train/val/test splits:  JSON split files (stratified)
  3. Same per-layer StandardScaler normalisation (fitted on train)
  4. Same test-set loading via load_test_unflattened_from_hdf5
  5. Same evaluation metrics:  AUROC, Brier, ECE, AUPR
  6. Same class-balance handling as eval_probe.py

Implements the non-IID variant (no Mahalanobis whitening), matching
MMProbe.forward(x, iid=False) from the original repo:
    d  = mean(h_correct) − mean(h_incorrect)        [unnormalised]
    score(h) = sigmoid(h · d)

Usage:
    python mass_mean_probe.py --seeds 17 3 42 59 76 169 369 445 884 905 --layer -1 
"""

import argparse
import logging
import sys
import random
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss, precision_recall_curve, auc
from sklearn.isotonic import IsotonicRegression

from utils import (
    load_and_preprocess_unflattened_from_hdf5,
    load_test_unflattened_from_hdf5,
    compute_ece,
)
from constants import DATASPLITS_MKQA, DATASPLITS_GMMLU, LOCAL_RESULTS_DIR

# ── CONFIG (mirror eval_probe.py) ─────────────────────────────────────────────

TRAIN_LANGS = ['fr']
TEST_LANGS  = ['en', 'es', 'ja', 'pl', 'ru']
ISOTONIC    = False  

# DATASET = "mkqa"
DATASET = "global_mmlu"
LLM = "llama_3.1_8B"
# LLM = "qwen3_8B"
QUERY = False  

EXCLUDE_TIME_SENSITIVE = True if DATASET == 'mkqa' else False
DATASPLITS  = DATASPLITS_MKQA if DATASET == 'mkqa' else DATASPLITS_GMMLU
NUM_LAYERS  = 32 if LLM == 'llama_3.1_8B' else 36
HIDDEN_DIM  = 4096
STRATIFIED  = True

RESULTS_ROOT = Path(LOCAL_RESULTS_DIR) / DATASET / LLM

# ── HELPERS ───────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_mass_mean_direction(X: np.ndarray, y: np.ndarray, layer: int) -> np.ndarray:
    """
    Compute the mass-mean truth direction on a single layer.

    Args:
        X:     (N, num_layers, hidden_dim)  — StandardScaler-normalised
        y:     (N,)  — binary labels (1 = correct, 0 = incorrect)
        layer: which layer index to use

    Returns:
        direction: (hidden_dim,) numpy array — unnormalised difference in means
    """
    h = X[:, layer, :]                          # (N, hidden_dim)
    correct_mask = (y == 1)
    mu_correct   = h[correct_mask].mean(axis=0)
    mu_incorrect = h[~correct_mask].mean(axis=0)
    direction    = mu_correct - mu_incorrect     # unnormalised, matching MMProbe
    return direction


def score_mass_mean(X: np.ndarray, direction: np.ndarray, layer: int) -> np.ndarray:
    """
    Score examples with sigmoid(h · d).

    Args:
        X:         (N, num_layers, hidden_dim)
        direction: (hidden_dim,)
        layer:     layer index

    Returns:
        scores: (N,) in (0, 1)
    """
    h = X[:, layer, :]                                          # (N, hidden_dim)
    h_t = torch.from_numpy(h).float()
    d_t = torch.from_numpy(direction).float()
    scores = torch.sigmoid(h_t @ d_t).numpy()                  # (N,)
    return scores


def evaluate_for_seed(
    seed, direction, layer, device,
    dataset, datasplits, results_root,
    train_langs, test_langs,
    query, epochs, stratified, exclude_time_sensitive,
    iso_reg=None,
):
    """Run few-shot + zero-shot evaluation, mirroring eval_probe.evaluate_for_seed."""

    set_seed(seed)
    results = {}

    def calibrate(p):
        p = p.reshape(-1)
        if iso_reg is not None:
            return iso_reg.transform(p)
        return p

    # ── Few-shot (test split of training language) ────────────────────────────
    logging.info(f"[seed={seed}] Few-shot evaluation")
    X_test_few, y_test_few = load_test_unflattened_from_hdf5(
        dataset, datasplits, results_root, train_langs, train_langs[0],
        query, epochs, stratified, exclude_time_sensitive,
    )
    y_test_few = np.asarray(y_test_few)
    y_pred_prob_few = calibrate(score_mass_mean(X_test_few, direction, layer))

    auroc_few = roc_auc_score(y_test_few, y_pred_prob_few) if len(np.unique(y_test_few)) > 1 else float("nan")
    brier_few = brier_score_loss(y_test_few, y_pred_prob_few)
    ece_few   = compute_ece(y_test_few, y_pred_prob_few, n_bins=10)
    prec_few, rec_few, _ = precision_recall_curve(y_test_few, y_pred_prob_few)
    aupr_few  = auc(rec_few, prec_few)

    n_pos_few = int(y_test_few.sum())
    n_neg_few = len(y_test_few) - n_pos_few
    ratio = n_pos_few / (n_pos_few + n_neg_few) if (n_pos_few + n_neg_few) > 0 else 0.0
    logging.info(f"[seed={seed}] Few-shot class balance — 0s: {n_neg_few}, 1s: {n_pos_few}, ratio: {ratio:.4f}")

    results['few-shot'] = {
        "auroc": auroc_few, "brier": brier_few, "ece": ece_few, "aupr": aupr_few,
    }

    # ── Zero-shot across test languages ───────────────────────────────────────
    for test_lang in test_langs:
        logging.info(f"[seed={seed}] Evaluating {test_lang}")
        X_test, y_test = load_test_unflattened_from_hdf5(
            dataset, datasplits, results_root, train_langs, test_lang,
            query, epochs, stratified, exclude_time_sensitive,
        )
        X_test = np.asarray(X_test)
        y_test = np.asarray(y_test)

        # Same class-balance handling as eval_probe.py (current: use all data, no resampling)
        pos_mask = (y_test == 1)
        neg_mask = (y_test == 0)
        X_pos, y_pos = X_test[pos_mask], y_test[pos_mask]
        X_neg, y_neg = X_test[neg_mask], y_test[neg_mask]

        if ratio == 0 or ratio == 1:
            logging.warning(f"[seed={seed}] Skipping {test_lang}: few-shot ratio is {ratio:.2f}")
            continue

        X_bal = np.concatenate([X_pos, X_neg], axis=0)
        y_bal = np.concatenate([y_pos, y_neg], axis=0)
        perm  = np.random.RandomState(seed).permutation(len(y_bal))
        X_bal = X_bal[perm]
        y_bal = y_bal[perm]
        y_test_arr = np.array(y_bal)

        num_ones  = int((y_test_arr == 1).sum())
        num_zeros = int((y_test_arr == 0).sum())
        logging.info(f"[seed={seed}] Class balance ({test_lang}) — 0s: {num_zeros}, 1s: {num_ones}")

        y_pred_prob = calibrate(score_mass_mean(X_bal, direction, layer))

        auroc = roc_auc_score(y_test_arr, y_pred_prob) if len(np.unique(y_test_arr)) > 1 else float("nan")
        brier = brier_score_loss(y_test_arr, y_pred_prob)
        ece   = compute_ece(y_test_arr, y_pred_prob, n_bins=10)
        prec, rec, _ = precision_recall_curve(y_test_arr, y_pred_prob)
        aupr  = auc(rec, prec)

        results[test_lang] = {"auroc": auroc, "brier": brier, "ece": ece, "aupr": aupr}

    # ── Assemble DataFrame ────────────────────────────────────────────────────
    rows = []
    for lang, metrics in results.items():
        test_lang_field = lang if lang != "few-shot" else train_langs[0]
        rows.append({
            "seed":       seed,
            "lang":       lang,
            "layer":      layer,
            "auroc":      metrics["auroc"],
            "brier":      metrics["brier"],
            "ece":        metrics["ece"],
            "aupr":       metrics["aupr"],
            "probe":      "MassMeanProbe",
            "dataset":    DATASET,
            "llm":        LLM,
            "train_langs": "-".join(train_langs),
            "test_lang":  test_lang_field,
            "timestamp":  datetime.now(timezone.utc).isoformat(),
        })

    return pd.DataFrame(rows)


# ── OUTPUT PATH ───────────────────────────────────────────────────────────────

def make_output_csv(layer: int, isotonic: bool) -> Path:
    train_tag = '_'.join(TRAIN_LANGS)
    if EXCLUDE_TIME_SENSITIVE:
        base_dir = (RESULTS_ROOT / "probe_query_without_time_sensitive") if QUERY else (RESULTS_ROOT / "probe_answer_without_time_sensitive")
    else:
        base_dir = (RESULTS_ROOT / "probe_query") if QUERY else (RESULTS_ROOT / "probe_answer")
    subdir = f"MassMeanProbe/train_{train_tag}"
    suffix = "multiseed_isotonic_eval.csv" if isotonic else "multiseed_eval.csv"
    fname  = f"layer_{layer}_{suffix}"
    return base_dir / subdir / fname


# ── MAIN ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Mass-Mean Probe evaluation aligned with SoftmaxLayerProbe pipeline."
    )
    parser.add_argument("--seeds",  type=int, nargs="+", required=True)
    parser.add_argument(
        "--layer", type=int, default=-1,
        help="Layer to probe (0-indexed). -1 = middle layer (default: -1).",
    )
    parser.add_argument(
        "--both", action="store_true",
        help="Run both non-isotonic and isotonic evaluations.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve layer
    layer = args.layer
    if layer == -1:
        layer = NUM_LAYERS // 2
    logging.info(f"Probing layer {layer}/{NUM_LAYERS}")

    # Use a dummy epochs tag consistent with your pipeline
    epochs_tag = "htune"

    modes = [False, True] if args.both else [ISOTONIC]

    for isotonic in modes:
        output_path = make_output_csv(layer, isotonic)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        eval_dfs = []

        for seed in args.seeds:
            set_seed(seed)
            logging.info(f"\n{'='*60}")
            logging.info(f"Seed: {seed} | Isotonic: {isotonic}")
            logging.info(f"{'='*60}")

            # ── Load data using the SAME function as train_probe.py ───────────
            X_tr, y_tr, X_va, y_va = load_and_preprocess_unflattened_from_hdf5(
                dataset=DATASET,
                datasplits=DATASPLITS,
                results_root=RESULTS_ROOT,
                train_languages=TRAIN_LANGS,
                query=QUERY,
                epochs=epochs_tag,
                stratified=STRATIFIED,
                exclude_time_sensitive=EXCLUDE_TIME_SENSITIVE,
            )
            y_tr = np.asarray(y_tr)
            y_va = np.asarray(y_va)

            # ── Fit mass-mean direction on training split ─────────────────────
            direction = fit_mass_mean_direction(X_tr, y_tr, layer)
            logging.info(f"Direction norm: {np.linalg.norm(direction):.4f}")
            logging.info(f"Train: {int(y_tr.sum())} correct, {int((y_tr == 0).sum())} incorrect")

            # ── Optional isotonic calibration on validation split ─────────────
            iso_reg = None
            if isotonic:
                prob_va = score_mass_mean(X_va, direction, layer)
                y_va_np = np.asarray(y_va)
                if len(np.unique(y_va_np)) >= 2 and np.unique(prob_va).size >= 3:
                    iso_reg = IsotonicRegression(out_of_bounds="clip")
                    iso_reg.fit(prob_va, y_va_np)
                    logging.info("Fitted isotonic calibrator on validation set")
                else:
                    logging.warning("Skipping isotonic: val set has single class or too few distinct probs")

            # ── Evaluate ──────────────────────────────────────────────────────
            try:
                df_eval = evaluate_for_seed(
                    seed=seed,
                    direction=direction,
                    layer=layer,
                    device=device,
                    dataset=DATASET,
                    datasplits=DATASPLITS,
                    results_root=RESULTS_ROOT,
                    train_langs=TRAIN_LANGS,
                    test_langs=TEST_LANGS,
                    query=QUERY,
                    epochs=epochs_tag,
                    stratified=STRATIFIED,
                    exclude_time_sensitive=EXCLUDE_TIME_SENSITIVE,
                    iso_reg=iso_reg,
                )
                eval_dfs.append(df_eval)
            except Exception as e:
                logging.exception(f"[seed={seed}] Evaluation failed: {e}")
                sys.exit(1)

        if not eval_dfs:
            logging.error("No successful evaluations; nothing written.")
            continue

        combined = pd.concat(eval_dfs, ignore_index=True)

        # ── Write per-seed results ────────────────────────────────────────────
        to_write = combined.copy()
        for col in ["auroc", "brier", "ece", "aupr"]:
            to_write[col] = to_write[col].round(2)
        write_header = not output_path.exists()
        to_write.to_csv(output_path, mode="a", index=False, header=write_header)
        logging.info(f"[{'ISOTONIC' if isotonic else 'NO-ISOTONIC'}] Wrote {len(to_write)} rows → {output_path.resolve()}")

        # ── Summary across seeds ─────────────────────────────────────────────
        summary = (
            combined.groupby(
                ["probe", "dataset", "llm", "train_langs", "test_lang", "lang", "layer"],
                dropna=False, observed=False,
            )
            .agg(
                count_seeds=("seed", "nunique"),
                seeds=("seed", lambda x: ",".join(str(s) for s in sorted(set(x)))),
                auroc_mean=("auroc", "mean"), auroc_std=("auroc", "std"),
                brier_mean=("brier", "mean"), brier_std=("brier", "std"),
                ece_mean=("ece", "mean"),     ece_std=("ece", "std"),
                aupr_mean=("aupr", "mean"),   aupr_std=("aupr", "std"),
            )
            .reset_index()
        )
        for col in ["auroc_mean", "auroc_std", "brier_mean", "brier_std",
                     "ece_mean", "ece_std", "aupr_mean", "aupr_std"]:
            summary[col] = summary[col].round(2)

        summary["auroc"] = summary.apply(
            lambda r: f"{r['auroc_mean']}±{r['auroc_std']}" if not pd.isna(r["auroc_std"]) else f"{r['auroc_mean']}", axis=1)
        summary["brier"] = summary.apply(
            lambda r: f"{r['brier_mean']}±{r['brier_std']}" if not pd.isna(r["brier_std"]) else f"{r['brier_mean']}", axis=1)
        summary["ece"] = summary.apply(
            lambda r: f"{r['ece_mean']}±{r['ece_std']}" if not pd.isna(r["ece_std"]) else f"{r['ece_mean']}", axis=1)
        summary["aupr"] = summary.apply(
            lambda r: f"{r['aupr_mean']}±{r['aupr_std']}" if not pd.isna(r["aupr_std"]) else f"{r['aupr_mean']}", axis=1)

        summary_path = output_path.with_name(output_path.stem + "_summary.csv")
        summary.to_csv(summary_path, index=False)
        logging.info(f"[{'ISOTONIC' if isotonic else 'NO-ISOTONIC'}] Summary → {summary_path.resolve()}")


if __name__ == "__main__":
    main()