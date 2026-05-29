#!/usr/bin/env python
"""
eval_uniform_probe.py — Evaluate all trained UniformLayerProbe configurations.

Evaluates every (dataset × LLM × query-mode) combination across all seeds,
writing every per-seed row to a single CSV so results can be directly compared
with eval_probe.py output.

Differences from eval_probe.py:
  * All 8 hyperparameter configurations are evaluated in one run via CONFIGS.
  * Probe is UniformLayerProbe — layer weights are fixed at 1/num_layers, so
    compute_layer_importance simply returns model.w (no softmax/sparsemax).
  * lr and epochs columns are added to each row to identify the configuration.
  * No isotonic calibration (not needed for uniform-weight baseline).
"""

import argparse
import logging
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import auc, brier_score_loss, precision_recall_curve, roc_auc_score

from constants import DATASPLITS_GMMLU, DATASPLITS_MKQA, LANGUAGES, LOCAL_RESULTS_DIR
from probes import UniformLayerProbe
from utils import compute_ece, load_test_unflattened_from_hdf5

# ── FIXED CONFIG ──────────────────────────────────────────────────────────────
PROBE          = "UniformLayerProbe"
TRAIN_LANGS    = ['fr']
TEST_LANGS     = LANGUAGES
SEEDS          = [42, 17, 3, 59, 884, 445, 369, 169, 76, 90]
TRAIN_BATCH_SIZE = 32
L1_W           = 0
STRATIFIED     = True
LOCAL_RESULTS_DIR = Path(LOCAL_RESULTS_DIR)

NUM_LAYERS = {
    "llama_3.1_8B": 32,
    "qwen3_8B":     36,
}
HIDDEN_DIM = 4096

# ── HYPERPARAMETER CONFIGURATIONS ────────────────────────────────────────────
# One entry per (dataset, llm, query) combination, matching the
# commented-in/out blocks in the training script exactly.
CONFIGS: dict = {
    # ── MKQA ─────────────────────────────────────────────────────────────────
    ("mkqa", "llama_3.1_8B", True):  {"lr": 0.001,  "epochs": 122, "l2_v": 0.28748733353801853},
    ("mkqa", "llama_3.1_8B", False): {"lr": 0.0009, "epochs": 500, "l2_v": 0.24085571039248585},
    ("mkqa", "qwen3_8B",     True):  {"lr": 0.001,  "epochs": 144, "l2_v": 0.24807848114946277},
    ("mkqa", "qwen3_8B",     False): {"lr": 0.002,  "epochs": 83,  "l2_v": 0.16474944313861403},
    # ── GLOBAL MMLU ──────────────────────────────────────────────────────────
    ("global_mmlu", "llama_3.1_8B", True):  {"lr": 0.007, "epochs": 29,  "l2_v": 0.23215243005888206},
    ("global_mmlu", "llama_3.1_8B", False): {"lr": 0.001, "epochs": 123, "l2_v": 0.28588330922438193},
    ("global_mmlu", "qwen3_8B",     True):  {"lr": 0.001, "epochs": 131, "l2_v": 0.261559514194021},
    ("global_mmlu", "qwen3_8B",     False): {"lr": 0.001, "epochs": 142, "l2_v": 0.22318202430751768},
}


# ── HELPERS ───────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def probe_dir(results_root: Path, query: bool, exclude_time_sensitive: bool,
              lr: float, l2_v: float, epochs: int, seed: int) -> Path:
    """Reproduce the PROBE_DIR path from the training script."""
    mode = "query" if query else "answer"
    suffix = "_without_time_sensitive" if exclude_time_sensitive else ""
    base = results_root / f"probe_{mode}{suffix}/{PROBE}_layer_scalers"
    name = (
        f"train_{'_'.join(TRAIN_LANGS)}/"
        f"batch_{TRAIN_BATCH_SIZE}_lr_{lr}_l1_{L1_W}_l2_{l2_v}"
        f"_epochs_{epochs}_seed_{seed}"
    )
    return base / name


def compute_layer_importance(model: torch.nn.Module) -> np.ndarray:
    """
    For UniformLayerProbe, model.w is a fixed buffer already set to 1/num_layers.
    Return it directly — no softmax/sparsemax needed.
    """
    with torch.no_grad():
        return model.w.detach().cpu().numpy().reshape(-1)


# ── EVALUATION FOR ONE SEED / ONE CONFIG ─────────────────────────────────────

def evaluate_for_seed(
    seed: int,
    model: torch.nn.Module,
    device,
    dataset: str,
    llm: str,
    datasplits: list,
    results_root: Path,
    query: bool,
    epochs: int,
    lr: float,
    l2_v: float,
    exclude_time_sensitive: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Mirrors eval_probe.py evaluate_for_seed exactly:
      - Few-shot evaluation on the training language
      - Zero-shot evaluation across all TEST_LANGS
    Returns (eval_df, layer_df).
    """
    set_seed(seed)
    results = {}
    query_label = "query" if query else "answer"

    # ── Few-shot ──────────────────────────────────────────────────────────────
    logging.info(f"[seed={seed}] Few-shot")
    X_few, y_few = load_test_unflattened_from_hdf5(
        dataset, datasplits, results_root, TRAIN_LANGS, TRAIN_LANGS[0],
        query, str(epochs), STRATIFIED, exclude_time_sensitive,
    )
    X_few = torch.from_numpy(X_few).float().to(device)
    y_few = np.array(y_few)
    with torch.no_grad():
        y_pred_few = torch.sigmoid(model(X_few)).cpu().numpy().reshape(-1)

    auroc_few = roc_auc_score(y_few, y_pred_few) if len(np.unique(y_few)) > 1 else float("nan")
    brier_few = brier_score_loss(y_few, y_pred_few)
    ece_few   = compute_ece(y_few, y_pred_few, n_bins=10)
    prec_few, rec_few, _ = precision_recall_curve(y_few, y_pred_few)
    aupr_few  = auc(rec_few, prec_few)

    n_pos = int(y_few.sum())
    n_neg = len(y_few) - n_pos
    ratio = n_pos / (n_pos + n_neg) if (n_pos + n_neg) > 0 else 0.0
    logging.info(f"[seed={seed}] Few-shot — 0s: {n_neg}, 1s: {n_pos}, ratio: {ratio:.4f}")
    results["few-shot"] = {"auroc": auroc_few, "brier": brier_few, "ece": ece_few, "aupr": aupr_few}

    del X_few
    torch.cuda.empty_cache()

    # ── Zero-shot across test languages ───────────────────────────────────────
    for test_lang in sorted(TEST_LANGS):
        logging.info(f"[seed={seed}] Evaluating {test_lang} (ratio={ratio:.4f})")
        X_test, y_test = load_test_unflattened_from_hdf5(
            dataset, datasplits, results_root, TRAIN_LANGS, test_lang,
            query, str(epochs), STRATIFIED, exclude_time_sensitive,
        )
        X_test = np.asarray(X_test)
        y_test = np.asarray(y_test)

        if ratio == 0 or ratio == 1:
            logging.warning(f"[seed={seed}] Skipping {test_lang} — single-class few-shot ratio.")
            continue

        pos_mask = (y_test == 1)
        neg_mask = (y_test == 0)
        X_bal = np.concatenate([X_test[pos_mask], X_test[neg_mask]], axis=0)
        y_bal = np.concatenate([y_test[pos_mask], y_test[neg_mask]], axis=0)
        perm  = np.random.RandomState(seed).permutation(len(y_bal))
        X_bal, y_bal = X_bal[perm], y_bal[perm]

        logging.info(
            f"[seed={seed}] {test_lang} — "
            f"0s: {int((y_bal == 0).sum())}, 1s: {int((y_bal == 1).sum())}"
        )

        with torch.no_grad():
            X_t = torch.from_numpy(X_bal).float().to(device)
            y_pred = torch.sigmoid(model(X_t)).cpu().numpy().reshape(-1)

        auroc = roc_auc_score(y_bal, y_pred) if len(np.unique(y_bal)) > 1 else float("nan")
        brier = brier_score_loss(y_bal, y_pred)
        ece   = compute_ece(y_bal, y_pred, n_bins=10)
        prec, rec, _ = precision_recall_curve(y_bal, y_pred)
        aupr  = auc(rec, prec)

        results[test_lang] = {"auroc": auroc, "brier": brier, "ece": ece, "aupr": aupr}

        del X_bal, X_t
        torch.cuda.empty_cache()

    # ── Assemble eval DataFrame ───────────────────────────────────────────────
    rows = []
    for lang, m in results.items():
        test_lang_field = lang if lang != "few-shot" else TRAIN_LANGS[0]
        rows.append({
            "seed":        seed,
            "lang":        lang,
            "test_lang":   test_lang_field,
            "train_langs": "-".join(TRAIN_LANGS),
            "l1_w":        L1_W,
            "l2_v":        l2_v,
            "lr":          lr,
            "epochs":      epochs,
            "auroc":       m["auroc"],
            "brier":       m["brier"],
            "ece":         m["ece"],
            "aupr":        m["aupr"],
            "probe":       PROBE,
            "dataset":     dataset,
            "llm":         llm,
            "mode":        query_label,
            "timestamp":   datetime.now(timezone.utc).isoformat(),
        })
    eval_df = pd.DataFrame(rows)

    # ── Assemble layer importance DataFrame ───────────────────────────────────
    layer_imp = compute_layer_importance(model)
    layer_df  = pd.DataFrame([
        {
            "seed":        seed,
            "probe":       PROBE,
            "dataset":     dataset,
            "llm":         llm,
            "mode":        query_label,
            "train_langs": "-".join(TRAIN_LANGS),
            "l1_w":        L1_W,
            "l2_v":        l2_v,
            "lr":          lr,
            "epochs":      epochs,
            "layer":       i,
            "importance":  float(imp),
        }
        for i, imp in enumerate(layer_imp)
    ])

    return eval_df, layer_df


# ── MAIN ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate all UniformLayerProbe configs across seeds."
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=SEEDS,
        help="Seeds to evaluate (default: all 10 training seeds).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_path      = LOCAL_RESULTS_DIR / f"{PROBE}_train_{'_'.join(sorted(TRAIN_LANGS))}_multiseed_eval.csv"
    layer_imp_path   = LOCAL_RESULTS_DIR / f"{PROBE}_train_{'_'.join(sorted(TRAIN_LANGS))}_layer_importance.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_eval_rows  = []
    all_layer_rows = []

    total = len(CONFIGS) * len(args.seeds)
    done  = 0

    for (dataset, llm, query), hp in CONFIGS.items():
        lr, epochs, l2_v = hp["lr"], hp["epochs"], hp["l2_v"]
        exclude_ts   = dataset == "mkqa"
        datasplits   = DATASPLITS_MKQA if dataset == "mkqa" else DATASPLITS_GMMLU
        results_root = LOCAL_RESULTS_DIR / dataset / llm
        num_layers   = NUM_LAYERS[llm]
        query_label  = "query" if query else "answer"

        logging.info("=" * 70)
        logging.info(
            f"Config: dataset={dataset}  llm={llm}  mode={query_label}  "
            f"lr={lr}  epochs={epochs}  l2_v={l2_v}"
        )
        logging.info("=" * 70)

        for seed in args.seeds:
            done += 1
            logging.info(f"[{done}/{total}] seed={seed}")

            pdir = probe_dir(results_root, query, exclude_ts, lr, l2_v, epochs, seed)
            model_path = pdir / "probe_state.pt"
            if not model_path.exists():
                logging.error(f"Model not found: {model_path}. Skipping.")
                continue

            model = UniformLayerProbe(num_layers=num_layers, hidden_dim=HIDDEN_DIM)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()

            try:
                eval_df, layer_df = evaluate_for_seed(
                    seed=seed, model=model, device=device,
                    dataset=dataset, llm=llm, datasplits=datasplits,
                    results_root=results_root, query=query,
                    epochs=epochs, lr=lr, l2_v=l2_v,
                    exclude_time_sensitive=exclude_ts,
                )
                all_eval_rows.append(eval_df)
                all_layer_rows.append(layer_df)
            except Exception as e:
                logging.exception(f"Evaluation failed for {dataset}/{llm}/{query_label}/seed={seed}: {e}")
                sys.exit(1)
            finally:
                del model
                torch.cuda.empty_cache()

    if not all_eval_rows:
        logging.error("No successful evaluations. Nothing written.")
        sys.exit(1)

    combined   = pd.concat(all_eval_rows,  ignore_index=True)
    layer_comb = pd.concat(all_layer_rows, ignore_index=True)

    # ── Per-seed results ──────────────────────────────────────────────────────
    to_write = combined.copy()
    for col in ["auroc", "brier", "ece", "aupr"]:
        to_write[col] = to_write[col].round(4)
    write_header = not output_path.exists()
    to_write.to_csv(output_path, mode="a", index=False, header=write_header)
    logging.info(f"Wrote {len(to_write)} rows → {output_path.resolve()}")

    # ── Summary: mean±std across seeds ───────────────────────────────────────
    summary = (
        combined
        .groupby(
            ["probe", "dataset", "llm", "mode", "train_langs",
             "test_lang", "lang", "l1_w", "l2_v", "lr", "epochs"],
            dropna=False, observed=False,
        )
        .agg(
            count_seeds = ("seed", "nunique"),
            seeds       = ("seed", lambda x: ",".join(str(s) for s in sorted(set(x)))),
            auroc_mean  = ("auroc", "mean"), auroc_std  = ("auroc", "std"),
            brier_mean  = ("brier", "mean"), brier_std  = ("brier", "std"),
            ece_mean    = ("ece",   "mean"), ece_std    = ("ece",   "std"),
            aupr_mean   = ("aupr",  "mean"), aupr_std   = ("aupr",  "std"),
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

    # ── Layer importance ──────────────────────────────────────────────────────
    write_header_layer = not layer_imp_path.exists()
    layer_comb.to_csv(layer_imp_path, mode="a", index=False, header=write_header_layer)

    layer_summary = (
        layer_comb
        .groupby(
            ["probe", "dataset", "llm", "mode", "train_langs",
             "l1_w", "l2_v", "lr", "epochs", "layer"],
            dropna=False, observed=False,
        )
        .agg(
            count_seeds     = ("seed", "nunique"),
            importance_mean = ("importance", "mean"),
            importance_std  = ("importance", "std"),
        )
        .reset_index()
    )
    for col in ["importance_mean", "importance_std"]:
        layer_summary[col] = layer_summary[col].round(4)
    layer_summary["importance"] = layer_summary.apply(
        lambda r: (
            f"{r['importance_mean']}±{r['importance_std']}"
            if not pd.isna(r["importance_std"])
            else f"{r['importance_mean']}"
        ),
        axis=1,
    )
    layer_summary_path = layer_imp_path.with_name(layer_imp_path.stem + "_summary.csv")
    layer_summary.to_csv(layer_summary_path, index=False)
    logging.info(f"Wrote layer importance → {layer_imp_path.resolve()}")
    logging.info("All configurations complete.")


if __name__ == "__main__":
    main()