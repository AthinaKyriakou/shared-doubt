#!/usr/bin/env python
import argparse
import torch
from pathlib import Path
from utils import load_test_unflattened_from_hdf5, compute_ece, load_and_preprocess_unflattened_from_hdf5
from sklearn.metrics import roc_auc_score, brier_score_loss, precision_recall_curve, auc
import numpy as np
import pandas as pd
from sklearn.utils import resample
import random
import logging
import sys
from datetime import datetime, timezone
import torch.nn.functional as F
import joblib
from probes import SparsemaxLayerProbe, SoftmaxLayerProbe
from sparsemax import Sparsemax
from sklearn.isotonic import IsotonicRegression

from constants import LOCAL_RESULTS_DIR, DATASPLITS_MKQA, DATASPLITS_GMMLU

# ---- CONFIG --------
PROBE = "SoftmaxLayerProbe"
L1_W = 0

TRAIN_LANGS = ['fr'] 
TEST_LANGS = ['es']
ISOTONIC = False  # default mode if --both is not used

DATASET = "mkqa"
# DATASET = "global_mmlu"
LLM = "llama_3.1_8B"
# LLM = "qwen3_8B"

EXCLUDE_TIME_SENSITIVE = True if DATASET == 'mkqa' else False
DATASPLITS = DATASPLITS_MKQA if DATASET == 'mkqa' else DATASPLITS_GMMLU
NUM_LAYERS = 32 if LLM == 'llama_3.1_8B' else 36
NUM_HIDDEN_STATE = 4096

# MKQA - LLAMA 3.1 8 B
# QUERY
# fr - softmax
# QUERY = True
# LR = 0.0008
# EPOCHS = 139 
# L2_V = 0.286102950514828

# ANSWER
# fr - softmax
QUERY = False
LR = 0.0005
EPOCHS = 214
L2_V = 0.27276092363216636

# ---- MKQA - QWEN 3 8B
# QUERY
# fr - softmax
# QUERY = True
# LR = 0.0009
# EPOCHS = 98
# L2_V = 0.29527869618547636

# ANSWER
# fr - softmax
# QUERY = False
# LR = 0.0009
# EPOCHS = 111
# L2_V = 0.22252863623823427

# ---- GLOBAL MMLU - LLAMA 3.1 8B
# QUERY
# fr - softmax
# QUERY = True
# LR = 0.0011
# EPOCHS = 94 
# L2_V = 0.28258845606106203

# ANSWER
# fr - softmax
# QUERY = False
# LR = 0.0020
# EPOCHS = 60
# L2_V = 0.2557367492257226

# ---- GLOBAL MMLU - QWEN 3 8B
# QUERY
# fr - softmax
# QUERY = True
# LR = 0.001
# EPOCHS = 103
# L2_V = 0.2686538578969079

# ANSWER
# fr - softmax
# QUERY = False
# LR = 0.001
# EPOCHS = 165
# L2_V = 0.2115332075762067

TRAIN_BATCH_SIZE = 32
STRATIFIED = True
RESULTS_ROOT = Path(LOCAL_RESULTS_DIR) / DATASET / LLM


# ---- path helper (preserves your filenames) ----
def make_output_csv(isotonic: bool) -> Path:
    train_tag = '_'.join(TRAIN_LANGS)
    if EXCLUDE_TIME_SENSITIVE:
        base_dir = (RESULTS_ROOT / "probe_query_without_time_sensitive") if QUERY else (RESULTS_ROOT / "probe_answer_without_time_sensitive")
    else:
        base_dir = (RESULTS_ROOT / "probe_query") if QUERY else (RESULTS_ROOT / "probe_answer")
    subdir = f"{PROBE}_layer_scalers/train_{train_tag}"
    suffix = "multiseed_isotonic_eval_pure_es.csv" if isotonic else "multiseed_eval_pure_es.csv"
    fname = f"lr_{LR}_l1_{L1_W}_l2_{L2_V}_epochs_{EPOCHS}_{suffix}"
    return base_dir / subdir / fname


# ---- utility functions ----
def get_probe_class(name: str):
    name = name.lower()
    if name in ("sparsemaxlayerprobe", "sparsemax"):
        return SparsemaxLayerProbe
    elif name in ("softmaxlayerprobe", "softmax"):
        return SoftmaxLayerProbe
    else:
        raise ValueError(f"Unknown probe class '{name}'.")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_layer_importance(model: torch.nn.Module) -> np.ndarray:
    """Return numpy array of layer importance (softmax or sparsemax over model.w)."""
    with torch.no_grad():
        weights = model.w.detach()
        if PROBE.lower().startswith("softmax"):
            layer_importance = F.softmax(weights, dim=-1).cpu().numpy()
        else:
            layer_importance = Sparsemax(dim=-1)(weights).cpu().numpy()
    return layer_importance  # shape: (num_layers,)


def evaluate_for_seed(seed: int, model: torch.nn.Module, device, dataset: str, datasplits: tuple, results_root, train_langs: list, test_langs: list, 
                      query: bool, epochs: str, stratified: bool,  exclude_time_sensitive: bool, iso_reg=None):
    
    set_seed(seed)
    results = {}

    def calibrate(p):
        p = p.reshape(-1)
        if iso_reg is not None:
            return iso_reg.transform(p)
        return p

    # Few-shot evaluation
    logging.info(f"[seed={seed}] Few-shot")
    X_test_few, y_test_few = load_test_unflattened_from_hdf5(
        dataset, datasplits, results_root, train_langs, train_langs[0], query, epochs, 
        stratified, exclude_time_sensitive
    )
    X_test_few = torch.from_numpy(X_test_few).float().to(device)
    y_test_few = np.array(y_test_few)
    with torch.no_grad():
        test_logits_few = model(X_test_few)
        y_pred_prob_few = torch.sigmoid(test_logits_few).cpu().numpy()
    y_pred_prob_few = calibrate(y_pred_prob_few)  # isotonic calibration if a calibrator has been given

    auroc_few = roc_auc_score(y_test_few, y_pred_prob_few) if len(np.unique(y_test_few)) > 1 else float("nan")
    brier_few = brier_score_loss(y_test_few, y_pred_prob_few)
    ece_few = compute_ece(y_test_few, y_pred_prob_few, n_bins=10)
    precision_few, recall_few, _ = precision_recall_curve(y_test_few, y_pred_prob_few)
    aupr_few = auc(recall_few, precision_few)

    n_pos_few = int(y_test_few.sum())
    n_neg_few = len(y_test_few) - n_pos_few
    ratio = n_pos_few / (n_pos_few + n_neg_few) if (n_pos_few + n_neg_few) > 0 else 0.0
    logging.info(f"[seed={seed}] Few-shot class balance - 0s: {n_neg_few}, 1s: {n_pos_few}, ratio: {ratio:.4f}")
    results['few-shot'] = {
        "l1_w": L1_W, "l2_v": L2_V, "auroc": auroc_few, "brier": brier_few, "ece": ece_few, "aupr": aupr_few
    }

    # Zero-shot evaluation across test languages
    for test_lang in test_langs:
        logging.info(f"[seed={seed}] Evaluating for {test_lang} (few-shot ratio={ratio:.4f})")
        X_test, y_test = load_test_unflattened_from_hdf5(
            dataset, datasplits, results_root, train_langs, test_lang, query, epochs, 
            stratified, exclude_time_sensitive
        )
        X_test = np.asarray(X_test)
        y_test = np.asarray(y_test)

        # maximum M so that pos_count >= ratio * M → M <= pos_count / ratio
        pos_mask = (y_test == 1)
        neg_mask = (y_test == 0)
        X_pos, y_pos = X_test[pos_mask], y_test[pos_mask]
        X_neg, y_neg = X_test[neg_mask], y_test[neg_mask]

        if ratio == 0 or ratio == 1:
            logging.warning(f"[seed={seed}] Skipping {test_lang} because few-shot ratio is {ratio:.2f}.")
            continue
        
        # for eavluation with resampling
        # ensure target_pos ≤ len(y_pos)  →  M ≤ len(y_pos)/ratio
        # max_M_pos = int(len(y_pos) / ratio) if ratio > 0 else 0
        # ensure target_neg ≤ len(y_neg)  →  M − ratio*M ≤ len(y_neg) →  M*(1 − ratio) ≤ len(y_neg) →  M ≤ len(y_neg)/(1 − ratio)
        # max_M_neg = int(len(y_neg) / (1 - ratio)) if ratio < 1 else 0
        # final pool‐size
        # M = min(len(y_test), max_M_pos, max_M_neg)
        # target_pos = int(ratio * M)
        # target_neg = M - target_pos
        # if target_pos > len(y_pos) or target_neg > len(y_neg):
        #     logging.warning(f"[seed={seed}] Cannot sample required positives/negatives for {test_lang}; skipping.")
        #     continue
        # X_pos_s, y_pos_s = resample(X_pos, y_pos, replace=False, n_samples=target_pos, random_state=seed)
        # X_neg_s, y_neg_s = resample(X_neg, y_neg, replace=False, n_samples=target_neg, random_state=seed)

        # concatenate and shuffle
        # X_bal = np.concatenate([X_pos_s, X_neg_s], axis=0)
        # y_bal = np.concatenate([y_pos_s, y_neg_s], axis=0)
        X_bal = np.concatenate([X_pos, X_neg], axis=0)
        y_bal = np.concatenate([y_pos, y_neg], axis=0)
        perm = np.random.RandomState(seed).permutation(len(y_bal))
        X_bal = X_bal[perm]
        y_bal = y_bal[perm]
        X_test_tensor = torch.from_numpy(X_bal).float().to(device)
        y_test_arr = np.array(y_bal)

        # sanity checks
        num_ones = int((y_test_arr == 1).sum())
        num_zeros = int((y_test_arr == 0).sum())
        logging.info(f"[seed={seed}] Class balance ({test_lang}) - 0s: {num_zeros}, 1s: {num_ones}")

        # evaluation
        with torch.no_grad():
            test_logits = model(X_test_tensor).view(-1)
            y_pred_prob = torch.sigmoid(test_logits).cpu().numpy().reshape(-1)
        y_pred_prob = calibrate(y_pred_prob)  # isotonic calibration is a calibrator has been given

        auroc_test = roc_auc_score(y_test_arr, y_pred_prob) if len(np.unique(y_test_arr)) > 1 else float("nan")
        brier_test = brier_score_loss(y_test_arr, y_pred_prob)
        ece_test = compute_ece(y_test_arr, y_pred_prob, n_bins=10)
        precision_test, recall_test, _ = precision_recall_curve(y_test_arr, y_pred_prob)
        aupr_test = auc(recall_test, precision_test)

        results[test_lang] = {
            "l1_w": L1_W, "l2_v": L2_V, "auroc": auroc_test, "brier": brier_test, "ece": ece_test, "aupr": aupr_test
        }

        del X_pos, X_neg, X_bal
        # del X_pos_s, X_neg_s
        torch.cuda.empty_cache()

    # assemble DataFrame for this seed
    rows = []
    for lang, metrics in results.items():
        test_lang_field = lang if lang != "few-shot" else train_langs[0]
        row = {
            "seed": seed,
            "lang": lang,
            "l1_w": metrics["l1_w"],
            "l2_v": metrics["l2_v"],
            "auroc": metrics["auroc"],
            "brier": metrics["brier"],
            "ece": metrics["ece"],
            "aupr": metrics["aupr"],
            "probe": PROBE,
            "dataset": DATASET,
            "llm": LLM,
            "train_langs": "-".join(train_langs),
            "test_lang": test_lang_field,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        rows.append(row)
    eval_df = pd.DataFrame(rows)

    # layer importance for this seed (shared across languages)
    layer_imp = compute_layer_importance(model)  # shape: (num_layers,)
    layer_rows = []
    for layer_idx, importance in enumerate(layer_imp):
        layer_rows.append(
            {
                "seed": seed,
                "probe": PROBE,
                "dataset": DATASET,
                "llm": LLM,
                "train_langs": "-".join(train_langs),
                "l1_w": L1_W,
                "l2_v": L2_V,
                "layer": layer_idx,
                "importance": float(importance),
            }
        )
    layer_df = pd.DataFrame(layer_rows)

    return eval_df, layer_df


def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation over multiple seeds and append results to CSV.")
    parser.add_argument("--seeds", type=int, nargs="+", required=True, help="Random seeds to run.")
    parser.add_argument("--both", action="store_true",
                        help="Run both non-isotonic and isotonic evaluations, writing to their respective files.")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Choose which isotonic modes to run
    modes = [False, True] if args.both else [ISOTONIC]

    for isotonic in modes:
        output_path = make_output_csv(isotonic)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        eval_dfs = []
        layer_imp_dfs = []

        # Evaluation for each seed
        for seed in args.seeds:

            # Build probe directory / model path
            if QUERY:
                if EXCLUDE_TIME_SENSITIVE:
                    probe_dir = RESULTS_ROOT / f"probe_query_without_time_sensitive/{PROBE}_layer_scalers/train_{'_'.join(TRAIN_LANGS)}/batch_{TRAIN_BATCH_SIZE}_lr_{LR}_l1_{L1_W}_l2_{L2_V}_epochs_{EPOCHS}_seed_{seed}"
                else:
                    probe_dir = RESULTS_ROOT / f"probe_query/{PROBE}_layer_scalers/train_{'_'.join(TRAIN_LANGS)}/batch_{TRAIN_BATCH_SIZE}_lr_{LR}_l1_{L1_W}_l2_{L2_V}_epochs_{EPOCHS}_seed_{seed}"
            else:
                if EXCLUDE_TIME_SENSITIVE:
                    probe_dir = RESULTS_ROOT / f"probe_answer_without_time_sensitive/{PROBE}_layer_scalers/train_{'_'.join(TRAIN_LANGS)}/batch_{TRAIN_BATCH_SIZE}_lr_{LR}_l1_{L1_W}_l2_{L2_V}_epochs_{EPOCHS}_seed_{seed}"
                else:
                    probe_dir = RESULTS_ROOT / f"probe_answer/{PROBE}_layer_scalers/train_{'_'.join(TRAIN_LANGS)}/batch_{TRAIN_BATCH_SIZE}_lr_{LR}_l1_{L1_W}_l2_{L2_V}_epochs_{EPOCHS}_seed_{seed}"
            model_path = probe_dir / "probe_state.pt"
            if not model_path.exists():
                logging.error(f"Model file not found at {model_path.resolve()}")
                sys.exit(1)

            # Load the probe in the evaluation mode
            ProbeClass = get_probe_class(PROBE)
            model = ProbeClass(num_layers=NUM_LAYERS, hidden_dim=NUM_HIDDEN_STATE)
            state = torch.load(model_path, map_location=device)
            model.load_state_dict(state)
            model.to(device)
            model.eval()

            # Fit an isotonic regressor on the validation split used during training
            iso_reg = None
            if isotonic:
                X_tr, y_tr, X_va, y_va = load_and_preprocess_unflattened_from_hdf5(
                    DATASET, DATASPLITS, RESULTS_ROOT, TRAIN_LANGS, 
                    query=QUERY, epochs=str(EPOCHS), stratified=STRATIFIED, exclude_time_sensitive=EXCLUDE_TIME_SENSITIVE
                )
                X_va_t = torch.from_numpy(X_va).float().to(device)
                y_va_np = np.asarray(y_va)
                with torch.no_grad():
                    logits_va = model(X_va_t).view(-1)
                    prob_va = torch.sigmoid(logits_va).cpu().numpy().reshape(-1)
                # need both classes and some variation in probs
                if len(np.unique(y_va_np)) >= 2 and np.unique(prob_va).size >= 3:
                    iso_reg = IsotonicRegression(out_of_bounds="clip")
                    iso_reg.fit(prob_va, y_va_np)
                    l = TRAIN_LANGS[0]
                    if QUERY:
                        cal_path = probe_dir / f"isotonic_calibrator_query_{l}.joblib"
                    else:
                        cal_path = probe_dir / f"isotonic_calibrator_answer_{l}.joblib"
                    joblib.dump(iso_reg, cal_path)
                else:
                    logging.warning("Skipping isotonic: val set has single class or too few distinct probabilities.")

            try:
                df_eval_seed, df_layer_seed = evaluate_for_seed(
                    seed, model, device,
                    dataset=DATASET,
                    datasplits=DATASPLITS, results_root=RESULTS_ROOT,
                    train_langs=TRAIN_LANGS, test_langs=TEST_LANGS,
                    query=QUERY, epochs=str(EPOCHS),
                    stratified=STRATIFIED, exclude_time_sensitive=EXCLUDE_TIME_SENSITIVE,
                    iso_reg=iso_reg)
                eval_dfs.append(df_eval_seed)
                layer_imp_dfs.append(df_layer_seed)
            except Exception as e:
                logging.exception(f"[seed={seed}] Evaluation failed: {e}")
                sys.exit()
            finally:
                torch.cuda.empty_cache()

        if not eval_dfs:
            logging.error("No successful evaluations; nothing written.")
            continue

        combined = pd.concat(eval_dfs, ignore_index=True)
        layer_combined = pd.concat(layer_imp_dfs, ignore_index=True)

        # append per-seed evaluation results (rounded)
        to_write = combined.copy()
        for col in ["auroc", "brier", "ece", "aupr"]:
            to_write[col] = to_write[col].round(2)
        write_header = not output_path.exists()
        to_write.to_csv(output_path, mode="a", index=False, header=write_header)
        logging.info(f"[{'ISOTONIC' if isotonic else 'NO-ISOTONIC'}] Wrote {len(to_write)} rows to {output_path.resolve()}")

        # summary roll-up across seeds
        summary = (
            combined.groupby(["probe", "dataset", "llm", "train_langs", "test_lang", "lang", "l1_w", "l2_v"], dropna=False, observed=False)
            .agg(count_seeds=("seed", "nunique"),
                 seeds=("seed", lambda x: ",".join(str(s) for s in sorted(set(x)))),
                 auroc_mean=("auroc", "mean"), auroc_std=("auroc", "std"),
                 brier_mean=("brier", "mean"), brier_std=("brier", "std"),
                 ece_mean=("ece", "mean"), ece_std=("ece", "std"),
                 aupr_mean=("aupr", "mean"), aupr_std=("aupr", "std"))
            .reset_index()
        )

        for col in ["auroc_mean", "auroc_std", "brier_mean", "brier_std", "ece_mean", "ece_std", "aupr_mean", "aupr_std"]:
            summary[col] = summary[col].round(2)

        # human-readable mean±std
        summary["auroc"] = summary.apply(
            lambda r: f"{r['auroc_mean']}±{r['auroc_std']}" if not pd.isna(r["auroc_std"]) else f"{r['auroc_mean']}", axis=1
        )
        summary["brier"] = summary.apply(
            lambda r: f"{r['brier_mean']}±{r['brier_std']}" if not pd.isna(r["brier_std"]) else f"{r['brier_mean']}", axis=1
        )
        summary["ece"] = summary.apply(
            lambda r: f"{r['ece_mean']}±{r['ece_std']}" if not pd.isna(r["ece_std"]) else f"{r['ece_mean']}", axis=1
        )
        summary["aupr"] = summary.apply(
            lambda r: f"{r['aupr_mean']}±{r['aupr_std']}" if not pd.isna(r["aupr_std"]) else f"{r['aupr_mean']}", axis=1
        )

        summary_path = output_path.with_name(output_path.stem + "_summary_pure_es.csv")
        summary.to_csv(summary_path, index=False)
        logging.info(f"[{'ISOTONIC' if isotonic else 'NO-ISOTONIC'}] Wrote summary roll-up to {summary_path.resolve()}")

        # layer importance per seed
        layer_imp_path = output_path.with_name(output_path.stem + "_layer_importance_pure_es.csv")
        write_header_layer = not layer_imp_path.exists()
        layer_combined.to_csv(layer_imp_path, mode="a", index=False, header=write_header_layer)
        logging.info(f"[{'ISOTONIC' if isotonic else 'NO-ISOTONIC'}] Wrote per-seed layer importance to {layer_imp_path.resolve()}")

        # layer importance summary across seeds
        layer_summary = (
            layer_combined.groupby(["probe", "dataset", "llm", "train_langs", "l1_w", "l2_v", "layer"], dropna=False, observed=False)
            .agg(
                count_seeds=("seed", "nunique"),
                seeds=("seed", lambda x: ",".join(str(s) for s in sorted(set(x)))),
                importance_mean=("importance", "mean"),
                importance_std=("importance", "std"),
            )
            .reset_index()
        )

        for col in ["importance_mean", "importance_std"]:
            layer_summary[col] = layer_summary[col].round(2)

        layer_summary["importance"] = layer_summary.apply(
            lambda r: f"{r['importance_mean']}±{r['importance_std']}"
            if not pd.isna(r["importance_std"])
            else f"{r['importance_mean']}",
            axis=1,
        )
        layer_summary_path = output_path.with_name(output_path.stem + "_layer_importance_summary__es.csv")
        layer_summary.to_csv(layer_summary_path, index=False)
        logging.info(f"[{'ISOTONIC' if isotonic else 'NO-ISOTONIC'}] Wrote layer importance summary to {layer_summary_path.resolve()}")


if __name__ == "__main__":
    main()
