#!/usr/bin/env python
import argparse
import torch
from pathlib import Path
from utils import (
    load_test_unflattened_from_hdf5,
    compute_ece,
    load_and_preprocess_unflattened_from_hdf5,
)
from sklearn.metrics import roc_auc_score, brier_score_loss, precision_recall_curve, auc
import numpy as np
import pandas as pd
from sklearn.utils import resample
import random
import logging
import sys
from datetime import datetime, timezone
import torch.nn.functional as F
from sparsemax import Sparsemax
import joblib
from probes import SparsemaxLayerProbe, SoftmaxLayerProbe
from sklearn.isotonic import IsotonicRegression

from constants import LOCAL_RESULTS_DIR, DATASPLITS

# ---- hardcoded configuration ----
TIMESENSITIVE = True
L1_W = 0

TRAIN_LANGS = ['fr']
TEST_LANGS = ['en', 'es', 'pl', 'ru', 'ja']
ISOTONIC = False  # default mode if --both is not used

DATASET = "mkqa"
LLM = "llama_3.1_8B"
# LLM = "qwen3_8B"

if LLM == "llama_3.1_8B":
    NUM_LAYERS = 32
elif LLM == "qwen3_8B":
    NUM_LAYERS = 36
NUM_HIDDEN_STATE = 4096

# llama 3.1 8B - with time sensitive
# QUERY
# fr - softmax
# QUERY = True
# PROBE = "SoftmaxLayerProbe"
# LR = 0.0008678569352340735
# EPOCHS = 83 
# L2_V = 0.2639490824127109
# ANSWER
# fr - softmax
# QUERY = False
# PROBE = "SoftmaxLayerProbe"
# LR = 0.0008148822621480094
# EPOCHS = 126
# L2_V = 0.26474818664703487

# llama 3.1 8B - without time sensitive
# QUERY
# QUERY = True
# PROBE = "SoftmaxLayerProbe"
# LR = 0.0029103296177981977
# EPOCHS = 27
# L2_V = 0.2994972929159425

# ANSWER
QUERY = False
PROBE = "SoftmaxLayerProbe"
LR = 0.005814750959418485
EPOCHS = 30
L2_V = 0.16106232877187474

# qwen 3 8B
# QUERY
# fr - softmax
# QUERY = True
# PROBE = "SoftmaxLayerProbe"
# LR = 0.0006956555443343052
# EPOCHS = 112
# L2_V = 0.21661544769753235

# ANSWER
# fr - softmax
# QUERY = False
# PROBE = "SoftmaxLayerProbe"
# LR = 0.0008743853100864169
# EPOCHS = 321
# L2_V = 0.1754312779467239

TRAIN_BATCH_SIZE = 32
STRATIFIED = True
LAST = True

RESULTS_ROOT = Path(LOCAL_RESULTS_DIR) / DATASET / LLM

# ---- path helper (preserves your filenames) ----
def make_output_csv(isotonic: bool) -> Path:
    train_tag = '_'.join(TRAIN_LANGS)
    if TIMESENSITIVE:
        base_dir = (RESULTS_ROOT / "paper_probes_query_nottimesensitive") if QUERY else (RESULTS_ROOT / "paper_probes_answer_nottimesensitive")
    else:
        base_dir = (RESULTS_ROOT / "paper_probes_query") if QUERY else (RESULTS_ROOT / "paper_probes_answer")
    subdir = f"{PROBE}_layer_scalers/train_{train_tag}"
    suffix = "datasplits_multiseed_isotonic_eval.csv" if isotonic else "datasplits_multiseed_eval.csv"
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


def evaluate_for_seed(seed: int, model: torch.nn.Module, device, iso_reg=None):
    """
    Evaluates per datasplit.

    Outputs eval_df with rows for:
      - each datasplit + "few-shot" (train lang)
      - each datasplit + each test language
    """
    set_seed(seed)

    def calibrate(p):
        p = p.reshape(-1)
        if iso_reg is not None:
            return iso_reg.transform(p)
        return p

    # Collect per-(datasplit, lang) metrics
    results = {}  # key: (datasplit, lang_label)

    for ds in DATASPLITS:
        logging.info(f"[seed={seed}] ===== datasplit={ds} =====")

        # ---- few-shot evaluation (for this split) ----
        logging.info(f"[seed={seed}] Few-shot (datasplit={ds})")
        X_test_few, y_test_few = load_test_unflattened_from_hdf5(
            [ds],
            RESULTS_ROOT,
            TRAIN_LANGS,
            TRAIN_LANGS[0],  # evaluate train language as "few-shot"
            query=QUERY,
            stratified=STRATIFIED,
            last=LAST, exclude_time_sensitive=TIMESENSITIVE
        )
        X_test_few_t = torch.from_numpy(np.asarray(X_test_few)).float().to(device)
        y_test_few = np.asarray(y_test_few)

        with torch.no_grad():
            test_logits_few = model(X_test_few_t)
            y_pred_prob_few = torch.sigmoid(test_logits_few).cpu().numpy()
        y_pred_prob_few = calibrate(y_pred_prob_few)

        auroc_few = roc_auc_score(y_test_few, y_pred_prob_few) if len(np.unique(y_test_few)) > 1 else float("nan")
        brier_few = brier_score_loss(y_test_few, y_pred_prob_few)
        ece_few = compute_ece(y_test_few, y_pred_prob_few, n_bins=10)
        precision_few, recall_few, _ = precision_recall_curve(y_test_few, y_pred_prob_few)
        aupr_few = auc(recall_few, precision_few)

        n_pos_few = int(y_test_few.sum())
        n_neg_few = len(y_test_few) - n_pos_few
        ratio = n_pos_few / (n_pos_few + n_neg_few) if (n_pos_few + n_neg_few) > 0 else 0.0
        logging.info(
            f"[seed={seed}] Few-shot balance (datasplit={ds}) - 0s:{n_neg_few}, 1s:{n_pos_few}, ratio:{ratio:.4f}"
        )

        results[(ds, "few-shot")] = {
            "l1_w": L1_W,
            "l2_v": L2_V,
            "auroc": auroc_few,
            "brier": brier_few,
            "ece": ece_few,
            "aupr": aupr_few,
            "ratio": ratio,
        }

        # ---- test languages (for this split) ----
        for test_lang in TEST_LANGS:
            logging.info(f"[seed={seed}] Evaluating {test_lang} (datasplit={ds}, few-shot ratio={ratio:.4f})")

            X_test, y_test = load_test_unflattened_from_hdf5(
                [ds],
                RESULTS_ROOT,
                TRAIN_LANGS,
                test_lang,
                query=QUERY,
                stratified=STRATIFIED,
                last=LAST, exclude_time_sensitive=TIMESENSITIVE
            )
            X_test = np.asarray(X_test)
            y_test = np.asarray(y_test)

            if ratio == 0 or ratio == 1:
                logging.warning(
                    f"[seed={seed}] Skipping {test_lang} (datasplit={ds}) because few-shot ratio is {ratio:.2f}."
                )
                continue

            pos_mask = (y_test == 1)
            neg_mask = (y_test == 0)
            X_pos, y_pos = X_test[pos_mask], y_test[pos_mask]
            X_neg, y_neg = X_test[neg_mask], y_test[neg_mask]

            # maximum M so that pos_count >= ratio * M → M <= pos_count / ratio
            max_M_pos = int(len(y_pos) / ratio) if ratio > 0 else 0
            # and M <= len(y_neg)/(1-ratio)
            max_M_neg = int(len(y_neg) / (1 - ratio)) if ratio < 1 else 0
            M = min(len(y_test), max_M_pos, max_M_neg)

            target_pos = int(ratio * M)
            target_neg = M - target_pos

            if target_pos > len(y_pos) or target_neg > len(y_neg):
                logging.warning(
                    f"[seed={seed}] Cannot sample required positives/negatives for {test_lang} (datasplit={ds}); skipping."
                )
                continue

            X_pos_s, y_pos_s = resample(X_pos, y_pos, replace=False, n_samples=target_pos, random_state=seed)
            X_neg_s, y_neg_s = resample(X_neg, y_neg, replace=False, n_samples=target_neg, random_state=seed)

            # concatenate and shuffle
            X_bal = np.concatenate([X_pos_s, X_neg_s], axis=0)
            y_bal = np.concatenate([y_pos_s, y_neg_s], axis=0)
            perm = np.random.RandomState(seed).permutation(len(y_bal))
            X_bal = X_bal[perm]
            y_bal = y_bal[perm]

            X_test_tensor = torch.from_numpy(X_bal).float().to(device)
            y_test_arr = np.asarray(y_bal)

            # sanity checks
            num_ones = int((y_test_arr == 1).sum())
            num_zeros = int((y_test_arr == 0).sum())
            logging.info(f"[seed={seed}] Class balance ({test_lang}, datasplit={ds}) - 0s:{num_zeros}, 1s:{num_ones}")

            with torch.no_grad():
                test_logits = model(X_test_tensor).view(-1)
                y_pred_prob = torch.sigmoid(test_logits).cpu().numpy()
            y_pred_prob = calibrate(y_pred_prob)

            auroc_test = roc_auc_score(y_test_arr, y_pred_prob) if len(np.unique(y_test_arr)) > 1 else float("nan")
            brier_test = brier_score_loss(y_test_arr, y_pred_prob)
            ece_test = compute_ece(y_test_arr, y_pred_prob, n_bins=10)
            precision_test, recall_test, _ = precision_recall_curve(y_test_arr, y_pred_prob)
            aupr_test = auc(recall_test, precision_test)

            results[(ds, test_lang)] = {
                "l1_w": L1_W,
                "l2_v": L2_V,
                "auroc": auroc_test,
                "brier": brier_test,
                "ece": ece_test,
                "aupr": aupr_test,
                "ratio": ratio,
            }

            # cleanup
            del X_pos, X_neg, X_pos_s, X_neg_s, X_bal
            torch.cuda.empty_cache()

    # ---- assemble DataFrame for this seed ----
    rows = []
    for (ds, lang_label), metrics in results.items():
        # Keep your original semantics:
        # - lang column is "few-shot" or actual test language
        # - test_lang column is train lang for few-shot, otherwise the actual test language
        test_lang_field = TRAIN_LANGS[0] if lang_label == "few-shot" else lang_label

        rows.append(
            {
                "seed": seed,
                "datasplit": ds,
                "lang": lang_label,
                "l1_w": metrics["l1_w"],
                "l2_v": metrics["l2_v"],
                "auroc": metrics["auroc"],
                "brier": metrics["brier"],
                "ece": metrics["ece"],
                "aupr": metrics["aupr"],
                "probe": PROBE,
                "dataset": DATASET,
                "llm": LLM,
                "train_langs": "-".join(TRAIN_LANGS),
                "test_lang": test_lang_field,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
    eval_df = pd.DataFrame(rows)

    # ---- layer importance for this seed (shared across languages/splits) ----
    layer_imp = compute_layer_importance(model)  # shape: (num_layers,)
    layer_rows = []
    for layer_idx, importance in enumerate(layer_imp):
        layer_rows.append(
            {
                "seed": seed,
                "probe": PROBE,
                "dataset": DATASET,
                "llm": LLM,
                "train_langs": "-".join(TRAIN_LANGS),
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
    parser.add_argument(
        "--both",
        action="store_true",
        help="Run both non-isotonic and isotonic evaluations, writing to their respective files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # choose which isotonic modes to run
    modes = [False, True] if args.both else [ISOTONIC]

    for isotonic in modes:
        output_path = make_output_csv(isotonic)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        eval_dfs = []
        layer_imp_dfs = []

        # evaluation for each seed
        for seed in args.seeds:
            # build probe directory / model path
            if QUERY:
                if TIMESENSITIVE:
                    probe_dir = (
                        RESULTS_ROOT
                        / f"paper_probes_query_nottimesensitive/{PROBE}_layer_scalers/train_{'_'.join(TRAIN_LANGS)}/batch_{TRAIN_BATCH_SIZE}_lr_{LR}_l1_{L1_W}_l2_{L2_V}_epochs_{EPOCHS}_seed_{seed}"
                    )
                else:
                    probe_dir = (
                        RESULTS_ROOT
                        / f"paper_probes_query/{PROBE}_layer_scalers/train_{'_'.join(TRAIN_LANGS)}/batch_{TRAIN_BATCH_SIZE}_lr_{LR}_l1_{L1_W}_l2_{L2_V}_epochs_{EPOCHS}_seed_{seed}"
                    )
            else:
                if TIMESENSITIVE:
                    probe_dir = (
                        RESULTS_ROOT
                        / f"paper_probes_answer_nottimesensitive/{PROBE}_layer_scalers/train_{'_'.join(TRAIN_LANGS)}/batch_{TRAIN_BATCH_SIZE}_lr_{LR}_l1_{L1_W}_l2_{L2_V}_epochs_{EPOCHS}_seed_{seed}"
                    )
                else:
                    probe_dir = (
                        RESULTS_ROOT
                        / f"paper_probes_answer/{PROBE}_layer_scalers/train_{'_'.join(TRAIN_LANGS)}/batch_{TRAIN_BATCH_SIZE}_lr_{LR}_l1_{L1_W}_l2_{L2_V}_epochs_{EPOCHS}_seed_{seed}"
                    )
            model_path = probe_dir / "probe_state.pt"
            if not model_path.exists():
                logging.error(f"Model file not found at {model_path.resolve()}")
                sys.exit(1)

            # load probe
            ProbeClass = get_probe_class(PROBE)
            model = ProbeClass(num_layers=NUM_LAYERS, hidden_dim=NUM_HIDDEN_STATE)
            state = torch.load(model_path, map_location=device)
            model.load_state_dict(state)
            model.to(device)
            model.eval()

            # fit an isotonic regressor on the pooled validation split used during training
            iso_reg = None
            if isotonic:
                X_tr, y_tr, X_va, y_va = load_and_preprocess_unflattened_from_hdf5(
                    DATASPLITS,
                    RESULTS_ROOT,
                    TRAIN_LANGS,
                    query=QUERY,
                    stratified=STRATIFIED,
                    last=LAST, exclude_time_sensitive=TIMESENSITIVE
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
                df_eval_seed, df_layer_seed = evaluate_for_seed(seed, model, device, iso_reg=iso_reg)
                eval_dfs.append(df_eval_seed)
                layer_imp_dfs.append(df_layer_seed)
            except Exception as e:
                logging.exception(f"[seed={seed}] Evaluation failed: {e}")
                sys.exit(1)
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
        logging.info(
            f"[{'ISOTONIC' if isotonic else 'NO-ISOTONIC'}] Wrote {len(to_write)} rows to {output_path.resolve()}"
        )

        # summary roll-up across seeds (NOW includes datasplit)
        summary = (
            combined.groupby(
                ["probe", "dataset", "llm", "train_langs", "datasplit", "test_lang", "lang", "l1_w", "l2_v"],
                dropna=False,
                observed=False,
            )
            .agg(
                count_seeds=("seed", "nunique"),
                seeds=("seed", lambda x: ",".join(str(s) for s in sorted(set(x)))),
                auroc_mean=("auroc", "mean"),
                auroc_std=("auroc", "std"),
                brier_mean=("brier", "mean"),
                brier_std=("brier", "std"),
                ece_mean=("ece", "mean"),
                ece_std=("ece", "std"),
                aupr_mean=("aupr", "mean"),
                aupr_std=("aupr", "std"),
            )
            .reset_index()
        )

        for col in [
            "auroc_mean",
            "auroc_std",
            "brier_mean",
            "brier_std",
            "ece_mean",
            "ece_std",
            "aupr_mean",
            "aupr_std",
        ]:
            summary[col] = summary[col].round(2)

        # human-readable mean±std
        summary["auroc"] = summary.apply(
            lambda r: f"{r['auroc_mean']}±{r['auroc_std']}" if not pd.isna(r["auroc_std"]) else f"{r['auroc_mean']}",
            axis=1,
        )
        summary["brier"] = summary.apply(
            lambda r: f"{r['brier_mean']}±{r['brier_std']}" if not pd.isna(r["brier_std"]) else f"{r['brier_mean']}",
            axis=1,
        )
        summary["ece"] = summary.apply(
            lambda r: f"{r['ece_mean']}±{r['ece_std']}" if not pd.isna(r["ece_std"]) else f"{r['ece_mean']}",
            axis=1,
        )
        summary["aupr"] = summary.apply(
            lambda r: f"{r['aupr_mean']}±{r['aupr_std']}" if not pd.isna(r["aupr_std"]) else f"{r['aupr_mean']}",
            axis=1,
        )

        summary_path = output_path.with_name(output_path.stem + "datasplits_summary.csv")
        summary.to_csv(summary_path, index=False)
        logging.info(
            f"[{'ISOTONIC' if isotonic else 'NO-ISOTONIC'}] Wrote summary roll-up to {summary_path.resolve()}"
        )

        # layer importance per seed (unchanged)
        layer_imp_path = output_path.with_name(output_path.stem + "datasplits_layer_importance.csv")
        write_header_layer = not layer_imp_path.exists()
        layer_combined.to_csv(layer_imp_path, mode="a", index=False, header=write_header_layer)
        logging.info(
            f"[{'ISOTONIC' if isotonic else 'NO-ISOTONIC'}] Wrote per-seed layer importance to {layer_imp_path.resolve()}"
        )

        # layer importance summary across seeds (unchanged)
        layer_summary = (
            layer_combined.groupby(
                ["probe", "dataset", "llm", "train_langs", "l1_w", "l2_v", "layer"], dropna=False, observed=False
            )
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
        layer_summary_path = output_path.with_name(output_path.stem + "_datasplits_layer_importance_summary.csv")
        layer_summary.to_csv(layer_summary_path, index=False)
        logging.info(
            f"[{'ISOTONIC' if isotonic else 'NO-ISOTONIC'}] Wrote layer importance summary to {layer_summary_path.resolve()}"
        )


if __name__ == "__main__":
    main()
