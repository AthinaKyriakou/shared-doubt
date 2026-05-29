#!/usr/bin/env python
"""
Sliding-window readout ablation.

Usage
-----
python weight_ablation.py --seeds 42 17 3 59 884 445 369 169 76 905 --k 3
"""

import argparse
import logging
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, brier_score_loss, precision_recall_curve, auc
from sparsemax import Sparsemax

from probes import SoftmaxLayerProbe, SparsemaxLayerProbe
from utils import load_test_unflattened_from_hdf5, compute_ece
from constants import LOCAL_RESULTS_DIR, DATASPLITS_MKQA, DATASPLITS_GMMLU


# ═══════════════════════════ CONFIG ═══════════════════════════
PROBE = "SoftmaxLayerProbe"
L1_W = 0
ISOTONIC = False

TRAIN_LANGS = ["fr"]
TEST_LANGS  = ["en", "es", "pl", "ru", "ja"]

EXCLUDE_TIME_SENSITIVE = False          # True → MKQA, False → Global MMLU
# DATASET = "mkqa"
DATASET = "global_mmlu"
# LLM   = "llama_3.1_8B"
LLM     = "qwen3_8B"

DATASPLITS   = DATASPLITS_MKQA if DATASET == "mkqa" else DATASPLITS_GMMLU
NUM_LAYERS   = 32 if LLM == "llama_3.1_8B" else 36
HIDDEN_DIM   = 4096
TRAIN_BATCH  = 32
STRATIFIED   = True
K_WINDOW     = 3

RESULTS_ROOT = Path(LOCAL_RESULTS_DIR) / DATASET / LLM

# ---- hyperparams that identify the trained probe ----
# MKQA - LLAMA 3.1 8 B
# QUERY = True
# LR = 0.0008
# EPOCHS = 139 
# L2_V = 0.286102950514828

# QUERY = False
# LR = 0.0005
# EPOCHS = 210
# L2_V = 0.27276092363216636

# ---- MKQA - QWEN 3 8B
# QUERY = True
# LR = 0.0009
# EPOCHS = 98
# L2_V = 0.29527869618547636

# QUERY = False
# LR = 0.0009
# EPOCHS = 111
# L2_V = 0.22252863623823427

# ---- GLOBAL MMLU - LLAMA 3.1 8B
# QUERY = True
# LR = 0.0011
# EPOCHS = 94 
# L2_V = 0.28258845606106203

# QUERY = False
# LR = 0.0020
# EPOCHS = 60
# L2_V = 0.2557367492257226

# ---- GLOBAL MMLU - QWEN 3 8B
# QUERY = True
# LR = 0.001
# EPOCHS = 103
# L2_V = 0.2686538578969079

QUERY = False
LR = 0.001
EPOCHS = 165
L2_V = 0.2115332075762067

# ═══════════════════════════ HELPERS ══════════════════════════

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_probe_class(name: str):
    n = name.lower()
    if "sparsemax" in n:
        return SparsemaxLayerProbe
    if "softmax" in n:
        return SoftmaxLayerProbe
    raise ValueError(f"Unknown probe class '{name}'.")


def probe_dir_for_seed(seed: int) -> Path:
    """Build the checkpoint directory that matches train_probe.py's naming."""
    tag = "_".join(TRAIN_LANGS)
    if QUERY:
        folder = ("probe_query_without_time_sensitive"
                  if EXCLUDE_TIME_SENSITIVE else "probe_query")
    else:
        folder = ("probe_answer_without_time_sensitive"
                  if EXCLUDE_TIME_SENSITIVE else "probe_answer")
    return (
        RESULTS_ROOT / folder
        / f"{PROBE}_layer_scalers"
        / f"train_{tag}"
        / f"batch_{TRAIN_BATCH}_lr_{LR}_l1_{L1_W}_l2_{L2_V}_epochs_{EPOCHS}_seed_{seed}"
    )


def output_dir() -> Path:
    """Directory for ablation CSVs (one level above the per-seed folders)."""
    # reuse probe_dir_for_seed with a dummy seed, then take the parent
    d = probe_dir_for_seed(seed=0).parent
    d.mkdir(parents=True, exist_ok=True)
    return d


# ═══════════════════════════ METRICS ══════════════════════════

def _metrics(y, p):
    """Compute AUROC, AUPR, Brier, ECE from labels and probabilities."""
    y = np.asarray(y).reshape(-1)
    p = np.asarray(p).reshape(-1)
    # keep only samples with valid binary labels
    valid = np.isin(y, [0, 1])
    y, p = y[valid], p[valid]
    out = {}
    out["auroc"] = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
    prec, rec, _ = precision_recall_curve(y, p)
    out["aupr"]  = auc(rec, prec)
    out["brier"] = brier_score_loss(y, p)
    out["ece"]   = float(compute_ece(y, p, n_bins=10))
    return out


def _delta(base, masked):
    """Signed so that positive = performance got worse when we ablated.
    For AUROC/AUPR (higher is better): base − masked.
    For Brier/ECE  (lower  is better): masked − base."""
    d = {}
    for k in base:
        if k in ("brier", "ece"):
            d[k] = masked[k] - base[k]
        else:
            d[k] = base[k] - masked[k]
    return d


def _make_windows(L, k):
    """Non-wrapping sliding windows of width k over L layers."""
    assert 1 <= k <= L
    return [list(range(i, i + k)) for i in range(L - k + 1)]


# ════════════════════ CORE ABLATION FUNCTION ══════════════════

@torch.no_grad()
def sliding_window_ablation(model, X, y, *, k=3, iso_reg=None):
    """
    Sliding-window readout ablation for a Softmax/SparsemaxLayerProbe.

    Masks the probe's layer-mixing logits (model.w) for each window so
    softmax assigns ~zero weight to those layers and re-normalises over
    the rest (Eq. 6.2).

    Returns
    -------
    per_layer_raw  : dict[str, ndarray]   – coverage-normalised per-layer deltas (RAW)
    per_layer_cal  : dict[str, ndarray] | None
    win_df_raw     : DataFrame            – one row per window (RAW)
    win_df_cal     : DataFrame | None
    base_raw       : dict                 – baseline metrics (RAW)
    base_cal       : dict | None
    """
    B, L, H = X.shape
    layer_logits = model.w
    orig_logits  = layer_logits.detach().clone()
    NEG_INF      = float("-inf")

    # ---- baseline (unmasked) ----
    p_raw_base = torch.sigmoid(model(X)).view(-1).cpu().numpy()
    base_raw   = _metrics(y, p_raw_base)

    base_cal, use_cal = None, False
    if iso_reg is not None:
        p_cal_base = iso_reg.transform(p_raw_base)
        base_cal   = _metrics(y, p_cal_base)
        use_cal    = True

    # ---- sliding windows ----
    wins = _make_windows(L, k)
    rows_raw, rows_cal = [], []

    try:
        for w in wins:
            # mask logits → −∞ for this window
            layer_logits.copy_(orig_logits)
            masked = orig_logits.clone()
            masked[w] = masked[w].new_full(masked[w].shape, NEG_INF)
            layer_logits.copy_(masked)

            p_raw = torch.sigmoid(model(X)).view(-1).cpu().numpy()
            m_raw = _metrics(y, p_raw)
            d_raw = _delta(base_raw, m_raw)
            rows_raw.append({
                "window": "-".join(map(str, w)),
                "layers": w,
                **{f"masked_{k_}": v for k_, v in m_raw.items()},
                **{f"delta_{k_}": v for k_, v in d_raw.items()},
            })

            if use_cal:
                p_cal = iso_reg.transform(p_raw)
                m_cal = _metrics(y, p_cal)
                d_cal = _delta(base_cal, m_cal)
                rows_cal.append({
                    "window": "-".join(map(str, w)),
                    "layers": w,
                    **{f"masked_{k_}": v for k_, v in m_cal.items()},
                    **{f"delta_{k_}": v for k_, v in d_cal.items()},
                })
    finally:
        # always restore the original logits
        layer_logits.copy_(orig_logits)

    win_df_raw = pd.DataFrame(rows_raw)
    win_df_cal = pd.DataFrame(rows_cal) if use_cal else None

    # ---- per-layer importance (coverage-normalised avg delta) ----
    def _per_layer(rows, metric_keys):
        imp  = {m: np.zeros(L) for m in metric_keys}
        hits = np.zeros(L, dtype=int)
        for row in rows:
            for idx in row["layers"]:
                hits[idx] += 1
                for m in metric_keys:
                    imp[m][idx] += row.get(f"delta_{m}", 0.0)
        denom = np.maximum(hits, 1)
        for m in imp:
            imp[m] /= denom
        return imp

    per_layer_raw = _per_layer(rows_raw, base_raw.keys())
    per_layer_cal = _per_layer(rows_cal, base_cal.keys()) if use_cal else None

    return per_layer_raw, per_layer_cal, win_df_raw, win_df_cal, base_raw, base_cal


# ═══════════════════════ SANITY CHECK ═════════════════════════

@torch.no_grad()
def sanity_check(model, X):
    """Verify that manually reconstructing z from per-layer scores matches
    the probe's forward pass (catches shape / weight-sharing bugs)."""
    B, L, H = X.shape

    if isinstance(model, SoftmaxLayerProbe):
        alpha = F.softmax(model.w, dim=-1)
    else:
        alpha = Sparsemax(dim=-1)(model.w)

    # per-layer scores: s_{b,l} = X_{b,l,:} · v
    s = torch.matmul(X.reshape(B * L, H), model.v).reshape(B, L)
    z_manual  = (s * alpha.unsqueeze(0)).sum(dim=1) + model.bias.squeeze()
    z_forward = model(X).squeeze(-1)

    diff = (z_manual - z_forward).abs().max().item()
    logging.info(f"  SANITY  max|z_manual − z_forward| = {diff:.6e}")
    return diff


# ═══════════════════════════ RUNNER ═══════════════════════════

def run_ablation_for_seed(seed: int, k: int, device):
    set_seed(seed)

    # ---- load trained probe ----
    pdir       = probe_dir_for_seed(seed)
    model_path = pdir / "probe_state.pt"
    if not model_path.exists():
        logging.error(f"Model not found: {model_path.resolve()}")
        sys.exit(1)

    ProbeClass = get_probe_class(PROBE)
    model = ProbeClass(num_layers=NUM_LAYERS, hidden_dim=HIDDEN_DIM)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    logging.info(f"[seed={seed}] Loaded probe from {model_path}")

    # ---- optional isotonic calibrator ----
    iso_reg = None
    if ISOTONIC:
        l = TRAIN_LANGS[0]
        cal_name = f"isotonic_calibrator_{'query' if QUERY else 'answer'}_{l}.joblib"
        cal_path = pdir / cal_name
        if cal_path.exists():
            iso_reg = joblib.load(cal_path)
            logging.info(f"[seed={seed}] Loaded isotonic calibrator: {cal_name}")
        else:
            logging.warning(f"[seed={seed}] No calibrator at {cal_path.name}; RAW only.")

    # ---- few-shot ratio (for balancing zero-shot test sets) ----
    _, y_fs = load_test_unflattened_from_hdf5(
        DATASET, DATASPLITS, RESULTS_ROOT, TRAIN_LANGS, TRAIN_LANGS[0],
        QUERY, str(EPOCHS), STRATIFIED, EXCLUDE_TIME_SENSITIVE,
    )
    y_fs = np.asarray(y_fs)
    n_pos_fs = int(y_fs.sum())
    n_neg_fs = len(y_fs) - n_pos_fs
    few_ratio = n_pos_fs / (n_pos_fs + n_neg_fs) if (n_pos_fs + n_neg_fs) > 0 else 0.0
    logging.info(f"[seed={seed}] few-shot ratio = {few_ratio:.4f}  (pos={n_pos_fs}, neg={n_neg_fs})")

    # accumulators for this seed
    win_rows  = []
    lay_rows  = []
    base_rows = []

    def process_lang(lang, balance_ratio):
        """Load data for *lang*, optionally balance, ablate, collect results."""
        X_np, y_list = load_test_unflattened_from_hdf5(
            DATASET, DATASPLITS, RESULTS_ROOT, TRAIN_LANGS, lang,
            QUERY, str(EPOCHS), STRATIFIED, EXCLUDE_TIME_SENSITIVE,
        )
        X_np = np.asarray(X_np, dtype=np.float32)
        y_np = np.asarray(y_list)

        if balance_ratio is not None:
            if balance_ratio == 0.0 or balance_ratio == 1.0:
                logging.warning(f"  [{lang}] skipping because few-shot ratio is {balance_ratio:.2f}.")
                return
            logging.info(f"  [{lang}] pos={int((y_np==1).sum())}, neg={int((y_np==0).sum())}, total={len(y_np)}")

        X_t   = torch.from_numpy(X_np).to(device)
        y_arr = y_np

        # sanity check
        sanity_check(model, X_t)

        # core ablation
        (per_layer_raw, per_layer_cal,
         win_df_raw, win_df_cal,
         base_raw, base_cal) = sliding_window_ablation(
            model, X_t, y_arr, k=k, iso_reg=iso_reg
        )

        # ---- collect per-window rows ----
        meta = {
            "seed": seed, "probe": PROBE, "dataset": DATASET, "llm": LLM,
            "train_langs": "-".join(TRAIN_LANGS), "test_lang": lang,
            "k": k, "l1_w": L1_W, "l2_v": L2_V, "lr": LR, "epochs": EPOCHS,
            "query": int(QUERY), "ablation": "weight",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        def _tag_win(df, cal_label):
            if df is None:
                return
            df = df.copy()
            for col, val in meta.items():
                df[col] = val
            df["calibration"] = cal_label
            win_rows.append(df)

        _tag_win(win_df_raw, "RAW")
        _tag_win(win_df_cal, "CAL")

        # ---- collect per-layer rows ----
        layer_df = pd.DataFrame({"layer": np.arange(NUM_LAYERS)})
        for m in ("auroc", "aupr", "brier", "ece"):
            layer_df[f"imp_{m}_RAW"] = per_layer_raw[m]
            if per_layer_cal is not None:
                layer_df[f"imp_{m}_CAL"] = per_layer_cal[m]
        for col, val in meta.items():
            layer_df[col] = val
        lay_rows.append(layer_df)

        # ---- collect baseline row ----
        brow = {**meta}
        for m, v in base_raw.items():
            brow[f"{m}_RAW"] = v
        if base_cal is not None:
            for m, v in base_cal.items():
                brow[f"{m}_CAL"] = v
        base_rows.append(brow)

        del X_t
        torch.cuda.empty_cache()

    # ---- few-shot split (train lang, no balancing) ----
    logging.info(f"[seed={seed}] === few-shot ({TRAIN_LANGS[0]}) ===")
    process_lang(TRAIN_LANGS[0], balance_ratio=None)

    # ---- zero-shot splits (test langs, balanced to few-shot ratio) ----
    for tl in TEST_LANGS:
        logging.info(f"[seed={seed}] === zero-shot ({tl}) ===")
        process_lang(tl, balance_ratio=few_ratio)

    return win_rows, lay_rows, base_rows


# ═══════════════════════════ CLI ══════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Sliding-window readout ablation (§6.1)")
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--k", type=int, default=K_WINDOW, help="Window size (default: 3)")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        stream=sys.stdout)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    TRAIN_LANGS.sort()
    TEST_LANGS.sort()

    out = output_dir()
    OUT_WIN  = out / f"lr_{LR}_l1_{L1_W}_l2_{L2_V}_epochs_{EPOCHS}_readout_ablation_k{args.k}_windows_multiseed.csv"
    OUT_LAY  = out / f"lr_{LR}_l1_{L1_W}_l2_{L2_V}_epochs_{EPOCHS}_readout_ablation_k{args.k}_per_layer_multiseed.csv"
    OUT_BASE = out / f"lr_{LR}_l1_{L1_W}_l2_{L2_V}_epochs_{EPOCHS}_readout_ablation_k{args.k}_baselines_multiseed.csv"

    all_win, all_lay, all_base = [], [], []

    for seed in args.seeds:
        logging.info(f"{'='*20} Seed {seed} {'='*20}")
        w, l, b = run_ablation_for_seed(seed, args.k, device)
        all_win.extend(w)
        all_lay.extend(l)
        all_base.extend(b)

    # ──────────── write per-seed results ────────────
    if all_win:
        win_df = pd.concat(all_win, ignore_index=True)
        win_df.drop(columns=["layers"], errors="ignore", inplace=True)
        hdr = not OUT_WIN.exists()
        win_df.to_csv(OUT_WIN, mode="a", index=False, header=hdr)
        logging.info(f"Wrote {len(win_df)} window rows → {OUT_WIN.resolve()}")

    if all_lay:
        lay_df = pd.concat(all_lay, ignore_index=True)
        hdr = not OUT_LAY.exists()
        lay_df.to_csv(OUT_LAY, mode="a", index=False, header=hdr)
        logging.info(f"Wrote {len(lay_df)} layer rows → {OUT_LAY.resolve()}")

    if all_base:
        base_df = pd.DataFrame(all_base)
        hdr = not OUT_BASE.exists()
        base_df.to_csv(OUT_BASE, mode="a", index=False, header=hdr)
        logging.info(f"Wrote {len(base_df)} baseline rows → {OUT_BASE.resolve()}")

    # ──────────── summary roll-ups across seeds ────────────
    try:
        if OUT_WIN.exists():
            win_all = pd.read_csv(OUT_WIN)
            delta_cols = [c for c in win_all.columns if c.startswith("delta_")]
            grp = ["probe", "dataset", "llm", "train_langs", "test_lang",
                   "k", "l1_w", "l2_v", "lr", "epochs", "query",
                   "ablation", "calibration", "window"]
            win_sum = (win_all
                       .groupby(grp, dropna=False, observed=False)[delta_cols]
                       .agg(["mean", "std"])
                       .reset_index())
            win_sum.columns = [
                "_".join(c).strip("_") if isinstance(c, tuple) else c
                for c in win_sum.columns
            ]
            sp = OUT_WIN.with_name(OUT_WIN.stem + "_summary.csv")
            win_sum.to_csv(sp, index=False)
            logging.info(f"Wrote window summary → {sp.resolve()}")

        if OUT_LAY.exists():
            lay_all = pd.read_csv(OUT_LAY)
            imp_cols = [c for c in lay_all.columns if c.startswith("imp_")]
            grp = ["probe", "dataset", "llm", "train_langs", "test_lang",
                   "k", "l1_w", "l2_v", "lr", "epochs", "query",
                   "ablation", "layer"]
            lay_sum = (lay_all
                       .groupby(grp, dropna=False, observed=False)[imp_cols]
                       .agg(["mean", "std"])
                       .reset_index())
            lay_sum.columns = [
                "_".join(c).strip("_") if isinstance(c, tuple) else c
                for c in lay_sum.columns
            ]
            sp = OUT_LAY.with_name(OUT_LAY.stem + "_summary.csv")
            lay_sum.to_csv(sp, index=False)
            logging.info(f"Wrote per-layer summary → {sp.resolve()}")

    except Exception as e:
        logging.warning(f"Summary roll-up failed: {e}")


if __name__ == "__main__":
    main()