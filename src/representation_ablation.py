#!/usr/bin/env python
"""
Sliding-window representation ablation.

Usage
-----
python representation_ablation.py --seeds 42 17 3 59 884 445 369 169 76 905 --k 3
"""

import argparse
import logging
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
from utils import load_test_unflattened_from_hdf5, compute_ece 
from probes import SparsemaxLayerProbe, SoftmaxLayerProbe
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
# LLM = "llama_3.1_8B"
LLM = "qwen3_8B"

DATASPLITS   = DATASPLITS_MKQA if DATASET == "mkqa" else DATASPLITS_GMMLU
NUM_LAYERS   = 32 if LLM == "llama_3.1_8B" else 36
HIDDEN_DIM   = 4096
TRAIN_BATCH  = 32
STRATIFIED   = True
K_WINDOW     = 3
MRA_REPEATS  = 10  # FIX 1: was referenced in parse_args but never defined

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

def get_probe_class(name: str):
    name = name.lower()
    if name in ("sparsemaxlayerprobe", "sparsemax"):
        return SparsemaxLayerProbe
    elif name in ("softmaxlayerprobe", "softmax"):
        return SoftmaxLayerProbe
    raise ValueError(f"Unknown probe class '{name}'.")


# -------------------- metrics helpers --------------------
def _metrics_binary(y, p, ece_fn=None):
    y = np.asarray(y).reshape(-1)
    p = np.asarray(p).reshape(-1)
    out = {}
    out["auroc"] = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
    pr, rc, _ = precision_recall_curve(y, p)
    out["aupr"]  = auc(rc, pr)
    out["brier"] = brier_score_loss(y, p)
    if ece_fn is not None:
        out["ece"] = float(ece_fn(y, p, n_bins=10))
    return out


def _delta(base, masked):
    d = {}
    for k in base:
        if k in ("brier", "ece"):   # lower is better → drop is positive
            d[k] = masked[k] - base[k]
        else:                        # higher is better → drop is positive
            d[k] = base[k] - masked[k]
    return d


def _make_windows_no_wrap(L, k=3):
    assert 1 <= k <= L
    return [list(range(i, i + k)) for i in range(L - k + 1)]


@torch.no_grad()
def _alpha_from_model(model):
    if isinstance(model, SoftmaxLayerProbe):
        return F.softmax(model.w, dim=-1)
    return Sparsemax(dim=-1)(model.w)


@torch.no_grad()
def _scores_from_Xv(X, v):
    """X: (B,L,H), v: (H,) → s: (B,L) where s[:,l] = X[:,l,:]·v"""
    B, L, H = X.shape
    return torch.matmul(X.reshape(B * L, H), v).reshape(B, L)


# ---------- Sliding-window Representation Ablation (RAW + CAL) ----------
@torch.no_grad()
def sliding_window_ablation(
    model, X, y, *, k=3, repeats=10, iso_reg=None, ece_fn=None, seed=None
):
    """
    Implements Equation 6.3:
        z_i^(W) = Σ_{l∉W} α_l s_{i,l}  +  Σ_{l∈W} α_l s̃_{i,l}  +  b
    where s̃_{i,l} = s_{π(i),l} and π is an independent uniform permutation
    of the test set for each layer l ∈ W.

    Returns:
        per_layer_raw, per_layer_cal, win_df_raw, win_df_cal, base_raw, base_cal
    """
    device = X.device
    B, L, H = X.shape

    alpha = _alpha_from_model(model).to(device)   # (L,)
    b     = model.bias.squeeze()
    s     = _scores_from_Xv(X, model.v)           # (B,L)

    # baseline (no ablation)
    logits      = model(X)
    p_raw_base  = torch.sigmoid(logits).cpu().numpy().reshape(-1)
    base_raw    = _metrics_binary(y, p_raw_base, ece_fn=ece_fn)

    base_cal, use_cal = None, False
    if iso_reg is not None:
        p_cal_base = iso_reg.transform(p_raw_base)
        base_cal   = _metrics_binary(y, p_cal_base, ece_fn=ece_fn)
        use_cal    = True

    wins = _make_windows_no_wrap(L, k=k)

    # precompute Σ_{l∈W} α_l s_l for every sliding window at once → (B, W)
    kernel  = torch.ones(1, 1, k, device=device, dtype=s.dtype)
    contrib = (s * alpha.view(1, L)).unsqueeze(1)          # (B,1,L)
    roll    = F.conv1d(contrib, kernel, stride=1).squeeze(1)  # (B,W)

    z_base = (s * alpha.view(1, L)).sum(dim=1) + b         # (B,)

    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(seed)

    rows_raw, rows_cal = [], []

    for w_idx, w in enumerate(wins):
        metrics_raw_list, metrics_cal_list = [], []

        for _ in range(repeats):
            s_tilde = s.clone()
            # independently permute each layer column in the window (Eq. 6.3)
            for l in w:
                perm          = torch.randperm(B, generator=g, device=device)
                s_tilde[:, l] = s[perm, l]

            win_sum_tilde = (s_tilde[:, w] * alpha[w].view(1, -1)).sum(dim=1)  # (B,)
            z_mra  = z_base - roll[:, w_idx] + win_sum_tilde
            p_raw  = torch.sigmoid(z_mra).cpu().numpy()

            m_raw  = _metrics_binary(y, p_raw, ece_fn=ece_fn)
            metrics_raw_list.append(m_raw)

            if use_cal:
                p_cal = iso_reg.transform(p_raw)
                m_cal = _metrics_binary(y, p_cal, ece_fn=ece_fn)
                metrics_cal_list.append(m_cal)

        def _avg(dicts):
            keys = dicts[0].keys()
            return {k: float(np.nanmean([d[k] for d in dicts])) for k in keys}

        mr = _avg(metrics_raw_list)
        dr = _delta(base_raw, mr)
        rows_raw.append({
            "window": "-".join(map(str, w)), "layers": w,
            **{f"masked_{k_}": v for k_, v in mr.items()},
            **{f"delta_{k_}":  v for k_, v in dr.items()},
        })

        if use_cal:
            mc = _avg(metrics_cal_list)
            dc = _delta(base_cal, mc)
            rows_cal.append({
                "window": "-".join(map(str, w)), "layers": w,
                **{f"masked_{k_}": v for k_, v in mc.items()},
                **{f"delta_{k_}":  v for k_, v in dc.items()},
            })

    win_df_raw = pd.DataFrame(rows_raw)
    win_df_cal = pd.DataFrame(rows_cal) if use_cal else None

    # coverage-normalised per-layer importance (average delta over all windows that cover l)
    def _per_layer_from_rows(rows, base_keys):
        per_layer = {m: np.zeros(L, dtype=float) for m in base_keys}
        counts    = np.zeros(L, dtype=int)
        for row in rows:
            for Lidx in row["layers"]:
                counts[Lidx] += 1
                for m in base_keys:
                    per_layer[m][Lidx] += row.get(f"delta_{m}", 0.0)
        denom = np.maximum(counts, 1)
        for m in per_layer:
            per_layer[m] = per_layer[m] / denom
        return per_layer

    per_layer_raw = _per_layer_from_rows(rows_raw, base_raw.keys())
    per_layer_cal = _per_layer_from_rows(rows_cal, base_cal.keys()) if use_cal else None

    return per_layer_raw, per_layer_cal, win_df_raw, win_df_cal, base_raw, base_cal


# -------------------------- runner --------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Sliding-window representation ablation (RAW+CAL, multi-seed)")
    p.add_argument("--seeds",   type=int, nargs="+", required=True)
    p.add_argument("--k",       type=int, default=K_WINDOW,   help="Sliding window size")
    p.add_argument("--repeats", type=int, default=MRA_REPEATS, help="Permutation repeats per window")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        stream=sys.stdout)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    TRAIN_LANGS.sort()
    TEST_LANGS.sort()
    train_tag = "_".join(TRAIN_LANGS)

    # ---- output root: mirrors eval_probe.py directory layout ----
    if EXCLUDE_TIME_SENSITIVE:
        probe_subdir = "probe_query_without_time_sensitive" if QUERY else "probe_answer_without_time_sensitive"
    else:
        probe_subdir = "probe_query" if QUERY else "probe_answer"

    out_root = RESULTS_ROOT / f"{probe_subdir}/{PROBE}_layer_scalers/train_{train_tag}"
    out_root.mkdir(parents=True, exist_ok=True)

    # multi-seed CSV targets
    stem     = f"lr_{LR}_l1_{L1_W}_l2_{L2_V}_epochs_{EPOCHS}_representation_k{args.k}"
    OUT_WIN  = out_root / f"{stem}_windows_multiseed.csv"
    OUT_LAY  = out_root / f"{stem}_per_layer_multiseed.csv"
    OUT_BASE = out_root / f"{stem}_baselines_multiseed.csv"

    for seed in args.seeds:
        # ---- probe dir: matches train_probe.py exactly ----
        probe_dir = (
            RESULTS_ROOT
            / f"{probe_subdir}/{PROBE}_layer_scalers/train_{train_tag}"
            / f"batch_{TRAIN_BATCH}_lr_{LR}_l1_{L1_W}_l2_{L2_V}_epochs_{EPOCHS}_seed_{seed}"
        )
        model_path = probe_dir / "probe_state.pt"
        if not model_path.exists():
            logging.error(f"Model file not found at {model_path.resolve()}")
            sys.exit(1)

        # load model
        ProbeClass = get_probe_class(PROBE)
        model = ProbeClass(num_layers=NUM_LAYERS, hidden_dim=HIDDEN_DIM)
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()

        # optional isotonic calibrator (saved by eval_probe.py)
        iso_reg = None
        if ISOTONIC:
            l = TRAIN_LANGS[0]
            cal_path = probe_dir / (
                f"isotonic_calibrator_query_{l}.joblib" if QUERY
                else f"isotonic_calibrator_answer_{l}.joblib"
            )
            if cal_path.exists():
                iso_reg = joblib.load(cal_path)
                logging.info(f"Loaded isotonic calibrator from {cal_path.name}")
            else:
                logging.warning(f"No isotonic calibrator at {cal_path.name}; running RAW-only.")

        # few-shot class ratio (for logging; zero-shot splits use the full test set, not balanced)
        _, y_fs   = load_test_unflattened_from_hdf5(
            DATASET, DATASPLITS, RESULTS_ROOT,
            TRAIN_LANGS, TRAIN_LANGS[0],
            QUERY, str(EPOCHS), STRATIFIED, EXCLUDE_TIME_SENSITIVE
        )
        y_fs      = np.asarray(y_fs)
        n_pos_fs  = int(y_fs.sum())
        n_neg_fs  = len(y_fs) - n_pos_fs
        few_ratio = n_pos_fs / len(y_fs) if len(y_fs) > 0 else 0.0
        logging.info(f"[seed={seed}] Few-shot ratio = {few_ratio:.4f} (pos={n_pos_fs}, neg={n_neg_fs})")

        # shared metadata for every CSV row
        meta = dict(
            seed        = seed,
            probe       = PROBE,
            dataset     = DATASET,
            llm         = LLM,
            train_langs = train_tag,
            k           = args.k,
            repeats     = args.repeats,
            l1_w        = L1_W,
            l2_v        = L2_V,
            lr          = LR,
            epochs      = EPOCHS,
            query       = int(QUERY),
            ablation    = "representation",
        )

        def run_split(tag_lang, ratio=None):
            # ---- load test data ----
            X_np, y_list = load_test_unflattened_from_hdf5(
                DATASET, DATASPLITS, RESULTS_ROOT,
                TRAIN_LANGS, tag_lang,
                QUERY, str(EPOCHS), STRATIFIED, EXCLUDE_TIME_SENSITIVE
            )
            X_np = np.asarray(X_np, dtype=np.float32)
            y_np = np.asarray(y_list)

            # FIX 2: mirror eval_probe.py — skip degenerate splits
            if ratio is not None and (ratio == 0.0 or ratio == 1.0):
                logging.warning(f"[seed={seed}] Skipping {tag_lang} because few-shot ratio is {ratio:.2f}.")
                return

            # FIX 3: mirror eval_probe.py — use the full split (no resampling), just shuffle
            pos_mask = (y_np == 1)
            neg_mask = (y_np == 0)
            X_bal = np.concatenate([X_np[pos_mask], X_np[neg_mask]], axis=0)
            y_bal = np.concatenate([y_np[pos_mask], y_np[neg_mask]], axis=0)
            perm  = np.random.RandomState(seed).permutation(len(y_bal))
            X_np  = X_bal[perm]
            y_np  = y_bal[perm]

            num_ones  = int((y_np == 1).sum())
            num_zeros = int((y_np == 0).sum())
            logging.info(f"[seed={seed}] Class balance ({tag_lang}) - 0s: {num_zeros}, 1s: {num_ones}")

            X_t   = torch.from_numpy(X_np).to(device)
            y_arr = np.asarray(y_np)

            # sanity: score-space logits must match model forward pass
            with torch.no_grad():
                alpha_s = _alpha_from_model(model).to(device)
                B, L, H = X_t.shape
                s_check = torch.matmul(X_t.reshape(B * L, H), model.v).reshape(B, L)
                z_ss    = (s_check * alpha_s.view(1, L)).sum(dim=1) + model.bias.squeeze()
                z_fwd   = model(X_t).squeeze(-1)
                diff    = (z_ss - z_fwd).abs().max().item()
                logging.info(f"[seed={seed}][{tag_lang}] SANITY max|z_score - z_fwd| = {diff:.2e}")

            # run ablation
            per_layer_raw, per_layer_cal, win_df_raw, win_df_cal, base_raw, base_cal = \
                sliding_window_ablation(
                    model, X_t, y_arr,
                    k=args.k, repeats=args.repeats,
                    iso_reg=iso_reg, ece_fn=compute_ece, seed=seed
                )

            # ---- per-window CSV ----
            def enrich_win(df, cal_tag):
                if df is None:
                    return None
                df = df.copy()
                for k_, v in meta.items():
                    df[k_] = v
                df["test_lang"]   = tag_lang
                df["calibration"] = cal_tag
                df["timestamp"]   = datetime.now(timezone.utc).isoformat()
                return df

            out_raw      = enrich_win(win_df_raw, "RAW")
            out_cal      = enrich_win(win_df_cal, "CAL") if win_df_cal is not None else None
            to_write_win = out_raw if out_cal is None else pd.concat([out_raw, out_cal], ignore_index=True)
            write_header = not OUT_WIN.exists()
            to_write_win.to_csv(OUT_WIN, mode="a", index=False, header=write_header)

            # ---- per-layer CSV ----
            df_lay = pd.DataFrame({
                "layer":         np.arange(X_t.size(1)),
                "imp_auroc_RAW": per_layer_raw["auroc"],
                "imp_aupr_RAW":  per_layer_raw["aupr"],
                "imp_brier_RAW": per_layer_raw["brier"],
                "imp_ece_RAW":   per_layer_raw.get("ece", np.nan),
            })
            if per_layer_cal is not None:
                df_lay["imp_auroc_CAL"] = per_layer_cal["auroc"]
                df_lay["imp_aupr_CAL"]  = per_layer_cal["aupr"]
                df_lay["imp_brier_CAL"] = per_layer_cal["brier"]
                df_lay["imp_ece_CAL"]   = per_layer_cal.get("ece", np.nan)
            for k_, v in meta.items():
                df_lay[k_] = v
            df_lay["test_lang"] = tag_lang
            df_lay["timestamp"] = datetime.now(timezone.utc).isoformat()
            write_header_lay    = not OUT_LAY.exists()
            df_lay.to_csv(OUT_LAY, mode="a", index=False, header=write_header_lay)

            # ---- baselines CSV ----
            base_row = {**meta, "test_lang": tag_lang,
                        "auroc_RAW": base_raw["auroc"],
                        "aupr_RAW":  base_raw["aupr"],
                        "brier_RAW": base_raw["brier"],
                        "ece_RAW":   base_raw.get("ece", np.nan),
                        "auroc_CAL": base_cal["auroc"]  if base_cal else np.nan,
                        "aupr_CAL":  base_cal["aupr"]   if base_cal else np.nan,
                        "brier_CAL": base_cal["brier"]  if base_cal else np.nan,
                        "ece_CAL":   base_cal.get("ece", np.nan) if base_cal else np.nan,
                        "timestamp": datetime.now(timezone.utc).isoformat()}
            write_header_base = not OUT_BASE.exists()
            pd.DataFrame([base_row]).to_csv(OUT_BASE, mode="a", index=False, header=write_header_base)

        # few-shot split (train language, unbalanced — no ratio guard needed)
        run_split(TRAIN_LANGS[0], ratio=None)
        # zero-shot splits (test languages, full split + shuffle, with ratio guard)
        for test_lang in TEST_LANGS:
            run_split(test_lang, ratio=few_ratio)

        torch.cuda.empty_cache()

    # --------- SUMMARY ROLL-UPS (across seeds) ----------
    try:
        if OUT_WIN.exists():
            win_all    = pd.read_csv(OUT_WIN)
            delta_cols = [c for c in win_all.columns if c.startswith("delta_")]
            group_keys = ["probe", "dataset", "llm", "train_langs", "test_lang",
                          "k", "repeats", "l1_w", "l2_v", "lr", "epochs", "query",
                          "ablation", "calibration", "window"]
            win_sum    = (win_all.groupby(group_keys, dropna=False)[delta_cols]
                          .agg(["mean", "std"]).reset_index())
            win_sum.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c
                               for c in win_sum.columns]
            win_sum_path = OUT_WIN.with_name(OUT_WIN.stem + "_summary.csv")
            win_sum.to_csv(win_sum_path, index=False)
            logging.info(f"Wrote window summary to {win_sum_path.resolve()}")

        if OUT_LAY.exists():
            lay_all    = pd.read_csv(OUT_LAY)
            imp_cols   = [c for c in lay_all.columns if c.startswith("imp_")]
            group_keys = ["probe", "dataset", "llm", "train_langs", "test_lang",
                          "k", "repeats", "l1_w", "l2_v", "lr", "epochs", "query",
                          "ablation", "layer"]
            lay_sum    = (lay_all.groupby(group_keys, dropna=False)[imp_cols]
                          .agg(["mean", "std"]).reset_index())
            lay_sum.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c
                               for c in lay_sum.columns]
            lay_sum_path = OUT_LAY.with_name(OUT_LAY.stem + "_summary.csv")
            lay_sum.to_csv(lay_sum_path, index=False)
            logging.info(f"Wrote per-layer summary to {lay_sum_path.resolve()}")

    except Exception as e:
        logging.warning(f"Failed to write summaries: {e}")


if __name__ == "__main__":
    main()