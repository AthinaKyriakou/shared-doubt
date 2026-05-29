#!/usr/bin/env python3
"""Generate ablation result plots for all probe hyperparameters, LLMs, and datasets.

Usage:
    python eval_ablation_results.py --ablation weight
    python eval_ablation_results.py --ablation representation spearman
    python eval_ablation_results.py --ablation all --datasets mkqa --llms llama_3.1_8B
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.append(str(Path(__file__).parent.parent))
from src.constants import LOCAL_RESULTS_DIR, DATASPLITS_MKQA, DATASPLITS_GMMLU

# ---- Colors ----
WORSE = "#772244"
BETTER = "#117733"
LINE = "#332288"

SEEDS = [42, 17, 3, 59, 884, 445, 369, 169, 76, 905]
TRAIN_LANG = "fr"
L1_W = 0
USE_CALIBRATED = False
K_WINDOW = 3

METRIC_CONFIGS = [
    dict(name="auroc", display="AUROC", lim=0.08,  step=0.02, ylabel="$\\Delta$AUROC"),
    dict(name="aupr",  display="AUPR",  lim=0.12,  step=0.03, ylabel="$\\Delta$AUPR"),
    dict(name="brier", display="Brier", lim=0.16,  step=0.04, ylabel="$\\Delta$Brier"),
    dict(name="ece",   display="ECE",   lim=0.20,  step=0.05, ylabel="$\\Delta$ECE"),
]

# Best probe hyperparameters per (dataset, llm, query)
PROBE_HPARAMS = {
    ("mkqa",        "llama_3.1_8B", True):  dict(lr=0.0008, epochs=139,  l2_v=0.286102950514828),
    ("mkqa",        "llama_3.1_8B", False): dict(lr=0.0005, epochs=210,  l2_v=0.27276092363216636),
    ("mkqa",        "qwen3_8B",     True):  dict(lr=0.0009, epochs=98,   l2_v=0.29527869618547636),
    ("mkqa",        "qwen3_8B",     False): dict(lr=0.0009, epochs=111,  l2_v=0.22252863623823427),
    ("global_mmlu", "llama_3.1_8B", True):  dict(lr=0.0011, epochs=94,   l2_v=0.28258845606106203),
    ("global_mmlu", "llama_3.1_8B", False): dict(lr=0.0020, epochs=60,   l2_v=0.2557367492257226),
    ("global_mmlu", "qwen3_8B",     True):  dict(lr=0.001,  epochs=103,  l2_v=0.2686538578969079),
    ("global_mmlu", "qwen3_8B",     False): dict(lr=0.001,  epochs=165,  l2_v=0.2115332075762067),
}

LLMS = ["llama_3.1_8B", "qwen3_8B"]
DATASETS = ["mkqa", "global_mmlu"]


# ---- Plotting helpers ----

def _importance_legend(metric_name, better_when_positive):
    m = metric_name
    if better_when_positive:
        better_patch = mpatches.Patch(color=BETTER, alpha=0.55, label=f"Δ{m} > 0 (better model)")
        worse_patch  = mpatches.Patch(color=WORSE,  alpha=0.55, label=f"Δ{m} < 0 (worse model)")
    else:
        worse_patch  = mpatches.Patch(color=WORSE,  alpha=0.55, label=f"Δ{m} > 0 (worse model)")
        better_patch = mpatches.Patch(color=BETTER, alpha=0.55, label=f"Δ{m} < 0 (better model)")
    return [worse_patch, better_patch]


def _align_zero_and_fit(ax_left, ax_right, line_values, pad_frac=0.08):
    y0_left_min, y0_left_max = ax_left.get_ylim()
    zero_pos = -y0_left_min / max(y0_left_max - y0_left_min, 1e-12)
    finite_vals = np.asarray(line_values)[np.isfinite(line_values)]
    if finite_vals.size == 0:
        yr_min, yr_max = ax_right.get_ylim()
        rng = max(yr_max - yr_min, 1.0)
    else:
        data_min, data_max = finite_vals.min(), finite_vals.max()
        span = data_max - data_min
        margin = pad_frac * (span if span > 0 else (abs(data_max) + abs(data_min) + 1e-9))
        want_min = data_min - margin
        want_max = data_max + margin
        r1 = (-want_min) / max(zero_pos, 1e-12)
        r2 = ( want_max) / max(1 - zero_pos, 1e-12)
        rng = max(r1, r2, 1.0)
    ymin = -zero_pos * rng
    ymax = (1 - zero_pos) * rng
    ax_right.set_ylim(ymin, ymax)


def _flip_if_brier_ece(metric_name, values):
    if metric_name.lower() in ("brier", "ece"):
        return -values
    return values


def _better_when_positive(metric_name):
    return metric_name.lower() in ("brier", "ece")


def plot_one(dfi, fname, metric_col, metric_name, ylabel, ylim, yticks,
             alpha_mean=None, output_dir=".", num_layers=32):
    L = int(num_layers)
    mean_by_layer = dfi.groupby("layer", dropna=False)[metric_col].mean().reindex(range(L))
    std_by_layer  = dfi.groupby("layer", dropna=False)[metric_col].std().reindex(range(L))
    layers = np.arange(L) + 1
    mean_imp_raw = mean_by_layer.values
    std_imp = np.nan_to_num(std_by_layer.values, nan=0.0)
    mean_imp = _flip_if_brier_ece(metric_name, mean_imp_raw)
    better_pos = _better_when_positive(metric_name)
    colors = np.where(mean_imp > 0,
                      BETTER if better_pos else WORSE,
                      WORSE if better_pos else BETTER)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(layers, mean_imp, yerr=std_imp, align="center", alpha=0.55, capsize=3,
            color=colors, ecolor="black", error_kw=dict(alpha=1, lw=1))
    ax1.axhline(0.0, linestyle="--", linewidth=1, color="black")
    ax1.set_xlabel("Layer", fontsize=20)
    ax1.set_ylabel(ylabel, fontsize=20)
    ax1.set_ylim(*ylim)
    ax1.set_yticks(yticks)
    ax1.grid(True, axis="both", linestyle="--", alpha=0.5)
    ax1.set_xticks(np.arange(1, L + 1, 2))
    ax1.tick_params(axis="both", labelsize=16)
    line_handle = None
    if alpha_mean is not None and np.all(np.isfinite(alpha_mean)) and len(alpha_mean) == L:
        ax2 = ax1.twinx()
        (line_handle,) = ax2.plot(layers, alpha_mean, marker="o", linewidth=1.8,
                                   color=LINE, alpha=0.8, label=r"$\alpha_\ell$")
        ax2.set_ylabel("Layer importance", fontsize=20)
        ax2.set_xlim(0.5, L + 0.5)
        _align_zero_and_fit(ax1, ax2, alpha_mean, pad_frac=0.08)
        ymin_cur, ymax_cur = ax2.get_ylim()
        zero_pos = -ymin_cur / max(ymax_cur - ymin_cur, 1e-12)
        pad_frac = 0.10
        target_max = 0.1 * (1 + pad_frac)
        target_min = -(zero_pos / max(1 - zero_pos, 1e-12)) * target_max
        ax2.set_ylim(target_min, target_max)
        ax2.patch.set_visible(False)
        ax2.tick_params(axis="both", labelsize=16)
    handles = _importance_legend(metric_name, better_when_positive=better_pos)
    if line_handle is not None:
        handles.append(line_handle)
    leg = ax1.legend(handles=handles, loc="lower right", frameon=True, fontsize=15)
    for text in leg.get_texts():
        if r"$\alpha_\ell$" in text.get_text():
            text.set_fontsize(19)
    os.makedirs(output_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{fname}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_alpha_mean_exact(models_root, seeds, num_layers, lr, l1_w, l2_v, epochs):
    alphas = []
    missing = []
    for seed in seeds:
        ckpt = (
            models_root
            / f"batch_32_lr_{lr}_l1_{l1_w}_l2_{l2_v}_epochs_{epochs}_seed_{seed}"
            / "probe_state.pt"
        )
        if not ckpt.exists():
            missing.append(seed)
            continue
        st = torch.load(ckpt, map_location="cpu")
        w = st["w"].reshape(-1)
        if w.numel() != num_layers:
            raise ValueError(f"[seed {seed}] expected {num_layers} layers, got {w.numel()}")
        alphas.append(F.softmax(w, dim=-1).numpy())
    if missing:
        print(f"  [warn] missing checkpoints for seeds: {missing}")
    if not alphas:
        raise RuntimeError(f"No checkpoints found under {models_root}")
    return np.mean(np.stack(alphas), axis=0)


def get_probe_dir(dataset, llm, query):
    exclude_time_sensitive = dataset == "mkqa"
    if query:
        probe_folder = "probe_query_without_time_sensitive" if exclude_time_sensitive else "probe_query"
    else:
        probe_folder = "probe_answer_without_time_sensitive" if exclude_time_sensitive else "probe_answer"
    return (
        Path(LOCAL_RESULTS_DIR)
        / dataset / llm / probe_folder
        / f"SoftmaxLayerProbe_layer_scalers/train_{TRAIN_LANG}"
    )


def _load_alpha(models_root, num_layers, lr, l2_v, epochs):
    alpha_path = models_root / f"alpha_mean_lr_{lr}_l1_{L1_W}_l2_{l2_v}_epochs_{epochs}.npy"
    if alpha_path.exists():
        alpha_path.unlink()
    try:
        alpha_mean = load_alpha_mean_exact(models_root, SEEDS, num_layers, lr, L1_W, l2_v, epochs)
        np.save(alpha_path, alpha_mean)
        return alpha_mean
    except RuntimeError as e:
        print(f"  [warn] {e}")
        return None


def _plot_all_metrics(df, langs, alpha_mean, num_layers, ablation_prefix, dataset, llm, probe_type, plots_root):
    suffix = "CAL" if USE_CALIBRATED else "RAW"
    output_dir = str(Path(plots_root) / ablation_prefix)
    prefix = f"{dataset}_{llm}_{probe_type}_{TRAIN_LANG}"
    for mcfg in METRIC_CONFIGS:
        mname = mcfg["name"]
        metric_col = f"imp_{mname}_{suffix}"
        ylim = (-mcfg["lim"], mcfg["lim"])
        yticks = np.arange(-mcfg["lim"], mcfg["lim"] + mcfg["step"] / 2, mcfg["step"])
        for lang in langs:
            dfg = df[df["test_lang"] == lang]
            shot = "few_shot" if lang == TRAIN_LANG else "zero_shot"
            fname = f"{prefix}_{mname}_{shot}_{lang}"
            plot_one(dfg, fname=fname, metric_col=metric_col,
                     metric_name=mcfg["display"], ylabel=mcfg["ylabel"],
                     ylim=ylim, yticks=yticks, alpha_mean=alpha_mean,
                     output_dir=output_dir, num_layers=num_layers)
        dfg_all = df[df["test_lang"] != TRAIN_LANG]
        plot_one(dfg_all, fname=f"{prefix}_{mname}_zero_shot_all",
                 metric_col=metric_col, metric_name=mcfg["display"], ylabel=mcfg["ylabel"],
                 ylim=ylim, yticks=yticks, alpha_mean=alpha_mean,
                 output_dir=output_dir, num_layers=num_layers)
        print(f"  [done] {ablation_prefix} {dataset}/{llm}/{probe_type}/{mname} → {output_dir}")


# ---- Ablation runners ----

def run_weight_ablation(datasets, llms, plots_root):
    print("\n=== Weight ablation ===")
    for dataset in datasets:
        for llm in llms:
            num_layers = 32 if llm == "llama_3.1_8B" else 36
            for query in [True, False]:
                probe_type = "query" if query else "answer"
                hparams = PROBE_HPARAMS.get((dataset, llm, query))
                if hparams is None:
                    print(f"[skip] No hparams for ({dataset}, {llm}, query={query})")
                    continue
                lr, epochs, l2_v = hparams["lr"], hparams["epochs"], hparams["l2_v"]
                models_root = get_probe_dir(dataset, llm, query)
                per_layer_csv = (
                    models_root
                    / f"lr_{lr}_l1_{L1_W}_l2_{l2_v}_epochs_{epochs}"
                      f"_readout_ablation_k{K_WINDOW}_per_layer_multiseed.csv"
                )
                if not per_layer_csv.exists():
                    print(f"[skip] {per_layer_csv} not found")
                    continue
                print(f"\n{dataset}/{llm}/{probe_type}")
                alpha_mean = _load_alpha(models_root, num_layers, lr, l2_v, epochs)
                df = pd.read_csv(per_layer_csv)
                langs = sorted(df["test_lang"].dropna().unique().tolist())
                if TRAIN_LANG in langs:
                    langs = [TRAIN_LANG] + [l for l in langs if l != TRAIN_LANG]
                _plot_all_metrics(df, langs, alpha_mean, num_layers,
                                  "weight_ablation", dataset, llm, probe_type, plots_root)


def run_representation_ablation(datasets, llms, plots_root):
    print("\n=== Representation ablation ===")
    for dataset in datasets:
        for llm in llms:
            num_layers = 32 if llm == "llama_3.1_8B" else 36
            for query in [True, False]:
                probe_type = "query" if query else "answer"
                hparams = PROBE_HPARAMS.get((dataset, llm, query))
                if hparams is None:
                    print(f"[skip] No hparams for ({dataset}, {llm}, query={query})")
                    continue
                lr, epochs, l2_v = hparams["lr"], hparams["epochs"], hparams["l2_v"]
                models_root = get_probe_dir(dataset, llm, query)
                per_layer_csv = (
                    models_root
                    / f"lr_{lr}_l1_{L1_W}_l2_{l2_v}_epochs_{epochs}"
                      f"_representation_k{K_WINDOW}_per_layer_multiseed.csv"
                )
                if not per_layer_csv.exists():
                    print(f"[skip] {per_layer_csv} not found")
                    continue
                print(f"\n{dataset}/{llm}/{probe_type}")
                alpha_mean = _load_alpha(models_root, num_layers, lr, l2_v, epochs)
                df = pd.read_csv(per_layer_csv)
                langs = sorted(df["test_lang"].dropna().unique().tolist())
                if TRAIN_LANG in langs:
                    langs = [TRAIN_LANG] + [l for l in langs if l != TRAIN_LANG]
                _plot_all_metrics(df, langs, alpha_mean, num_layers,
                                  "representation_ablation", dataset, llm, probe_type, plots_root)



def main():
    parser = argparse.ArgumentParser(
        description="Generate ablation result plots for all probe hyperparameters, LLMs, and datasets."
    )
    parser.add_argument(
        "--ablation", nargs="+",
        choices=["weight", "representation", "all"],
        required=True,
        help="Which ablation(s) to run. Use 'all' for both.",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=DATASETS, choices=DATASETS,
        help="Datasets to process (default: all).",
    )
    parser.add_argument(
        "--llms", nargs="+", default=LLMS, choices=LLMS,
        help="LLMs to process (default: all).",
    )
    parser.add_argument(
        "--plots-root", default="../plots",
        help="Root directory for output plots (default: <repo>/plots).",
    )
    args = parser.parse_args()

    ablations = set(args.ablation)
    if "all" in ablations:
        ablations = {"weight", "representation"}

    if "weight" in ablations:
        run_weight_ablation(args.datasets, args.llms, args.plots_root)
    if "representation" in ablations:
        run_representation_ablation(args.datasets, args.llms, args.plots_root)


if __name__ == "__main__":
    main()
