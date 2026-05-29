import torch
import torch.nn as nn
import numpy as np
import random
import wandb
from time import strftime
import json
from pathlib import Path
from utils import load_and_preprocess_unflattened_from_hdf5, compute_ece, load_stratified_batches
from sklearn.metrics import roc_auc_score, brier_score_loss, precision_recall_curve, auc
from probes import UniformLayerProbe
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
import argparse
from constants import DATASPLITS_MKQA, DATASPLITS_GMMLU, LOCAL_RESULTS_DIR


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run UniformLayerProbe training across multiple random seeds"
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[884],
        help="List of random seeds to run experiments with (e.g., --seeds 42 123 999)"
    )
    return parser.parse_args()


def run_experiment(seed: int):
    # reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    TRAIN_LANGS = ['fr']
    # DATASET = "mkqa"
    DATASET = "global_mmlu"
    LLM = "llama_3.1_8B"
    # LLM = "qwen3_8B"

    NUM_LAYERS = 32 if LLM == 'llama_3.1_8B' else 36 
    HIDDEN_DIM = 4096
    RESULTS_ROOT = Path(LOCAL_RESULTS_DIR) / DATASET / LLM
    STRATIFIED = True
    EXCLUDE_TIME_SENSITIVE = True if DATASET == 'mkqa' else False

    PROBE = "UniformLayerProbe"
    TRAIN_BATCH_SIZE = 32
    L1_W = 0

    # ---- MKQA - LLAMA 3.1 8B
    # QUERY
    # fr - softmax
    # QUERY = True
    # LR = 0.001
    # EPOCHS = 122 
    # L2_V = 0.28748733353801853

    # ANSWER
    # fr - softmax
    # QUERY = False
    # LR = 0.0009
    # EPOCHS = 500 
    # L2_V = 0.24085571039248585

    # ---- MKQA - QWEN 3 8B
    # QUERY
    # fr - softmax
    # QUERY = True
    # LR = 0.001
    # EPOCHS = 144
    # L2_V = 0.24807848114946277

    # ANSWER
    # fr - softmax
    # QUERY = False
    # LR = 0.002
    # EPOCHS = 83
    # L2_V = 0.16474944313861403

    # ---- GLOBAL MMLU - LLAMA 3.1 8B
    # QUERY
    # fr - softmax
    # QUERY = True
    # LR = 0.007
    # EPOCHS = 29 
    # L2_V = 0.23215243005888206

    # ANSWER
    # fr - softmax
    QUERY = False
    LR = 0.001
    EPOCHS = 123 
    L2_V = 0.28588330922438193

    # ---- GLOBAL MMLU - QWEN 3 8B
    # QUERY
    # fr - softmax
    # QUERY = True
    # LR = 0.001
    # EPOCHS = 131
    # L2_V = 0.261559514194021

    # ANSWER
    # fr - softmax
    # QUERY = False
    # LR = 0.001
    # EPOCHS = 142
    # L2_V = 0.22318202430751768

    for lang in TRAIN_LANGS:

        print(f"---------- Training lang: {lang}")

        # paths
        if QUERY:
            if EXCLUDE_TIME_SENSITIVE:
                PROBE_DIR = RESULTS_ROOT / f"probe_query_without_time_sensitive/{PROBE}_layer_scalers/train_{'_'.join(TRAIN_LANGS)}/batch_{TRAIN_BATCH_SIZE}_lr_{LR}_l1_{L1_W}_l2_{L2_V}_epochs_{EPOCHS}_seed_{seed}"
            else:
                PROBE_DIR = RESULTS_ROOT / f"probe_query/{PROBE}_layer_scalers/train_{'_'.join(TRAIN_LANGS)}/batch_{TRAIN_BATCH_SIZE}_lr_{LR}_l1_{L1_W}_l2_{L2_V}_epochs_{EPOCHS}_seed_{seed}"
        else:
            if EXCLUDE_TIME_SENSITIVE:
                PROBE_DIR = RESULTS_ROOT / f"probe_answer_without_time_sensitive/{PROBE}_layer_scalers/train_{'_'.join(TRAIN_LANGS)}/batch_{TRAIN_BATCH_SIZE}_lr_{LR}_l1_{L1_W}_l2_{L2_V}_epochs_{EPOCHS}_seed_{seed}"
            else:
                PROBE_DIR = RESULTS_ROOT / f"probe_answer/{PROBE}_layer_scalers/train_{'_'.join(TRAIN_LANGS)}/batch_{TRAIN_BATCH_SIZE}_lr_{LR}_l1_{L1_W}_l2_{L2_V}_epochs_{EPOCHS}_seed_{seed}"
        PROBE_DIR.mkdir(parents=True, exist_ok=True)

        # save hyperparameters
        hyperparams = {
            'probe': PROBE,
            'dataset': DATASET,
            'llm': LLM,
            'seed': seed,
            'train_langs': TRAIN_LANGS,
            'hidden_dim': HIDDEN_DIM,
            'num_layers': NUM_LAYERS,
            'batch_size': TRAIN_BATCH_SIZE,
            'max_epochs': EPOCHS,
            'lr': LR,
            'l1_w': L1_W,
            'l2_v': L2_V,
            'is_query': QUERY,
            'remove_time_sensitive': EXCLUDE_TIME_SENSITIVE
        }
        with open(PROBE_DIR / 'hyperparams.json', 'w') as f:
            json.dump(hyperparams, f, indent=2)

        # start a new wandb run to track this script
        load_dotenv()
        wandb.login(key=os.environ.get("WANDB_API_KEY"))
        exp_name = f"exp-{strftime('%Y%m%d_%H%M%S')}_probe_{DATASET}_{LLM}"
        run = wandb.init(name=exp_name, entity="anonymous", project="multilingual_uncertainty_quantification", config=hyperparams)

        # device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # (1) Load training data
        datasplits = DATASPLITS_MKQA if DATASET == 'mkqa' else DATASPLITS_GMMLU
        X_tr, y_tr, X_va, y_va = load_and_preprocess_unflattened_from_hdf5(dataset=DATASET, datasplits=datasplits,
                                                                           results_root=RESULTS_ROOT, train_languages=TRAIN_LANGS,
                                                                           query=QUERY, epochs=str(EPOCHS), stratified=STRATIFIED,
                                                                           exclude_time_sensitive=EXCLUDE_TIME_SENSITIVE)

        # tensors
        X_tr_t = torch.from_numpy(X_tr).float().to(device)
        y_tr_t = torch.from_numpy(np.array(y_tr)).float().to(device)
        X_va_t = torch.from_numpy(X_va).float().to(device)
        y_va_t = torch.from_numpy(np.array(y_va)).float().to(device)

        # specify model — w is a frozen buffer, only v and bias are learned
        model = UniformLayerProbe(num_layers=NUM_LAYERS, hidden_dim=HIDDEN_DIM).to(device)

        # specify pos_weight due to class imbalance and optimisation criterion
        N_pos = np.sum(y_tr)
        N_neg = len(y_tr) - N_pos
        pos_weight = torch.tensor([N_neg / N_pos], dtype=torch.float, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # separate weight decay (L2) only for v; no decay for bias
        # note: w is a buffer (not a parameter) and is never passed to the optimizer
        optimizer = torch.optim.AdamW([
            {"params": [model.v],    "weight_decay": L2_V},
            {"params": [model.bias], "weight_decay": 0.0}
        ], lr=LR)

        # create the training batches
        train_loader = load_stratified_batches(X_tr_t, y_tr_t, batch_size=TRAIN_BATCH_SIZE, seed=seed)

        # (2) Train the model
        train_hist = []
        val_hist = []
        for epoch in range(EPOCHS):
            print(f"Current epoch: {epoch}")
            model.train()
            total_loss = 0.0
            for Xb, yb in train_loader:
                optimizer.zero_grad()
                logits = model(Xb).view(-1)
                loss = criterion(logits, yb.view(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * Xb.size(0)
            # average epoch train loss over the batches
            train_loss = total_loss / len(X_tr_t)
            train_hist.append(train_loss)

            # track validation metrics
            model.eval()
            with torch.no_grad():
                logits_va = model(X_va_t).view(-1)
                loss_va = criterion(logits_va, y_va_t.view(-1))
                val_loss = loss_va.item()
                val_hist.append(val_loss)

                y_pred_prob = torch.sigmoid(logits_va).cpu().numpy()
                y_va_np = np.array(y_va)
                val_auroc = roc_auc_score(y_va_np, y_pred_prob)
                val_brier = brier_score_loss(y_va_np, y_pred_prob)
                val_ece = compute_ece(y_va_np, y_pred_prob, n_bins=10)
                val_prec, val_rec, _ = precision_recall_curve(y_va_np, y_pred_prob)
                val_aupr = auc(val_rec, val_prec)

                # track layer importance — fixed uniform weights (1/num_layers) for all layers
                layer_importance = model.w.cpu().numpy().reshape(-1)
                layer_logs = {
                    f"layer_importance/layer_{i}": float(layer_importance[i])
                    for i in range(len(layer_importance))
                }

            # log metrics to wandb
            metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auroc": val_auroc,
                "val_brier": val_brier,
                "val_ece": val_ece,
                "val_aupr": val_aupr,
                **layer_logs,
            }
            run.log(metrics)

        # (3) Save results
        # save model
        torch.save(model.state_dict(), PROBE_DIR / "probe_state.pt")
        print(f"Training complete. Artifacts saved in {PROBE_DIR}")

        # train + validation loss
        np.save(PROBE_DIR / 'train_hist.npy', np.array(train_hist))
        np.save(PROBE_DIR / 'val_hist.npy', np.array(val_hist))
        fig, ax = plt.subplots()
        ax.plot(range(1, len(train_hist)+1), train_hist, label="Train Loss")
        ax.plot(range(1, len(val_hist)+1), val_hist, label="Val Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        train_langs_str = "_".join(sorted(TRAIN_LANGS))
        ax.set_title(
            f"Training and validation loss \n"
            f"(probe trained on {train_langs_str}, "
            f"lr: {LR}, l1_w: {L1_W}, l2_v: {L2_V}, "
            f"epochs: {EPOCHS}, seed: {seed})"
        )
        ax.legend(loc="upper right")
        fig.savefig(PROBE_DIR / f"loss_curve.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # finish the run and upload any remaining data to wandb
        run.finish()


if __name__ == "__main__":
    args = parse_args()
    for seed in args.seeds:
        run_experiment(seed)