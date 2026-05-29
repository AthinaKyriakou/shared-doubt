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
from probes import SoftmaxLayerProbe, SparsemaxLayerProbe
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
import torch.nn.functional as F
from sparsemax import Sparsemax
from sklearn.utils.class_weight import compute_class_weight
import argparse
from constants import DATASPLITS_MKQA, DATASPLITS_GMMLU, LOCAL_RESULTS_DIR

# source /work/tc067/tc067/s2742600/codebase/thesis_env/bin/activate
# sbatch --export=ALL,WANDB_API_KEY=$WANDB_API_KEY 06_probe.slurm
# wandb sync ./wandb/offline-run-*

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run -probe training across multiple random seeds"
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

    # constants
    TRAIN_LANGS = ['fr']
    TRAIN_LANGS.sort()

    DATASET = "global_mmlu"
    # LLM = "llama_3.1_8B"
    LLM = "qwen3_8B"

    NUM_LAYERS = 32 if LLM == 'llama_3.1_8B' else 36 
    HIDDEN_DIM = 4096
    RESULTS_ROOT = Path(LOCAL_RESULTS_DIR) / DATASET / LLM
    STRATIFIED = True
    EXCLUDE_TIME_SENSITIVE = True if DATASET == 'mkqa' else False

    PROBE = "SoftmaxLayerProbe"
    TRAIN_BATCH_SIZE = 32
    L1_W = 0

    # ---- MKQA - LLAMA 3.1 8B
    # QUERY
    # fr - softmax
    # QUERY = True
    # LR = 0.0008
    # EPOCHS = 139 
    # L2_V = 0.286102950514828

    # ANSWER
    # fr - softmax
    # QUERY = False
    # LR = 0.0005
    # EPOCHS = 210
    # L2_V = 0.27276092363216636

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
    QUERY = False
    LR = 0.001
    EPOCHS = 165
    L2_V = 0.2115332075762067


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

        # start a new wandb run to track this script.
        load_dotenv() # get environment variables from .env.
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

        # specify the model
        if PROBE == 'SoftmaxLayerProbe':
            model = SoftmaxLayerProbe(num_layers=NUM_LAYERS, hidden_dim=HIDDEN_DIM).to(device)
        else:
            model = SparsemaxLayerProbe(num_layers=NUM_LAYERS, hidden_dim=HIDDEN_DIM).to(device)

        # specify pos_weight due to class imbalance and optimisation criterion
        # classes = np.array([0, 1])
        # cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_tr)
        # pos_weight = torch.tensor([cw[1]], dtype=torch.float, device=device)
        N_pos = np.sum(y_tr)
        N_neg = len(y_tr) - N_pos
        pos_weight = torch.tensor([N_neg / N_pos], dtype=torch.float, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # specify optimiser
        # separate weight decay (L2) only for v; no decay for w and bias
        optimizer = torch.optim.AdamW([{"params": [model.v], "weight_decay": L2_V},
                                    {"params": [model.w, model.bias], "weight_decay": 0.0}
                                    ], lr=LR)
        # optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.0) # no regularisation

        # create the training batches
        # train_loader = load_batches(X_tr_t, y_tr_t, batch_size=TRAIN_BATCH_SIZE, seed=seed) # not stratified batches
        train_loader = load_stratified_batches(X_tr_t, y_tr_t, batch_size=TRAIN_BATCH_SIZE, seed=seed) # stratified batches

        # (2) Train the model
        train_hist = []
        val_hist = []
        sparsemax = Sparsemax(dim=-1) if PROBE != "SoftmaxLayerProbe" else None
        for epoch in range(EPOCHS):
            print(f"Current epoch: {epoch}")
            model.train()
            total_loss = 0.0
            for Xb, yb in train_loader:
                optimizer.zero_grad()
                logits = model(Xb).view(-1)
                loss = criterion(logits, yb.view(-1)) #+ L1_W * torch.norm(model.w, p=1)
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
                loss_va = criterion(logits_va, y_va_t.view(-1)) #+ L1_W * torch.norm(model.w, p=1)
                val_loss = loss_va.item()
                val_hist.append(val_loss)
                
                # track metrics
                y_pred_prob = torch.sigmoid(logits_va).cpu().numpy()
                y_va_np = np.array(y_va)
                val_auroc = roc_auc_score(y_va_np, y_pred_prob)
                val_brier = brier_score_loss(y_va_np, y_pred_prob)
                val_ece = compute_ece(y_va_np, y_pred_prob, n_bins=10)
                val_prec, val_rec, _ = precision_recall_curve(y_va_np, y_pred_prob)
                val_aupr = auc(val_rec, val_prec)

                # track layer importance
                weights = model.w.detach()
                if PROBE == 'SoftmaxLayerProbe':
                    layer_importance = F.softmax(weights, dim=-1)
                else:
                    layer_importance = sparsemax(weights)
                layer_importance = layer_importance.cpu().numpy().reshape(-1)
                layer_logs = {
                    f"layer_importance/layer_{i}": float(layer_importance[i])
                    for i in range(len(layer_importance))
                }
            
            # log metrics to wandb
            metrics = {"train_loss": train_loss, "val_loss": val_loss,
                    "val_auroc": val_auroc, "val_brier": val_brier,
                    "val_ece": val_ece, "val_aupr": val_aupr,
                    **layer_logs
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

        # plot layer importance
        weights = model.w.detach()
        if PROBE == 'SoftmaxLayerProbe':
            layer_importance = F.softmax(weights, dim=-1).cpu().numpy()
            y_label = 'Softmax value'
        else:
            sparsemax = Sparsemax(dim=-1)
            layer_importance = sparsemax(weights).cpu().numpy()
            y_label = 'Sparsemax value'
        layers = np.arange(len(layer_importance))
        plt.figure(figsize=(8, 4))
        plt.plot(layers, layer_importance, marker='o', linestyle='-')
        plt.xlabel('Layer')
        plt.ylabel(y_label)
        plt.title(f'Layer importance \n(probe trained on {"_".join(sorted(TRAIN_LANGS))}, lr: {LR}, l1_w: {L1_W}, l2_v: {L2_V}, epochs: {EPOCHS}, seed: {seed})')
        plt.savefig(PROBE_DIR / f'layer_importance_plot.png', dpi=150, bbox_inches='tight')
        plt.close()

        # finish the run and upload any remaining data to wandb
        run.finish()


if __name__ == "__main__":
    args = parse_args()
    for seed in args.seeds:
        run_experiment(seed)