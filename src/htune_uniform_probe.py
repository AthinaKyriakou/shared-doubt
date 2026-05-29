import os
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from utils import load_and_preprocess_unflattened_from_hdf5, load_stratified_batches
from probes import UniformLayerProbe
from constants import LOCAL_RESULTS_DIR, BATCH_SIZE, DATASPLITS_MKQA, DATASPLITS_GMMLU


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42


def train_and_report_layer(config, X_tr, y_tr, X_va, y_va, num_layers, hidden_dim, seed):

    # reproducibility
    trial_seed = int(config.get("trial_seed", seed))
    random.seed(trial_seed)
    np.random.seed(trial_seed)
    torch.manual_seed(trial_seed)
    torch.cuda.manual_seed_all(trial_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load train and validation tensors
    X_tr_t = torch.from_numpy(X_tr).float().to(device)
    y_tr_t = torch.from_numpy(np.array(y_tr)).float().to(device)
    X_va_t = torch.from_numpy(X_va).float().to(device)
    y_va_t = torch.from_numpy(np.array(y_va)).float().to(device)

    # specify model — w is a frozen buffer, only v and bias are learned
    model = UniformLayerProbe(num_layers=num_layers, hidden_dim=hidden_dim).to(device)

    # specify pos_weight due to class imbalance and optimisation criterion
    N_pos = np.sum(y_tr)
    N_neg = len(y_tr) - N_pos
    pos_weight = torch.tensor([N_neg / N_pos], dtype=torch.float, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # separate weight decay (L2) only for v; no decay for bias
    # note: w is a buffer (not a parameter) and is never passed to the optimizer
    optimizer = torch.optim.AdamW([
        {"params": [model.v],    "weight_decay": config["l2_lambda"]},
        {"params": [model.bias], "weight_decay": 0.0}
    ], lr=config["lr"])

    # create the training batches
    train_loader = load_stratified_batches(X_tr_t, y_tr_t, batch_size=config["batch_size"], seed=trial_seed)

    # in-trial early stopping hyperparameters
    patience = 20
    best_val_loss = float("inf")
    epochs_since_improve = 0

    for epoch in range(config["max_epochs"]):

        # -------------- train the model with mini batch --------------
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

        # ------- validation for hyperparameter tuning -----
        model.eval()
        with torch.no_grad():
            logits_va = model(X_va_t).view(-1)
            loss_va = criterion(logits_va, y_va_t.view(-1))
        avg_val_loss = loss_va.item()

        # --------- check early-stopping criterion ---------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        # --------------------- report metrics to ray -----------------
        metrics = {"train_loss": train_loss, "val_loss": avg_val_loss}
        tune.report(metrics)

        # if no improvement in the last patience epochs, stop
        if epochs_since_improve >= patience:
            print(f"Stopping early at epoch {epoch} (no val_loss improvement in {patience} epochs)")
            break


def main():

    # config
    # DATASET = "mkqa"
    DATASET = "global_mmlu"
    # LLM = "llama_3.1_8B"
    LLM = "qwen3_8B"
    HIDDEN_DIM = 4096
    IS_QUERY = False
    LANGUAGES = ['fr']
    
    EXCLUDE_TIME_SENSITIVE = True if DATASET == "mkqa" else False
    NUM_LAYERS = 32 if LLM == "llama_3.1_8B" else 36
    RESULTS_ROOT = Path(LOCAL_RESULTS_DIR) / DATASET / LLM
    NUM_SAMPLES = 50
    PROBE = 'UniformLayerProbe'
    STRATIFIED = True

    EXP_NAME = ""
    if IS_QUERY:
        EXP_NAME = EXP_NAME + "query_"
    else:
        EXP_NAME = EXP_NAME + "answer_"
    EXP_NAME = EXP_NAME + "last_"
    if STRATIFIED:
        if EXCLUDE_TIME_SENSITIVE:
            EXP_NAME = EXP_NAME + "stratified_without_time_sensitive"
        else:
            EXP_NAME = EXP_NAME + "stratified"
    else:
        EXP_NAME = EXP_NAME + "balanced"

    # load and preprocess data
    datasplits = DATASPLITS_MKQA if DATASET == 'mkqa' else DATASPLITS_GMMLU
    X_tr, y_tr, X_va, y_va = load_and_preprocess_unflattened_from_hdf5(dataset=DATASET, datasplits=datasplits, results_root=RESULTS_ROOT, train_languages=LANGUAGES,
                                                                       query=IS_QUERY, stratified=STRATIFIED, exclude_time_sensitive=EXCLUDE_TIME_SENSITIVE)
    print(X_tr.shape)
    print(len(y_tr))
    print(X_va.shape)
    print(len(y_va))

    # set storage dir
    if EXCLUDE_TIME_SENSITIVE:
        d_name = Path(RESULTS_ROOT) / "ray_results/paper_probes_without_time_sensitive"
    else:
        d_name = Path(RESULTS_ROOT) / "ray_results/paper_probes"
    if IS_QUERY:
        d_name = d_name / "query"
    else:
        d_name = d_name / "answer"
    d_name = d_name / f"{PROBE}/train_{' '.join(LANGUAGES)}/"
    ray_results_dir = os.path.abspath(d_name)
    os.makedirs(ray_results_dir, exist_ok=True)
    storage_uri = f"file://{ray_results_dir}"

    # define search space
    # note: no l1_lambda since w is frozen; search space is otherwise identical to SoftmaxLayerProbe
    config = {
        "lr": tune.loguniform(1e-5, 1e-1),
        "max_epochs": 1000,
        "l2_lambda": tune.loguniform(1e-4, 3e-1),  # for the neuron weights model.v
        "batch_size": BATCH_SIZE,
        "trial_seed": tune.randint(0, 10_000_000),
    }

    # scheduler and reporter
    # inter-trial early stopping: halt the worst-performing trials early
    scheduler = ASHAScheduler(metric="val_loss", mode="min", max_t=1000, grace_period=30, reduction_factor=2)
    reporter = CLIReporter(metric_columns=["train_loss", "val_loss", "training_iteration"])

    # run hyperparameter search
    analysis = tune.run(
        tune.with_parameters(train_and_report_layer, X_tr=X_tr, y_tr=y_tr, X_va=X_va, y_va=y_va, num_layers=NUM_LAYERS, hidden_dim=HIDDEN_DIM, seed=SEED),
        resources_per_trial={"cpu": 2, "gpu": 1 if torch.cuda.is_available() else 0},
        config=config,
        num_samples=NUM_SAMPLES,
        scheduler=scheduler,
        progress_reporter=reporter,
        storage_path=storage_uri,
        name=EXP_NAME
    )

    # retrieve best trial
    best_trial = analysis.get_best_trial(metric="val_loss", mode="min", scope="all")
    print(f"Best config: {best_trial.config}")
    print(f"Best val loss: {best_trial.metric_analysis['val_loss']['min']:.5f}")


if __name__ == "__main__":
    main()