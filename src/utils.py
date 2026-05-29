import torch
import math
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from torch.utils.data import TensorDataset, DataLoader, Sampler
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
import joblib
import sys
import matplotlib.patheffects as pe
import pandas as pd
import itertools
import torch.nn.functional as F
import seaborn as sns
import h5py
from typing import List

BINARY_ANSWER_MAPPING = {
    'en': {'yes': 'yes', 'no': 'no'},
    'es': {'sí': 'yes', 'no': 'no'},
    'fr': {'oui': 'yes', 'non': 'no'},
    'de': {'ja': 'yes', 'nein': 'no'},
    'ru': {'да': 'yes', 'нет': 'no' }
}

def load_from_hdf5(dataset: str, h5_path: Path, langs: List):
    """
    Load hidden states from the given HDF5 file into a dict:
        lang -> {eid: torch.Tensor}
    If langs is None, loads all languages present in the file.
    """
    result = defaultdict(dict)  # lang -> {eid: tensor}
    if not h5_path.exists():
        print(f"[WARNING] HDF5 file not found: {h5_path}")
        return result
    with h5py.File(h5_path, "r") as f:
        if langs is None:
            langs = list(f.keys())
        for lang_name in langs:
            if lang_name not in f:
                continue
            lang_grp = f[lang_name]
            for eid in lang_grp:
                # print(type(eid)) # str
                arr = lang_grp[eid][:]
                if dataset == 'mkqa':
                    result[lang_name][int(eid)] = torch.from_numpy(arr).clone()
                else:
                    result[lang_name][str(eid)] = torch.from_numpy(arr).clone()
    return result


def load_hidden_states(lang, results_dir):
    file_path = Path(results_dir) / f'{lang}_all_tokens_q_and_output_hidden_layers.pt'
    print(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f'Hidden states file not found: {file_path}')
    return torch.load(file_path, map_location="cpu")


def tensor_to_numpy_safe(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a torch tensor to a NumPy array, upcasting if necessary to a supported dtype.
    """
    t = tensor.detach().cpu()
    if t.dtype in (torch.bfloat16, torch.float16):
        t = t.to(torch.float32)
    return t.numpy()


def is_hdf5_readable(path: Path) -> bool:
    """
    Quick check whether an HDF5 file can be opened for reading without error.
    """
    try:
        with h5py.File(path, "r") as f:
            # trigger metadata reading
            _ = list(f.keys())
        return True
    except Exception:
        return False


def ensure_sane_hdf5(path: Path):
    """
    If the HDF5 file exists but is corrupted/unreadable, rename it out of the way
    with a timestamped .corrupt suffix so a fresh one can be created.
    """
    if path.exists():
        if not is_hdf5_readable(path):
            timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            backup_name = f"{path.stem}.corrupt.{timestamp}{path.suffix}"
            backup_path = path.parent / backup_name
            print(f"[WARN] HDF5 file {path} appears corrupted; renaming to {backup_path} and starting fresh.")
            path.rename(backup_path)


def append_to_hdf5_language_group(h5_path: Path, lang: str, data_dict: dict):
    """
    Append per-language hidden states into the given HDF5 file.
    All examples for the same language live under /<lang>/<eid>.
    Uses a temp name + rename to avoid partial writes. Skips existing eids.
    """
    ensure_sane_hdf5(h5_path)
    with h5py.File(h5_path, "a") as f:
        lang_grp = f.require_group(lang)
        for eid, tensor in data_dict.items():
            name = str(eid)
            # safe existence check
            try:
                if name in lang_grp:
                    continue  # already stored
            except (RuntimeError, OSError) as e:
                raise RuntimeError(
                    f"Error checking existence of '{name}' in '{h5_path}': {e}. "
                    "The HDF5 file may be corrupted; consider inspecting or rotating it."
                ) from e

            tmp_name = f"{name}.__tmp__"
            # clean up any stale tmp from prior crash
            if tmp_name in lang_grp:
                try:
                    del lang_grp[tmp_name]
                except Exception:
                    pass  # best effort

            arr = tensor_to_numpy_safe(tensor)
            try:
                # write to temporary dataset first
                tmp_ds = lang_grp.create_dataset(tmp_name, data=arr, compression="gzip", chunks=True)
                tmp_ds.attrs["complete"] = True  # mark successful write
                # atomic rename to final name
                lang_grp.move(tmp_name, name)
            except Exception:
                # cleanup if something went wrong
                if tmp_name in lang_grp:
                    try:
                        del lang_grp[tmp_name]
                    except Exception:
                        pass
                raise


def load_answers(lang, results_dir, suffix, dataset):
    file_path = Path(results_dir) / f'{lang}_{suffix}.jsonl'
    answers = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        cnt = 0
        for line in f:
            try:
                item = json.loads(line)
                if dataset == 'mkqa':
                    answers[item['example_id']] = item[suffix]
                else:
                    answers[item['sample_id']] = item[suffix]
                cnt += 1
            except Exception:
                cnt += 1
                print(cnt)
                continue
    return answers


def load_ground_truth(languages, file_path):
    """
    Load ground truth data from a JSONL file for specified languages.

    Args:
        file_path (str): Path to the JSONL file.
        languages (list): List of language codes to include.

    Returns:
        dict: Ground truth answers {eid: {lang: answer}}.
    """
    ground_truth = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            eid = item['example_id']
            ground_truth[eid] = {lang: item['answers'][lang][0]['text'].lower() for lang in languages}
    return ground_truth


def compute_correctness_binary(lang, answers_lang, ground_truth_lang, mapping_lang):
    correctness = {}
    for eid, ground_truth in ground_truth_lang.items():    
        if eid not in answers_lang:
            print(f"Warning: No LLM answer found for example ID {eid} in {lang}")
            continue
        model_answer = answers_lang[eid].lower()
        # if the model’s answer isn’t in our yes/no map, count as incorrect
        if model_answer not in mapping_lang:
            print(f"Warning: Unknown model answer '{model_answer}' for language {lang}, example {eid}")
            correctness[eid] = 0
            continue
        # otherwise compare mapped LLM answer with ground truth
        mapped_answer = mapping_lang[model_answer]
        correct_answer = ground_truth
        correctness[eid] = 1 if mapped_answer == correct_answer else 0
    return correctness


def compute_correctness_llm_judge(lang, answers_lang):
    yes_no_map = {
        "en": {"YES": 1, "NO": 0},
        "fr": {"YES": 1, "NO": 0},
        "es": {"YES": 1, "NO": 0},
        "ru": {"YES": 1, "NO": 0},
        "pl": {"YES": 1, "NO": 0},
        "ja": {"YES": 1, "NO": 0},
    }
    lang_map = yes_no_map.get(lang, {})
    correctness = {}
    for eid, jd in answers_lang.items():
        normalized = jd.strip().upper()
        correctness[eid] = lang_map.get(normalized, 0)
    return correctness

def prepare_data(dataset, languages, hidden_states, correctness, example_ids):
    """
    Prepares data by extracting the hidden states
    and correctness labels for the specified languages and example IDs.

    Args:
        languages (list of str):
            Language codes to process (e.g., ['en', 'es', 'de']).
        hidden_states (dict of str -> dict):
            Mapping from language to a dict of example IDs → torch.Tensor
            of shape (num_layers, seq_len, hidden_dim).
        correctness (dict of str -> dict):
            Mapping from language to a dict of example IDs → binary label (0 or 1).
        example_ids (set of str):
            Example IDs to include.

    Returns:
        X (list of np.ndarray):
            List of num_train_examples, each element is a NumPy array of shape (num_layers, hidden_dim),
            containing the hidden state of the last token.
        y (list of int):
            Corresponding binary labels.
    """
    X, y = [], []
    print("\nin prepare_data")
    for lang in languages:
        for eid in example_ids:
            try:
                if dataset == 'mkqa':
                    hs = hidden_states[lang][int(eid)].cpu().float()
                else:
                    lookup_key = eid.replace("/", "_")
                    hs = hidden_states[lang][lookup_key].cpu().float()
            except KeyError:
                print(f"{lang}: {eid} not found")
                continue
            X.append(hs.numpy())
            try:
                corr = correctness[lang][str(eid)]
            except KeyError:
                print(f"{lang}: {eid} correctness not found")
                corr = -1
            y.append(corr)
    return X, y

def load_batches(X: torch.Tensor, y: torch.Tensor, batch_size: int, seed: int):
    dataset = TensorDataset(X, y)
    # make a generator for reproducible shuffling
    g = torch.Generator().manual_seed(seed)
    # simple DataLoader that shuffles, preserving class frequencies
    # each example is seen exactly once per epoch, in a random order
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g)


def load_stratified_batches(X: torch.Tensor, y: torch.Tensor, batch_size: int, seed: int):
    """
    Returns a DataLoader whose batches each mirror the class distribution of y,
    and which covers every example exactly once per epoch.
    """
    # ensure labels are a 1D numpy array on CPU
    y_np = y.detach().cpu().numpy().astype(int)

    class StratifiedBatchSampler(Sampler):
        def __init__(self, labels, batch_size, seed, shuffle=True):
            self.labels = np.array(labels)
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.seed = seed
            self.epoch = 0
            # number of mini‑batches per epoch
            self.n_splits = math.ceil(len(self.labels) / self.batch_size)
            self._prepare_folds()

        def _prepare_folds(self):
            skf = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.seed + self.epoch)
            # take the "test" indices of each split as one batch
            self.batches = [test_idx.tolist() for _, test_idx in skf.split(self.labels, self.labels)]

        def __iter__(self):
            if self.shuffle:
                # re-split for fresh randomness each epoch
                self._prepare_folds()
                self.epoch += 1
            for batch in self.batches:
                yield batch

        def __len__(self):
            return self.n_splits

    # wrap features & labels into a TensorDataset
    dataset = TensorDataset(X, y)

    # instantiate the sampler & loader
    sampler = StratifiedBatchSampler(labels=y_np, batch_size=batch_size, shuffle=True, seed=seed)
    return DataLoader(dataset, batch_sampler=sampler)


def find_best_threshold(y_true: np.ndarray, y_probs: np.ndarray):
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    # compute F1 per threshold (ignore last point which has no threshold)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.nanargmax(f1_scores[:-1])
    return thresholds[best_idx], f1_scores[best_idx]

# for neuron_probe
def load_and_preprocess_flattened(datasets: list, results_root: Path, train_languages: list):
    
    # load query hidden states per language for all datasets
    # keep only the hidden state of the last token
    query_hidden_states_dict = defaultdict(dict)
    for ds in datasets:
        results_dir_ds = results_root / ds
        for lang in train_languages:
            # load hidden states
            hidden_states_lang_ds = load_hidden_states(lang, results_dir_ds)
            query_hidden_states_lang_ds = {}
            # extract query hidden states
            for eid, hs_dict in hidden_states_lang_ds.items():
                query_hidden_states_lang_ds[eid] = hs_dict['query_hidden_states'][:, -1, :]    
            query_hidden_states_dict[lang].update(query_hidden_states_lang_ds)
    
    # load correctness dict per language for all datasets
    correctness_dict = defaultdict(dict)
    for ds in datasets:
        for lang in train_languages:
            corr_path = results_root / f"{ds}/{lang}_correctness.jsonl"
            with corr_path.open("r", encoding="utf-8") as fin:
                line = fin.readline().strip()
                correctness_lang_ds = json.loads(line)
            correctness_dict[lang].update(correctness_lang_ds)
    
    # TODO: update logic when multiple training languages are used
    lang = train_languages[0]
    # for each dataset, load training and validation splits
    train_ids = set()
    val_ids = set()
    for ds in datasets:
        out_path = Path(results_root) / f"{ds}/train_lang_{lang}_splits.json"
        with out_path.open("r", encoding="utf-8") as f:
            splits_dict = json.load(f)
        train_ids_list = splits_dict["train"] 
        val_ids_list = splits_dict["val"]
        # add to sets
        train_ids.update(train_ids_list)
        val_ids.update(val_ids_list)

    # prepare data
    X_train, y_train = prepare_data(train_languages, query_hidden_states_dict, correctness_dict, train_ids)
    X_val, y_val = prepare_data(train_languages, query_hidden_states_dict, correctness_dict, val_ids)

    # stack and flatten
    X_train_stack = np.stack(X_train, axis=0)
    X_val_stack = np.stack(X_val, axis=0)
    X_train_flat = X_train_stack.reshape(X_train_stack.shape[0], -1)
    X_val_flat = X_val_stack.reshape(X_val_stack.shape[0], -1)

    # scale flattened data
    scaler = StandardScaler().fit(X_train_flat) #TODO: save scaler results
    X_train_flat_scaled = scaler.transform(X_train_flat)
    X_val_flat_scaled = scaler.transform(X_val_flat)

    return X_train_flat_scaled, y_train, X_val_flat_scaled, y_val

# for layer probe
def load_and_preprocess_unflattened(datasets: list, results_root: Path, train_languages: list, query: bool, stratified=True, last=True, timesensitive=True):
    
    # load hidden states per language for all datasets
    # keep only the hidden state of the last prompt token
    hidden_states_dict = defaultdict(dict)
    if query: # if loading the query hidden states
        for ds in datasets:
            results_dir_ds = results_root / ds
            for lang in train_languages:
                # load hidden states
                hidden_states_lang_ds = load_hidden_states(lang, results_dir_ds)
                query_hidden_states_lang_ds = {}
                # extract query hidden states
                for eid, hs_dict in hidden_states_lang_ds.items():
                    if last:
                        query_hidden_states_lang_ds[eid] = hs_dict['query_hidden_states'][:, -1, :]
                    else:
                        print("missing code in load_and_preprocess_unflattened") #TODO: fill with mean pooling
                        sys.exit()    
                hidden_states_dict[lang].update(query_hidden_states_lang_ds)
    else: # if loading the answer hidden states
        for ds in datasets:
            results_dir_ds = results_root / ds
            for lang in train_languages:
                # load hidden states
                hidden_states_lang_ds = load_hidden_states(lang, results_dir_ds)
                answer_hidden_states_lang_ds = {}
                # extract query hidden states
                for eid, hs_dict in hidden_states_lang_ds.items():
                    if last:
                        answer_hidden_states_lang_ds[eid] = hs_dict['gen_hidden_states'][:, -1, :]
                    else:
                        print("missing code in load_and_preprocess_unflattened") #TODO: fill with mean pooling
                        sys.exit()    
                hidden_states_dict[lang].update(answer_hidden_states_lang_ds)
    
    # load correctness dict per language for all datasets
    correctness_dict = defaultdict(dict)
    for ds in datasets:
        results_dir_ds = results_root / ds
        for lang in train_languages:
            corr_path = results_root / f"{ds}/{lang}_correctness.jsonl"
            with corr_path.open("r", encoding="utf-8") as fin:
                line = fin.readline().strip()
                correctness_lang_ds = json.loads(line)
            correctness_dict[lang].update(correctness_lang_ds)
    
    # TODO: update logic when multiple training languages are used
    lang = train_languages[0]
    # for each dataset, load training and validation splits
    train_ids = set()
    val_ids = set()
    for ds in datasets:
        if stratified:
            if timesensitive:
                out_path = Path(results_root) / f"{ds}/train_lang_{lang}_splits_stratified_nottimesensitive.json"
            else:
                out_path = Path(results_root) / f"{ds}/train_lang_{lang}_splits_stratified.json"
        else:
            out_path = Path(results_root) / f"{ds}/train_lang_{lang}_splits.json"
        with out_path.open("r", encoding="utf-8") as f:
            splits_dict = json.load(f)
        train_ids_list = splits_dict["train"] 
        val_ids_list = splits_dict["val"]
        # add to sets
        train_ids.update(train_ids_list)
        val_ids.update(val_ids_list)

    # prepare data
    X_train, y_train = prepare_data(train_languages, hidden_states_dict, correctness_dict, train_ids)
    X_val, y_val = prepare_data(train_languages, hidden_states_dict, correctness_dict, val_ids)

    # stack
    X_train_stack = np.stack(X_train, axis=0)
    X_val_stack = np.stack(X_val, axis=0)
    
    # fit per-layer scalers
    n_examples, n_layers, n_hidden = X_train_stack.shape
    scalers = [StandardScaler().fit(X_train_stack[:, i, :]) for i in range(n_layers)]
    # save the scaler parameters
    train_langs_str = "_".join(sorted(train_languages))
    datasets_str = "_".join(sorted(datasets))

    sp = f"{train_langs_str}_{datasets_str}_"
    if query:
        sp = sp + "query_"
    else:
        sp = sp + "answer_"
    if last:
        sp = sp + "last_"
    else:
        sp = sp + "mean_"
    if stratified:
        if timesensitive:
            sp = sp + "stratified_nottimesensitive"
        else:
            sp = sp + "stratified_"
    else:
        sp = sp + "balanced_"
    sp = sp + "layer_scalers.joblib"
    scaler_path = Path(results_root) / sp
    joblib.dump(scalers, scaler_path, compress=("gzip", 3))
    # scale the non-flattened representations of train and validation sets
    X_train_scaled = np.stack([scalers[i].transform(X_train_stack[:, i, :]) for i in range(n_layers)], axis=1)
    X_val_scaled = np.stack([scalers[i].transform(X_val_stack[:, i, :]) for i in range(n_layers)], axis=1)
    return X_train_scaled, y_train, X_val_scaled, y_val
    
    # # fit across dimensions scaler 
    # X_train_flat = X_train_stack.reshape(X_train_stack.shape[0], -1)
    # scaler = StandardScaler().fit(X_train_flat) 

    # # save the scaler parameters
    # train_langs_str = "_".join(sorted(train_languages))
    # datasets_str = "_".join(sorted(datasets))
    # if last and stratified:
    #     scaler_path = Path(results_root) / f"{train_langs_str}_{datasets_str}_last_stratified_dim_scaler.joblib"
    # elif last and not stratified:
    #     scaler_path = Path(results_root) / f"{train_langs_str}_{datasets_str}_last_balanced_dim_scaler.joblib"
    # elif not last and stratified:
    #     scaler_path = Path(results_root) / f"{train_langs_str}_{datasets_str}_mean_stratified_dim_scaler.joblib"
    # else:
    #     scaler_path = Path(results_root) / f"{train_langs_str}_{datasets_str}_mean_balanced_dim_scaler.joblib"
    # joblib.dump(scaler, scaler_path)

    # # scale the non-flattened representations
    # n_train, num_layers, hidden_dim = X_train_stack.shape
    # X_train_stack_2d = X_train_stack.reshape(n_train, num_layers * hidden_dim)
    # X_train_stack_2d_scaled = scaler.transform(X_train_stack_2d)
    # X_train_stack_scaled = X_train_stack_2d_scaled.reshape(n_train, num_layers, hidden_dim) # shape: (n_samples, num_layers, hidden_dim) 
    # n_val, num_layers, hidden_dim = X_val_stack.shape
    # X_val_stack_2d = X_val_stack.reshape(n_val, num_layers * hidden_dim)
    # X_val_stack_2d_scaled = scaler.transform(X_val_stack_2d)
    # X_val_stack_scaled = X_val_stack_2d_scaled.reshape(n_val, num_layers, hidden_dim) 

    # return X_train_stack_scaled, y_train, X_val_stack_scaled, y_val
        

# for layer probe
def load_and_preprocess_unflattened_from_hdf5(dataset: str, datasplits: list, results_root: Path, 
                                              train_languages: list, query: bool, epochs="htune", stratified=True, exclude_time_sensitive=True):

    # load last token prompt/answer hidden states per language for all datasets
    if query: 
        hdf5_query_path = results_root / "query_last_token_hidden_states.h5"
        hidden_states_dict = load_from_hdf5(dataset, hdf5_query_path, train_languages)
    else: # if loading the answer hidden states
        hdf5_answer_path = results_root / "answer_last_token_hidden_states.h5"
        hidden_states_dict = load_from_hdf5(dataset, hdf5_answer_path, train_languages)
    
    # helper to build the splits file path for a given datasplit
    lang = train_languages[0]
    def _splits_path(ds):
        if stratified:
            if exclude_time_sensitive:
                return Path(results_root) / f"{ds}/train_lang_{lang}_splits_stratified_without_time_sensitive.json"
            else:
                return Path(results_root) / f"{ds}/train_lang_{lang}_splits_stratified.json"
        else:
            return Path(results_root) / f"{ds}/train_lang_{lang}_splits.json"

    # filter out datasplits that don't have a splits file
    valid_datasplits = []
    for ds in datasplits:
        if _splits_path(ds).exists():
            valid_datasplits.append(ds)
        else:
            print(f"  [SKIP] Splits file not found for {ds}: {_splits_path(ds)}")
    
    # load correctness dict per language for all datasets
    correctness_dict = defaultdict(dict)
    for ds in valid_datasplits:
        for lang_corr in train_languages:
            corr_path = results_root / f"{ds}/{lang_corr}_correctness.jsonl"
            with corr_path.open("r", encoding="utf-8") as fin:
                line = fin.readline().strip()
                correctness_lang_ds = json.loads(line)
            correctness_dict[lang_corr].update(correctness_lang_ds)
    
    # for each dataset, load training and validation splits
    train_ids = set()
    val_ids = set()
    for ds in valid_datasplits:
        out_path = _splits_path(ds)
        with out_path.open("r", encoding="utf-8") as f:
            splits_dict = json.load(f)
        train_ids_list = splits_dict["train"] 
        val_ids_list = splits_dict["val"]
        # add to sets
        train_ids.update(train_ids_list)
        val_ids.update(val_ids_list)

    # prepare data
    X_train, y_train = prepare_data(dataset, train_languages, hidden_states_dict, correctness_dict, train_ids)
    X_val, y_val = prepare_data(dataset, train_languages, hidden_states_dict, correctness_dict, val_ids)

    # stack
    X_train_stack = np.stack(X_train, axis=0)
    X_val_stack = np.stack(X_val, axis=0)
    
    # fit per-layer scalers
    n_examples, n_layers, n_hidden = X_train_stack.shape
    scalers = [StandardScaler().fit(X_train_stack[:, i, :]) for i in range(n_layers)]
    # save the scaler parameters
    train_langs_str = "_".join(sorted(train_languages))
    datasets_str = "full"

    sp = f"{train_langs_str}_{datasets_str}_"
    if query:
        sp = sp + "query_"
    else:
        sp = sp + "answer_"

    # TODO: to extend if other tokens other than the last token are used in the future
    sp = sp + "last_"

    if stratified:
        if exclude_time_sensitive:
            sp = sp + "stratified_without_time_sensitive_"
        else:
            sp = sp + "stratified_"
    else:
        sp = sp + "balanced_"
    sp = sp + f"layer_scalers_{epochs}.joblib"
    scaler_path = Path(results_root) / sp
    joblib.dump(scalers, scaler_path, compress=("gzip", 3))
    # scale the non-flattened representations of train and validation sets
    X_train_scaled = np.stack([scalers[i].transform(X_train_stack[:, i, :]) for i in range(n_layers)], axis=1)
    X_val_scaled = np.stack([scalers[i].transform(X_val_stack[:, i, :]) for i in range(n_layers)], axis=1)
    return X_train_scaled, y_train, X_val_scaled, y_val


def load_test_unflattened_from_hdf5(dataset: str, datasplits: list, results_root: Path, train_langs: list, test_lang: str, query: bool, epochs: str,
                                    stratified=True, exclude_time_sensitive=True):

    train_test_langs = train_langs + [test_lang]
    if query:
        HDF5_PATH = Path(results_root) / "query_last_token_hidden_states.h5"
    else:
        HDF5_PATH = Path(results_root) / "answer_last_token_hidden_states.h5"
    hidden_states_dict = load_from_hdf5(dataset, HDF5_PATH, langs=train_test_langs)
    
    # helper to build the splits file path for a given datasplit
    lang = train_langs[0]
    def _splits_path(ds):
        if stratified:
            if exclude_time_sensitive:
                return Path(results_root) / f"{ds}/train_lang_{lang}_splits_stratified_without_time_sensitive.json"
            else:
                return Path(results_root) / f"{ds}/train_lang_{lang}_splits_stratified.json"
        else:
            return Path(results_root) / f"{ds}/train_lang_{lang}_splits.json"

    # filter out datasplits that don't have a splits file
    valid_datasplits = []
    for ds in datasplits:
        if _splits_path(ds).exists():
            valid_datasplits.append(ds)
        else:
            print(f"  [SKIP] Splits file not found for {ds}: {_splits_path(ds)}")

    # load correctness dict per train + test language for all datasets
    correctness_dict = defaultdict(dict)
    for ds in valid_datasplits:
        for lang_corr in train_test_langs:
            corr_path = results_root / f"{ds}/{lang_corr}_correctness.jsonl"
            with corr_path.open("r", encoding="utf-8") as fin:
                line = fin.readline().strip()
                correctness_lang_ds = json.loads(line)
            correctness_dict[lang_corr].update(correctness_lang_ds)
    
    # for each dataset, load test split
    test_ids = set()
    for ds in valid_datasplits:
        out_path = _splits_path(ds)
        with out_path.open("r", encoding="utf-8") as f:
            splits_dict = json.load(f)
        test_ids_list = splits_dict["test"] 
        test_ids.update(test_ids_list)

    # prepare data
    print(f"Test ids: {len(test_ids_list)}")
    print(f"Size correctness dict: {len(correctness_dict[test_lang])} - key type: {type(next(iter(correctness_dict[test_lang])))}")
    X_test, y_test = prepare_data(dataset, [test_lang], hidden_states_dict, correctness_dict, test_ids)
    X_test_stack = np.stack(X_test, axis=0)

    # per layer scaling based on training data
    train_langs_str = "_".join(sorted(train_langs))
    datasets_str = "full"
    
    sp = f"{train_langs_str}_{datasets_str}_"
    if query:
        sp = sp + "query_"
    else:
        sp = sp + "answer_"
    sp = sp + "last_"
    if stratified:
        if exclude_time_sensitive:
            sp = sp + "stratified_without_time_sensitive_"
        else:
            sp = sp + "stratified_"
    else:
        sp = sp + "balanced_"
    sp = sp + f"layer_scalers_{epochs}.joblib"
    scaler_path = Path(results_root) / sp
    scalers = joblib.load(scaler_path)
    # scale the test data
    n_examples, n_layers, n_hidden = X_test_stack.shape
    X_test_scaled = np.stack([scalers[i].transform(X_test_stack[:, i, :]) for i in range(n_layers)], axis=1,)

    return X_test_scaled, y_test

# for leave out ablation
def load_and_preprocess_unflattened_leave_out_ablation(datasets: list, results_root: Path, train_languages: list, ablation_layers_idx: list, stratified=True, last=True, timesensitive=True):
    
    # load query hidden states per language for all datasets
    # keep only the hidden state of the last prompt token
    query_hidden_states_dict = defaultdict(dict)
    for ds in datasets:
        results_dir_ds = results_root / ds
        for lang in train_languages:
            # load hidden states
            hidden_states_lang_ds = load_hidden_states(lang, results_dir_ds)
            query_hidden_states_lang_ds = {}
            # extract query hidden states
            for eid, hs_dict in hidden_states_lang_ds.items():
                if last:
                    query_hidden_states_lang_ds[eid] = hs_dict['query_hidden_states'][:, -1, :]
                else:
                    print("missing code in load_and_preprocess_unflattened") #TODO: fill with mean pooling
                    sys.exit()    
            query_hidden_states_dict[lang].update(query_hidden_states_lang_ds)
    
    # load correctness dict per language for all datasets
    correctness_dict = defaultdict(dict)
    for ds in datasets:
        results_dir_ds = results_root / ds
        for lang in train_languages:
            corr_path = results_root / f"{ds}/{lang}_correctness.jsonl"
            with corr_path.open("r", encoding="utf-8") as fin:
                line = fin.readline().strip()
                correctness_lang_ds = json.loads(line)
            correctness_dict[lang].update(correctness_lang_ds)
    
    # TODO: update logic when multiple training languages are used
    lang = train_languages[0]
    # for each dataset, load training and validation splits
    train_ids = set()
    val_ids = set()
    for ds in datasets:
        if stratified:
            if timesensitive:
                out_path = Path(results_root) / f"{ds}/train_lang_{lang}_splits_stratified_nottimesensitive.json"
            else:
                out_path = Path(results_root) / f"{ds}/train_lang_{lang}_splits_stratified.json"
        else:
            out_path = Path(results_root) / f"{ds}/train_lang_{lang}_splits.json"
        with out_path.open("r", encoding="utf-8") as f:
            splits_dict = json.load(f)
        train_ids_list = splits_dict["train"] 
        val_ids_list = splits_dict["val"]
        # add to sets
        train_ids.update(train_ids_list)
        val_ids.update(val_ids_list)

    # prepare data
    X_train, y_train = prepare_data(train_languages, query_hidden_states_dict, correctness_dict, train_ids)
    X_val, y_val = prepare_data(train_languages, query_hidden_states_dict, correctness_dict, val_ids)

    # stack
    X_train_stack = np.stack(X_train, axis=0) # shape: (num_examples, num_layers, num_hidden_dim)
    X_val_stack = np.stack(X_val, axis=0) # shape: (num_examples, num_layers, num_hidden_dim)
    print(X_train_stack.shape)
    print(X_val_stack.shape)

    # remove the layers that need to be ablated
    # sort indices descending so deletions don't shift remaining positions
    for idx in sorted(ablation_layers_idx, reverse=True):
        X_train_stack = np.delete(X_train_stack, idx, axis=1)
        X_val_stack = np.delete(X_val_stack, idx, axis=1)
    print(X_train_stack.shape)
    print(X_val_stack.shape)

    # fit per-layer scalers
    n_examples, n_layers, n_hidden = X_train_stack.shape
    scalers = [StandardScaler().fit(X_train_stack[:, i, :]) for i in range(n_layers)]

    # save the scaler parameters
    train_langs_str = "_".join(sorted(train_languages))
    datasets_str = "_".join(sorted(datasets))
    layers_str = "_".join(str(i) for i in ablation_layers_idx)
    if last and stratified:
        if timesensitive:
            scaler_path = Path(results_root) / f"{train_langs_str}_{datasets_str}_last_stratified_nottimesensitive_layer_scalers_ablation_layers_{layers_str}.joblib"
        else:
            scaler_path = Path(results_root) / f"{train_langs_str}_{datasets_str}_last_stratified_layer_scalers_ablation_layers_{layers_str}.joblib"
    elif last and not stratified:
        scaler_path = Path(results_root) / f"{train_langs_str}_{datasets_str}_last_balanced_layer_scalers_ablation_layers_{layers_str}.joblib"
    elif not last and stratified:
        if timesensitive:
            scaler_path = Path(results_root) / f"{train_langs_str}_{datasets_str}_mean_stratified_nottimesensitive_layer_scalers_ablation_layers_{layers_str}.joblib"
        else:
            scaler_path = Path(results_root) / f"{train_langs_str}_{datasets_str}_mean_stratified_layer_scalers_ablation_layers_{layers_str}.joblib"
    else:
        scaler_path = Path(results_root) / f"{train_langs_str}_{datasets_str}_mean_balanced_layer_scalers_ablation_layers_{layers_str}.joblib"
    joblib.dump(scalers, scaler_path, compress=("gzip", 3))

    # scale the non-flattened representations of train and validation sets
    X_train_scaled = np.stack([scalers[i].transform(X_train_stack[:, i, :]) for i in range(n_layers)], axis=1)
    X_val_scaled = np.stack([scalers[i].transform(X_val_stack[:, i, :]) for i in range(n_layers)], axis=1)

    return X_train_scaled, y_train, X_val_scaled, y_val


def load_test_unflattened_leave_out_ablation(datasets: list, results_root: Path, train_languages: list, test_lang: str, ablation_layers_idx: list, stratified=True, last=True, timesensitive=True):

    train_test_langs = train_languages + [test_lang]

    # load query hidden states per train + test language for all datasets
    # keep only the hidden state of the last token
    query_hidden_states_dict = defaultdict(dict)
    for ds in datasets:
        results_dir_ds = results_root / ds
        for lang in train_test_langs:
            # load hidden states
            hidden_states_lang_ds = load_hidden_states(lang, results_dir_ds)
            query_hidden_states_lang_ds = {}
            # extract query hidden states
            for eid, hs_dict in hidden_states_lang_ds.items():
                if last:
                    query_hidden_states_lang_ds[eid] = hs_dict['query_hidden_states'][:, -1, :]
                else:
                    print("TOADD CODE FOR MEAN POOLING") # TODO
                    sys.exit()    
            query_hidden_states_dict[lang].update(query_hidden_states_lang_ds)
    
    # load correctness dict per train + test language for all datasets
    correctness_dict = defaultdict(dict)
    for ds in datasets:
        results_dir_ds = results_root / ds
        for lang in train_test_langs:
            corr_path = results_root / f"{ds}/{lang}_correctness.jsonl"
            with corr_path.open("r", encoding="utf-8") as fin:
                line = fin.readline().strip()
                correctness_lang_ds = json.loads(line)
            correctness_dict[lang].update(correctness_lang_ds)
    
    # TODO: update logic when multiple training languages are used
    lang = train_languages[0]
    # for each dataset, load test split
    test_ids = set()
    for ds in datasets:
        if stratified:
            if timesensitive:
                out_path = Path(results_root) / f"{ds}/train_lang_{lang}_splits_stratified_nottimesensitive.json"
            else:
                out_path = Path(results_root) / f"{ds}/train_lang_{lang}_splits_stratified.json"
        else:
            out_path = Path(results_root) / f"{ds}/train_lang_{lang}_splits.json"
        with out_path.open("r", encoding="utf-8") as f:
            splits_dict = json.load(f)
        test_ids_list = splits_dict["test"] 
        test_ids.update(test_ids_list)

    # prepare data
    X_test, y_test = prepare_data([test_lang], query_hidden_states_dict, correctness_dict, test_ids)
    X_test_stack = np.stack(X_test, axis=0)
    print(X_test_stack.shape)

    # remove the layers that need to be ablated
    # sort indices descending so deletions don't shift remaining positions
    for idx in sorted(ablation_layers_idx, reverse=True):
        X_test_stack = np.delete(X_test_stack, idx, axis=1)
    print(X_test_stack.shape)

    # per layer scaling based on training data
    train_langs_str = "_".join(sorted(train_languages))
    datasets_str = "_".join(sorted(datasets))
    layers_str = "_".join(str(i) for i in ablation_layers_idx)
    if last and stratified:
        if timesensitive:
            scaler_path = Path(results_root) / f"{train_langs_str}_{datasets_str}_last_stratified_nottimesensitive_layer_scalers_ablation_layers_{layers_str}.joblib"
        else:
            scaler_path = Path(results_root) / f"{train_langs_str}_{datasets_str}_last_stratified_layer_scalers_ablation_layers_{layers_str}.joblib"
    elif last and not stratified:
        scaler_path = Path(results_root) / f"{train_langs_str}_{datasets_str}_last_balanced_layer_scalers_ablation_layers_{layers_str}.joblib"
    elif not last and stratified:
        if timesensitive:
            scaler_path = Path(results_root) / f"{train_langs_str}_{datasets_str}_mean_stratified_nottimesensitive_layer_scalers_ablation_layers_{layers_str}.joblib"
        else:
            scaler_path = Path(results_root) / f"{train_langs_str}_{datasets_str}_mean_stratified_layer_scalers_ablation_layers_{layers_str}.joblib"
    else:
        scaler_path = Path(results_root) / f"{train_langs_str}_{datasets_str}_mean_balanced_layer_scalers_ablation_layers_{layers_str}.joblib"
    scalers = joblib.load(scaler_path)

    # scale the test data
    n_examples, n_layers, n_hidden = X_test_stack.shape
    X_test_scaled = np.stack([scalers[i].transform(X_test_stack[:, i, :]) for i in range(n_layers)], axis=1,)

    return X_test_scaled, y_test


def load_and_preprocess_unflattened_random(datasets: list, results_root: Path, train_languages: list, stratified=True, last=True, timesensitive=True):
    
    # load query hidden states per language for all datasets
    # keep only the hidden state of the last prompt token
    query_hidden_states_dict = defaultdict(dict)
    for ds in datasets:
        results_dir_ds = results_root / ds
        for lang in train_languages:
            # load hidden states
            hidden_states_lang_ds = load_hidden_states(lang, results_dir_ds)
            query_hidden_states_lang_ds = {}
            # extract query hidden states
            for eid, hs_dict in hidden_states_lang_ds.items():
                if last:
                    query_hidden_states_lang_ds[eid] = hs_dict['query_hidden_states'][:, -1, :]
                else:
                    print("missing code in load_and_preprocess_unflattened") #TODO: fill with mean pooling
                    sys.exit()    
            query_hidden_states_dict[lang].update(query_hidden_states_lang_ds)
    
    # load correctness dict per language for all datasets
    correctness_dict = defaultdict(dict)
    for ds in datasets:
        results_dir_ds = results_root / ds
        for lang in train_languages:
            corr_path = results_root / f"{ds}/{lang}_correctness.jsonl"
            with corr_path.open("r", encoding="utf-8") as fin:
                line = fin.readline().strip()
                correctness_lang_ds = json.loads(line)
            correctness_dict[lang].update(correctness_lang_ds)
    
    # TODO: update logic when multiple training languages are used
    lang = train_languages[0]
    # for each dataset, load training and validation splits
    train_ids = set()
    val_ids = set()
    for ds in datasets:
        if stratified:
            if timesensitive:
                out_path = Path(results_root) / f"{ds}/train_lang_{lang}_splits_stratified_nottimesensitive.json"
            else:
                out_path = Path(results_root) / f"{ds}/train_lang_{lang}_splits_stratified.json"
        else:
            out_path = Path(results_root) / f"{ds}/train_lang_{lang}_splits.json"
        with out_path.open("r", encoding="utf-8") as f:
            splits_dict = json.load(f)
        train_ids_list = splits_dict["train"] 
        val_ids_list = splits_dict["val"]
        # add to sets
        train_ids.update(train_ids_list)
        val_ids.update(val_ids_list)

    # prepare data
    X_train, y_train = prepare_data(train_languages, query_hidden_states_dict, correctness_dict, train_ids)
    X_val, y_val = prepare_data(train_languages, query_hidden_states_dict, correctness_dict, val_ids)

    # stack
    X_train_stack = np.stack(X_train, axis=0)
    X_val_stack = np.stack(X_val, axis=0)

    # randomize training data
    rng = np.random.default_rng(42)
    X_rand = rng.random(X_train_stack.shape, dtype=X_train_stack.dtype)
    X_train_stack = X_rand

    # flatten and scale train features
    X_train_flat = X_train_stack.reshape(X_train_stack.shape[0], -1)
    scaler = StandardScaler().fit(X_train_flat) 

    # save the scaler parameters
    train_langs_str = "".join(train_languages)
    datasets_str = "".join(datasets)
    if last and stratified:
        if timesensitive:
            scaler_path = Path(results_root) / f"{train_langs_str}_{datasets_str}_last_stratified_nottimesensitive_scaler_random.joblib"
        else:
            scaler_path = Path(results_root) / f"{train_langs_str}_{datasets_str}_last_stratified_scaler_random.joblib"
    elif last and not stratified:
        scaler_path = Path(results_root) / f"{train_langs_str}_{datasets_str}_last_balanced_scaler_random.joblib"
    elif not last and stratified:
        if timesensitive:
            scaler_path = Path(results_root) / f"{train_langs_str}_{datasets_str}_mean_stratified_nottimesensitive_scaler_random.joblib"
        else:
            scaler_path = Path(results_root) / f"{train_langs_str}_{datasets_str}_mean_stratified_scaler_random.joblib"
    else:
        scaler_path = Path(results_root) / f"{train_langs_str}_{datasets_str}_mean_balanced_scaler_random.joblib"
    print(scaler_path)
    joblib.dump(scaler, scaler_path)

    # scale the non-flattened representations
    n_train, num_layers, hidden_dim = X_train_stack.shape
    X_train_stack_2d = X_train_stack.reshape(n_train, num_layers * hidden_dim)
    X_train_stack_2d_scaled = scaler.transform(X_train_stack_2d)
    X_train_stack_scaled = X_train_stack_2d_scaled.reshape(n_train, num_layers, hidden_dim) # shape: (n_samples, num_layers, hidden_dim) 
    n_val, num_layers, hidden_dim = X_val_stack.shape
    X_val_stack_2d = X_val_stack.reshape(n_val, num_layers * hidden_dim)
    X_val_stack_2d_scaled = scaler.transform(X_val_stack_2d)
    X_val_stack_scaled = X_val_stack_2d_scaled.reshape(n_val, num_layers, hidden_dim) 

    return X_train_stack_scaled, y_train, X_val_stack_scaled, y_val


def load_test_unflattened_random(datasets: list, results_root: Path, train_languages: list, test_lang: str, stratified=True, last=True, timesensitive=True):

    train_test_langs = train_languages + [test_lang]

    # load query hidden states per train + test language for all datasets
    # keep only the hidden state of the last token
    query_hidden_states_dict = defaultdict(dict)
    for ds in datasets:
        results_dir_ds = results_root / ds
        for lang in train_test_langs:
            # load hidden states
            hidden_states_lang_ds = load_hidden_states(lang, results_dir_ds)
            query_hidden_states_lang_ds = {}
            # extract query hidden states
            for eid, hs_dict in hidden_states_lang_ds.items():
                if last:
                    query_hidden_states_lang_ds[eid] = hs_dict['query_hidden_states'][:, -1, :]
                else:
                    print("TOADD CODE FOR MEAN POOLING") # TODO
                    sys.exit()    
            query_hidden_states_dict[lang].update(query_hidden_states_lang_ds)
    
    # load correctness dict per train + test language for all datasets
    correctness_dict = defaultdict(dict)
    for ds in datasets:
        results_dir_ds = results_root / ds
        for lang in train_test_langs:
            corr_path = results_root / f"{ds}/{lang}_correctness.jsonl"
            with corr_path.open("r", encoding="utf-8") as fin:
                line = fin.readline().strip()
                correctness_lang_ds = json.loads(line)
            correctness_dict[lang].update(correctness_lang_ds)
    
    # TODO: update logic when multiple training languages are used
    lang = train_languages[0]
    # for each dataset, load test split
    test_ids = set()
    for ds in datasets:
        if stratified:
            if timesensitive:
                out_path = Path(results_root) / f"{ds}/train_lang_{lang}_splits_stratified_nottimesensitive.json"
            else:
                out_path = Path(results_root) / f"{ds}/train_lang_{lang}_splits_stratified.json"
        else:
            out_path = Path(results_root) / f"{ds}/train_lang_{lang}_splits.json"
        with out_path.open("r", encoding="utf-8") as f:
            splits_dict = json.load(f)
        test_ids_list = splits_dict["test"] 
        test_ids.update(test_ids_list)

    # prepare data
    X_test, y_test = prepare_data([test_lang], query_hidden_states_dict, correctness_dict, test_ids)
    X_test_stack = np.stack(X_test, axis=0)

    # flatten and scale based on train set
    train_langs_str = "".join(train_languages)
    datasets_str = "".join(datasets)
    if last and stratified:
        if timesensitive:
            scaler_path = Path(results_root) / f"{train_langs_str}_{datasets_str}_last_stratified_nottimesensitive_scaler_random.joblib"
        else:
            scaler_path = Path(results_root) / f"{train_langs_str}_{datasets_str}_last_stratified_scaler_random.joblib"
    elif last and not stratified:
        scaler_path = Path(results_root) / f"{train_langs_str}_{datasets_str}_last_balanced_scaler_random.joblib"
    elif not last and stratified:
        if timesensitive:
            scaler_path = Path(results_root) / f"{train_langs_str}_{datasets_str}_mean_stratified_nottimesensitive_scaler_random.joblib"
        else:
            scaler_path = Path(results_root) / f"{train_langs_str}_{datasets_str}_mean_stratified_scaler_random.joblib"
    else:
        scaler_path = Path(results_root) / f"{train_langs_str}_{datasets_str}_mean_balanced_scaler_random.joblib"
    print(scaler_path)
    scaler = joblib.load(scaler_path)

    n_test, num_layers, hidden_dim = X_test_stack.shape
    X_test_stack_2d = X_test_stack.reshape(n_test, num_layers * hidden_dim)
    X_test_stack_2d_scaled = scaler.transform(X_test_stack_2d)
    X_test_stack_scaled = X_test_stack_2d_scaled.reshape(n_test, num_layers, hidden_dim)

    return X_test_stack_scaled, y_test


def load_test_flattened(datasets: list, results_root: Path, train_languages: list, test_lang: str):

    # TODO: remove all processing done for training languages once I have the parameters of the scaler saved

    train_test_langs = train_languages + [test_lang]

    # load query hidden states per train + test language for all datasets
    # keep only the hidden state of the last token
    # TODO: once I save the scaler, remove
    query_hidden_states_dict = defaultdict(dict)
    for ds in datasets:
        results_dir_ds = results_root / ds
        for lang in train_test_langs:
            # load hidden states
            hidden_states_lang_ds = load_hidden_states(lang, results_dir_ds)
            query_hidden_states_lang_ds = {}
            # extract query hidden states
            for eid, hs_dict in hidden_states_lang_ds.items():
                query_hidden_states_lang_ds[eid] = hs_dict['query_hidden_states'][:, -1, :]    
            query_hidden_states_dict[lang].update(query_hidden_states_lang_ds)
        
    
    # load correctness dict per train + test language for all datasets
    correctness_dict = defaultdict(dict)
    for ds in datasets:
        results_dir_ds = results_root / ds
        for lang in train_test_langs:
            corr_path = results_root / f"{ds}/{lang}_correctness.jsonl"
            with corr_path.open("r", encoding="utf-8") as fin:
                line = fin.readline().strip()
                correctness_lang_ds = json.loads(line)
            correctness_dict[lang].update(correctness_lang_ds)
    
    # TODO: update logic when multiple training languages are used
    # TODO: do not rescale, load the saved params 
    # test data taken based on the training language so that it was not seen during training
    lang = train_languages[0]
    # for each dataset, load test split
    train_ids = set()
    test_ids = set()
    for ds in datasets:
        out_path = Path(results_root) / f"{ds}/train_lang_{lang}_splits.json"
        with out_path.open("r", encoding="utf-8") as f:
            splits_dict = json.load(f)
        train_ids_list = splits_dict["train"]
        test_ids_list = splits_dict["test"] 
        # add to sets
        train_ids.update(train_ids_list)
        test_ids.update(test_ids_list)

    # prepare data
    X_train, y_train = prepare_data(train_languages, query_hidden_states_dict, correctness_dict, train_ids)
    X_test, y_test = prepare_data([test_lang], query_hidden_states_dict, correctness_dict, test_ids)
    
    # stack
    X_train_stack = np.stack(X_train, axis=0)
    X_test_stack = np.stack(X_test, axis=0)

    # flatten and scale train features
    X_train_flat = X_train_stack.reshape(X_train_stack.shape[0], -1)
    scaler = StandardScaler().fit(X_train_flat)
    X_test_flat = X_test_stack.reshape(X_test_stack.shape[0], -1)
    X_test_flat_scaled = scaler.transform(X_test_flat) 

    return X_test_flat_scaled, y_test


def evaluate_split(y_true, y_prob, thr):
    """Compute evaluation metrics given true labels and predicted probabilities."""
    y_pred = (y_prob >= thr).astype(int)
    acc = accuracy_score(y_true, y_pred)
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auroc = None
    brier = brier_score_loss(y_true, y_prob)
    ece = compute_ece(y_true, y_prob, n_bins=10)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    aupr = auc(rec, prec)
    return {
        'accuracy': float(acc),
        'auroc': float(auroc) if auroc is not None else None,
        'brier_score': float(brier),
        'ece': float(ece),
        'aupr': float(aupr),
    }


def compute_ece(y_true, y_pred_prob, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_pred_prob >= bin_edges[i]) & (y_pred_prob < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_true = y_true[mask]
            bin_prob = y_pred_prob[mask]
            bin_accuracy = bin_true.mean()
            bin_confidence = bin_prob.mean()
            ece += np.abs(bin_accuracy - bin_confidence) * mask.sum() / len(y_true)
    try:
        ret = ece.item()
    except Exception as e:
        print("Exception in compute_ece:", e)
        ret = ece
    return ret

# helper function to locate subsequence
def find_subsequence(subseq, seq):
    for i in range(len(seq) - len(subseq) + 1):
        if seq[i:i+len(subseq)] == subseq:
            return i, i+len(subseq)
    return -1, -1


# visualisations
# color-blind friendly palette: https://immunobiology.duke.edu/sites/default/files/2023-04/Colorblind-Palette.pdf
LANGUAGE_COLORS = {
    'en': '#332288',    # indigo
    'es': '#44AA99',    # teal
    'fr': '#117733',    # green
    'pl': '#E69F00', 
    # 'pl': '#DDCC77',    # sand
    'ru': '#CC6677',    # rose
    'ja': '#882255',    # wine
}

def compute_l2_norm(hidden_states_lang):
    """Compute L2 norm of hidden states per layer for each question."""
    norms = {}
    for example_id, states in hidden_states_lang.items():
        # states: [num_layers, hidden_dim]
        norms[example_id] = torch.norm(states.to(dtype=torch.float32), p=2, dim=1).numpy()  # [num_layers]
    return norms

def compute_l2_norm_standardised(hidden_states_lang):
    """
    For each layer, standardize that layer's hidden vectors across examples using
    scikit-learn's StandardScaler, then compute per-example, per-layer L2 norms.
    Args:
        hidden_states_lang: dict mapping example_id -> tensor or array of shape [num_layers, hidden_dim]
    Returns:
        norms: dict mapping example_id -> numpy array of shape (num_layers,) with L2 norms
    """
    example_ids = list(hidden_states_lang.keys())
    if not example_ids:
        return {}

    # Stack into (N, L, D)
    states = []
    for eid in example_ids:
        st = hidden_states_lang[eid]
        if isinstance(st, torch.Tensor):
            st = st.detach().cpu().to(dtype=torch.float32).numpy()
        else:
            st = np.asarray(st, dtype=np.float32)
        states.append(st)  # each is [num_layers, hidden_dim]
    data = np.stack(states, axis=0)  # [N, num_layers, hidden_dim]
    N, L, D = data.shape

    norms_matrix = np.zeros((N, L), dtype=float)  # will hold per-example, per-layer norms

    for layer_idx in range(L):
        layer_feats = data[:, layer_idx, :]  # [N, hidden_dim]
        scaler = StandardScaler().fit(layer_feats)
        scaled = scaler.transform(layer_feats)  # [N, hidden_dim]
        # L2 norm over hidden_dim
        norms_layer = np.linalg.norm(scaled, ord=2, axis=1)  # (N,)
        norms_matrix[:, layer_idx] = norms_layer

    # map back to dict
    norms = {
        eid: norms_matrix[idx]
        for idx, eid in enumerate(example_ids)
    }
    return norms


def plot_layerwise_l2_norms_per_language(all_norms, output_dir, num_layers, languages, ylim=160):
    """
    For each language in all_norms, make one plot of layerwise L2 norms
    (mean ± min/max) and save it to output_dir/<lang>_layerwise_l2_norms.png.
    """
    print("\nIn plot_layerwise_l2_norms_per_language")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    layers = np.arange(1, num_layers + 1)
    for lang in languages:
        arr = np.stack([all_norms[lang][eid] for eid in all_norms[lang]])  # shape (n_examples, NUM_LAYERS)

        mean_vals = arr.mean(axis=0)
        min_vals  = arr.min(axis=0)
        max_vals  = arr.max(axis=0)

        fig, ax = plt.subplots(figsize=(10, 5))
        color = LANGUAGE_COLORS[lang]

        # mean line
        ax.plot(layers, mean_vals, marker='o', color=color, linewidth=2, label="mean")

        # shaded min–max band
        ax.fill_between(layers, min_vals, max_vals, color=color, alpha=0.2, label="range (min→max)")

        ax.tick_params(axis="both", labelsize=16)
        ax.set_xticks(layers[::2])
        ax.set_ylim(0, ylim)
        ax.set_xlabel('Layer', fontsize=20)
        ax.set_ylabel('L2 Norm', fontsize=20)
        ax.set_title(f'{lang.lower()}', fontsize=24)
        ax.grid(True)
        ax.legend(fontsize=15)
        plt.grid(True, linestyle='--', alpha=0.5)
        fig.tight_layout()
        out_file = Path(output_dir) / f"{lang.lower()}.png"
        fig.savefig(out_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {out_file}")

def plot_pca(hidden_states, output_dir, num_layers, languages, xlim=(-100, 100), ylim=(-80, 80)):
    print("\nIn plot_pca")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_path = output_dir / "pca_cache.npz"
    if cache_path.exists():
        print(f"Loading PCA cache from {cache_path}")
        cache = np.load(cache_path, allow_pickle=True)
        pca_results = {int(k.split("_")[2]): (cache[k], cache[k.replace("X_pca", "labels")]) for k in cache.files if k.startswith("X_pca_")}
    else:
        print("Computing PCA for all layers and saving cache...")
        pca_results = {}
        for layer_idx in range(num_layers):
            X = []
            labels = []
            for lang in languages:
                ex_ids = list(hidden_states[lang])
                for eid in ex_ids:
                    X.append(hidden_states[lang][eid][layer_idx].to(dtype=torch.float32).numpy())
                    labels.append(lang)
            X = np.stack(X, axis=0)
            labels_arr = np.array(labels)
            X_scaled = StandardScaler().fit_transform(X)
            X_pca = PCA(n_components=2).fit_transform(X_scaled)
            pca_results[layer_idx] = (X_pca, labels_arr)
            print(f"  Layer {layer_idx + 1}/{num_layers} done")
        np.savez(cache_path, **{f"X_pca_{i}": v[0] for i, v in pca_results.items()},
                 **{f"labels_{i}": v[1] for i, v in pca_results.items()})
        print(f"Saved PCA cache to {cache_path}")

    # Split layers into two groups (16 layers each)
    layers_per_figure = 16
    num_figures = (num_layers + layers_per_figure - 1) // layers_per_figure

    for fig_idx in range(num_figures):
        start_layer = fig_idx * layers_per_figure
        end_layer = min((fig_idx + 1) * layers_per_figure, num_layers)
        num_layers_in_fig = end_layer - start_layer

        # create 4x4 grid for up to 16 layers
        rows, cols = 4, 4
        fig, axes = plt.subplots(rows, cols, figsize=(26, 26), sharex=True, sharey=True)
        axes = axes.flatten()

        for i, layer_idx in enumerate(range(start_layer, end_layer)):
            X_pca, labels_arr = pca_results[layer_idx]

            # plot on the corresponding subplot
            ax = axes[i]
            for lang in languages:
                mask = labels_arr == lang
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=LANGUAGE_COLORS.get(lang), label=lang, s=1, alpha=0.5)

            # overlay centroids
            for lang in languages:
                mask = labels_arr == lang
                pts = X_pca[mask]
                if pts.shape[0] == 0:
                    continue
                centroid = pts.mean(axis=0)
                # ax.scatter(centroid[0], centroid[1], marker='*', s=200, facecolor=LANGUAGE_COLORS.get(lang), edgecolor='black', linewidth=1.2, zorder=5)
                ax.text(centroid[0],centroid[1], lang, fontsize=18, ha='center', va='center', color='white', fontweight='bold',
                        path_effects=[pe.withStroke(linewidth=2, foreground='black')], zorder=6)

            ax.set_title(f'Layer {layer_idx + 1}', fontsize=24)
            ax.grid(True)

            # apply limits and aspect
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)

        # hide unused subplots
        for i in range(num_layers_in_fig, len(axes)):
            axes[i].set_visible(False)

        # set shared axis labels
        for ax in axes[-cols:][:cols]:
            ax.set_xlabel('PC1', fontsize=22, labelpad=5)
        for ax in axes[::cols]:
            ax.set_ylabel('PC2', fontsize=22, labelpad=2)
        for ax in axes[:num_layers_in_fig]:
            ax.tick_params(labelsize=20)

        # add legends: languages + centroid
        lang_srt = sorted(languages)
        lang_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=LANGUAGE_COLORS.get(lang), markersize=15, label=lang.lower()) for lang in lang_srt]
        # centroid_handle = plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='gray', markeredgecolor='black', markersize=15, label='centroid')
        # legend_handles = lang_handles + [centroid_handle]

        fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.02)
        fig.legend(handles=lang_handles, title='Languages (dots); centroid shown by language code', loc='upper center', 
                   bbox_to_anchor=(0.5, -0.05), ncol=len(lang_handles), frameon=False,
                   fontsize=20, title_fontsize=22, handletextpad=0.5, columnspacing=1.0, borderaxespad=0.0, markerscale=1)

        lang_str = "_".join(languages)
        output_file = Path(output_dir) / f"pca_{lang_str}_part{fig_idx + 1}.png"
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {output_file}")


def plot_selected_layers_pca(hidden_states, output_dir, total_num_layers, languages, selected_layers=[1, 13, 23, 32], xlim=(-80, 80), ylim=(-80, 80)):
    print("\nIn plot_selected_layers_pca")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Map 1-based layer indices to 0-based (used internally)
    selected_layer_indices = [l - 1 for l in selected_layers if 1 <= l <= total_num_layers]

    cache_path = output_dir / "pca_cache.npz"
    if cache_path.exists():
        print(f"Loading PCA cache from {cache_path}")
        cache = np.load(cache_path, allow_pickle=True)
        pca_results = {int(k.split("_")[2]): (cache[k], cache[k.replace("X_pca", "labels")]) for k in cache.files if k.startswith("X_pca_")}
    else:
        print("Computing PCA for selected layers and saving cache...")
        pca_results = {}
        for layer_idx in selected_layer_indices:
            X = []
            labels = []
            for lang in languages:
                ex_ids = list(hidden_states[lang])
                for eid in ex_ids:
                    X.append(hidden_states[lang][eid][layer_idx].to(dtype=torch.float32).numpy())
                    labels.append(lang)
            X = np.stack(X, axis=0)
            labels_arr = np.array(labels)
            X_scaled = StandardScaler().fit_transform(X)
            X_pca = PCA(n_components=2).fit_transform(X_scaled)
            pca_results[layer_idx] = (X_pca, labels_arr)
            print(f"  Layer {layer_idx + 1} done")
        np.savez(cache_path, **{f"X_pca_{i}": v[0] for i, v in pca_results.items()},
                 **{f"labels_{i}": v[1] for i, v in pca_results.items()})
        print(f"Saved PCA cache to {cache_path}")

    fig, axes = plt.subplots(1, len(selected_layer_indices), figsize=(6 * len(selected_layer_indices), 6), sharex=True, sharey=True)
    if len(selected_layer_indices) == 1:
        axes = [axes]  # ensure it's iterable

    for i, layer_idx in enumerate(selected_layer_indices):
        X_pca, labels_arr = pca_results[layer_idx]

        ax = axes[i]
        for lang in languages:
            mask = labels_arr == lang
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=LANGUAGE_COLORS.get(lang), label=lang, s=1, alpha=0.5)

        for lang in languages:
            mask = labels_arr == lang
            pts = X_pca[mask]
            if pts.shape[0] == 0:
                continue
            centroid = pts.mean(axis=0)
            ax.text(centroid[0], centroid[1], lang, fontsize=21, ha='center', va='center', color='white', fontweight='bold',
                    path_effects=[pe.withStroke(linewidth=2, foreground='black')], zorder=6)

        ax.set_title(f'Layer {layer_idx + 1}', fontsize=29)
        ax.grid(True)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel('PC1', fontsize=27, labelpad=5)
        if i == 0:
            ax.set_ylabel('PC2', fontsize=27, labelpad=2)
        ax.tick_params(labelsize=25)

    lang_srt = sorted(languages)
    lang_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=LANGUAGE_COLORS.get(lang), markersize=15, label=lang.lower()) for lang in lang_srt]

    fig.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.15)
    fig.legend(handles=lang_handles, title='Languages (dots); centroid shown by language code', loc='upper center',
               bbox_to_anchor=(0.5, -0.05), ncol=len(lang_handles), frameon=False,
               fontsize=25, title_fontsize=27, handletextpad=0.5, columnspacing=1.0, borderaxespad=0.0, markerscale=1)

    lang_str = "_".join(languages)
    output_file = Path(output_dir) / f"pca_{lang_str}_selected_layers.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_file}")


CORRECTNESS_COLORS = {
    0: '#FF0000',  # red - incorrect
    1: '#0000FF' # blue - correct
}

def plot_pca_correctness_all_languages(hidden_states, answer_correctness_dict, output_dir, num_layers, languages, xlim=(-80, 80), ylim=(-80, 80)):
    """
    Pooled PCA across ALL languages per layer.
    Points are colored by correctness (0/1).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_path = output_dir / "pca_cache.npz"
    if cache_path.exists():
        print(f"Loading PCA cache from {cache_path}")
        cache = np.load(cache_path, allow_pickle=True)
        pca_results = {int(k.split("_")[2]): (cache[k], cache[k.replace("X_pca", "corr_labels")]) for k in cache.files if k.startswith("X_pca_")}
    else:
        print("Computing PCA for all layers and saving cache...")
        pca_results = {}
        for layer_idx in range(num_layers):
            X, corr_labels = [], []
            for lang in languages:
                ex_ids = list(answer_correctness_dict.get(lang, {}))
                for eid in ex_ids:
                    if lang not in hidden_states or eid not in hidden_states[lang]:
                        continue
                    x = hidden_states[lang][eid][layer_idx].detach().cpu().to(dtype=torch.float32).numpy()
                    X.append(x)
                    corr = answer_correctness_dict[lang][eid]
                    if isinstance(corr, torch.Tensor):
                        corr = corr.item()
                    corr_labels.append(int(corr))
            if len(X) == 0:
                pca_results[layer_idx] = None
                continue
            X = np.stack(X, axis=0)
            corr_labels = np.array(corr_labels, dtype=int)
            X_scaled = StandardScaler().fit_transform(X)
            X_pca = PCA(n_components=2).fit_transform(X_scaled)
            pca_results[layer_idx] = (X_pca, corr_labels)
            print(f"  Layer {layer_idx + 1}/{num_layers} done")
        np.savez(cache_path,
                 **{f"X_pca_{i}": v[0] for i, v in pca_results.items() if v is not None},
                 **{f"corr_labels_{i}": v[1] for i, v in pca_results.items() if v is not None})
        print(f"Saved PCA cache to {cache_path}")

    layers_per_figure = 16
    num_figures = (num_layers + layers_per_figure - 1) // layers_per_figure

    for fig_idx in range(num_figures):
        start_layer = fig_idx * layers_per_figure
        end_layer = min((fig_idx + 1) * layers_per_figure, num_layers)
        num_layers_in_fig = end_layer - start_layer

        rows, cols = 4, 4
        fig, axes = plt.subplots(rows, cols, figsize=(24, 24), sharex=True, sharey=True)
        axes = axes.flatten()

        fig.suptitle("All languages (colored by correctness)", fontsize=18)

        for i, layer_idx in enumerate(range(start_layer, end_layer)):
            if pca_results.get(layer_idx) is None:
                axes[i].set_visible(False)
                continue
            X_pca, corr_labels = pca_results[layer_idx]

            ax = axes[i]
            for corr in (0, 1):
                mask = corr_labels == corr
                ax.scatter(
                    X_pca[mask, 0],
                    X_pca[mask, 1],
                    c=CORRECTNESS_COLORS[corr],
                    s=1,
                    alpha=0.5,
                )

            ax.set_title(f"Layer {layer_idx + 1}", fontsize=18)
            ax.grid(True)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.tick_params(labelsize=14)

        # Hide unused subplots
        for j in range(num_layers_in_fig, len(axes)):
            axes[j].set_visible(False)

        # Axis labels
        for ax in axes[-cols:]:
            ax.set_xlabel("PC1", fontsize=16)
        for ax in axes[::cols]:
            ax.set_ylabel("PC2", fontsize=16)

        # Legend
        corr_handles = [
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=CORRECTNESS_COLORS[0], markersize=12, label='incorrect'),
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=CORRECTNESS_COLORS[1], markersize=12, label='correct'),
        ]

        fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.02)
        fig.legend(
            handles=corr_handles,
            title="Correctness",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.005),
            ncol=2,
            frameon=False,
            fontsize=14,
            title_fontsize=14,
            handletextpad=0.5,
            columnspacing=1.0,
            borderaxespad=0.0,
        )

        lang_str = "_".join(languages)
        output_file = output_dir / f"pca_corr_alllangs_{lang_str}_part{fig_idx + 1}.png"
        fig.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {output_file}")


def plot_pca_correctness(hidden_states, answer_correctness_dict, output_dir, num_layers, lang, xlim=(-80, 80), ylim=(-80, 80)):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    layers_per_figure = 16
    num_figures = (num_layers + layers_per_figure - 1) // layers_per_figure

    for fig_idx in range(num_figures):
        start_layer = fig_idx * layers_per_figure
        end_layer = min((fig_idx + 1) * layers_per_figure, num_layers)
        num_layers_in_fig = end_layer - start_layer

        rows, cols = 4, 4
        fig, axes = plt.subplots(rows, cols, figsize=(24, 24), sharex=True, sharey=True)
        axes = axes.flatten()
        
        fig.suptitle(f'{lang}')

        ex_ids = list(answer_correctness_dict[lang])

        for i, layer_idx in enumerate(range(start_layer, end_layer)):
            X, corr_labels = [], []
            for eid in ex_ids:
                x = hidden_states[lang][eid][layer_idx]
                x = x.detach().cpu().to(dtype=torch.float32).numpy()
                X.append(x)
                corr = answer_correctness_dict[lang][eid]
                if isinstance(corr, torch.Tensor):
                    corr = corr.item()
                corr_labels.append(int(corr))

            X = np.stack(X, axis=0)
            corr_labels = np.array(corr_labels, dtype=int)

            X_scaled = StandardScaler().fit_transform(X)
            X_pca = PCA(n_components=2).fit_transform(X_scaled)

            ax = axes[i]
            for corr in (0, 1):
                mask = corr_labels == corr
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                           c=CORRECTNESS_COLORS[corr], s=1, alpha=0.5)

            ax.set_title(f'Layer {layer_idx + 1}', fontsize=18)
            ax.grid(True)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.tick_params(labelsize=14)

        for j in range(num_layers_in_fig, len(axes)):
            axes[j].set_visible(False)

        for ax in axes[-cols:]:
            ax.set_xlabel('PC1', fontsize=16)
        for ax in axes[::cols]:
            ax.set_ylabel('PC2', fontsize=16)

        corr_handles = [
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=CORRECTNESS_COLORS[0], markersize=12, label='incorrect'),
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=CORRECTNESS_COLORS[1], markersize=12, label='correct'),
        ]

        fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.02)
        fig.legend(handles=corr_handles, title='Correctness',
                   loc='upper center', bbox_to_anchor=(0.5, -0.005),
                   ncol=2, frameon=False, fontsize=14, title_fontsize=14,
                   handletextpad=0.5, columnspacing=1.0, borderaxespad=0.0)

        output_file = output_dir / f"pca_corr_{lang}_part{fig_idx + 1}.png"
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)


def compute_language_centroids(hidden_states, languages, num_layers, hidden_dim):
    """
    Compute language centroids (mean hidden states) for each layer and language.
    
    Args:
        hidden_states (dict): Dictionary mapping language to hidden states {lang: {example_id: tensor}}
        languages (list): List of language codes (e.g., ['en', 'es', 'fr', 'de'])
        num_layers (int): Number of model layers
        hidden_dim (int): Hidden dimension size
        example_ids (list): Sorted list of example IDs
    
    Returns:
        dict: Dictionary of centroids {lang: tensor of shape (num_layers, hidden_dim)}
    """
    centroids = {}
    for lang in languages:
        lang_centroids = torch.zeros((num_layers, hidden_dim))
        for layer in range(num_layers):
            layer_hidden_states = []
            ex_ids = list(hidden_states[lang])
            for example_id in ex_ids:
                # hidden states shape: (num_layers, hidden_dim)
                hidden_state = hidden_states[lang][example_id][layer]
                layer_hidden_states.append(hidden_state)
            # compute for this language and layer the centroid
            layer_hidden_states = torch.stack(layer_hidden_states)  # shape: (num_examples, hidden_dim)
            layer_centroid = torch.mean(layer_hidden_states, dim=0)  # shape: (hidden_dim,)
            lang_centroids[layer] = layer_centroid
        centroids[lang] = lang_centroids    
    return centroids

def compute_language_centroids_standardised(hidden_states, languages, num_layers, hidden_dim):
    """
    Compute language centroids (mean standardized hidden states) for each layer and language.
    
    Args:
        hidden_states (dict): Dictionary mapping language to hidden states {lang: {example_id: tensor}}
        languages (list): List of language codes (e.g., ['en', 'es', 'fr', 'de'])
        num_layers (int): Number of model layers
        hidden_dim (int): Hidden dimension size
    
    Returns:
        dict: Dictionary of centroids {lang: tensor of shape (num_layers, hidden_dim)}
    """
    # fit a StandardScaler per layer on all langs/examples
    scalers = {}
    for layer in range(num_layers):
        all_feats = []
        for lang in languages:
            for ex_id in hidden_states[lang]:
                h = hidden_states[lang][ex_id][layer].to(torch.float32).numpy()
                all_feats.append(h)
        all_feats = np.stack(all_feats, axis=0) # (total_examples, hidden_dim)
        scalers[layer] = StandardScaler().fit(all_feats)
    
    # for each language & layer, transform then average to get centroid
    centroids = {}
    for lang in languages:
        lang_centroids = torch.zeros((num_layers, hidden_dim), dtype=torch.float32)
        for layer in range(num_layers):
            feats = []
            for ex_id in hidden_states[lang]:
                h = hidden_states[lang][ex_id][layer].to(torch.float32).numpy()
                feats.append(h)
            feats = np.stack(feats, axis=0)               # (n_examples, hidden_dim)
            scaled = scalers[layer].transform(feats)     # standardize
            centroid = scaled.mean(axis=0)                # (hidden_dim,)
            lang_centroids[layer] = torch.from_numpy(centroid)
        centroids[lang] = lang_centroids
    
    return centroids    

def compute_centroids_cosine_similarities(centroids, languages, num_layers):
    """
    Compute cosine similarity between all pairs of language centroids for each layer.
    
    Args:
        centroids (dict): Dictionary of centroids {lang: tensor of shape (num_layers, hidden_dim)}
        languages (list): List of language codes (e.g., ['en', 'es', 'fr', 'de'])
        num_layers (int): Number of model layers
    
    Returns:
        pd.DataFrame: Table with columns [Layer, Language_Pair, Cosine_Similarity]
    """
    print("\nIn compute_centroids_cosine_similarities")
    results = [] # list
    language_pairs = list(itertools.combinations(languages, 2))
    
    for layer in range(num_layers):
        for lang1, lang2 in language_pairs:

            centroid1 = centroids[lang1][layer].to(dtype=torch.float32).unsqueeze(0)  # shape: (1, hidden_dim)
            centroid2 = centroids[lang2][layer].to(dtype=torch.float32).unsqueeze(0)  # shape: (1, hidden_dim)
            sim = F.cosine_similarity(centroid1, centroid2, dim=-1)
            
            # store result
            results.append({'Layer': layer, 'Language_Pair': f'{lang1}-{lang2}','Cosine_Similarity': sim})
    return pd.DataFrame(results)

def plot_lang_centroids_cos_sim(similarity_df, output_dir, num_layers):
    """
    Plot cosine similarities between language pairs across layers with x-axis labeled from 1 to num_layers.
    
    Args:
        similarity_df (pd.DataFrame): DataFrame with columns [Layer, Language_Pair, Cosine_Similarity]
        output_dir (str or Path): Directory to save the plot
        num_layers (int): Number of model layers
    """
    print("\nIn plot_lang_centroids_cos_sim")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))
    
    # plot a line for each language pair
    for lang_pair in similarity_df['Language_Pair'].unique():
        pair_data = similarity_df[similarity_df['Language_Pair'] == lang_pair]
        # adjust layer numbers to 1-based indexing for plotting
        plt.plot(pair_data['Layer'], pair_data['Cosine_Similarity'], label=lang_pair, marker='o', markersize=4)

    plt.xlabel('Layer')
    plt.ylabel('Cosine Similarity')
    plt.title('Cosine Similarity Between Language Centroids by Layer')
    plt.legend(title='Language Pairs')
    plt.xticks(range(1, num_layers + 1))
    plt.ylim(0, 1)  # cosine similarity ranges from 0 to 1
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'centroids_cosine_similarities.png'
    # plt.show()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    # plt.close()

def plot_lang_centroids_cos_sim_per_language(similarity_df, languages, output_dir, num_layers, standardize=False):
    """
    For each base language, plot its centroid's cosine similarity to all other languages across layers,
    using the global LANGUAGE_COLORS. Single unified legend appears as a horizontal line under the plot.
    All layers are shifted by +1 so that layer 0 appears at x=1.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for base in languages:
        fig, ax = plt.subplots(figsize=(10, 6))
        for pair in similarity_df['Language_Pair'].unique():
            if base not in pair:
                continue
            lang1, lang2 = pair.split('-')
            other = lang2 if lang1 == base else lang1
            pair_data = similarity_df[similarity_df['Language_Pair'] == pair].sort_values('Layer')
            ax.plot(pair_data['Layer'] + 1, pair_data['Cosine_Similarity'], label=other, marker='o', markersize=4, color=LANGUAGE_COLORS[other], linewidth=1.5)

        ax.set_xlabel('Layer')
        ax.set_ylabel('Cosine Similarity')
        # ax.set_title(f'{base}')
        ax.set_xticks(range(1, num_layers + 1))
        # ax.tick_params(labelsize=25)
        ax.grid(True)

        # unified legend below the plot as a single horizontal line
        handles, labels = ax.get_legend_handles_labels()
        fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.02)
        ax.legend(handles, labels, title='Language', loc='upper center', bbox_to_anchor=(0.5, -0.09), ncol=len(labels), frameon=False, 
                  columnspacing=1.0, handlelength=1.5)

        if standardize:
            out_path = output_dir / f'{base}_centroid_cosine_similarities_stand.png'
        else:
            ax.set_ylim(0, 1)
            out_path = output_dir / f'{base}_centroid_cosine_similarities.png'
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved plot for {base} to: {out_path}")

def save_centroids(centroids, out_path):
    """
    Save the centroids dict to disk.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(centroids, out_path)
    print(f"Saved centroids to {out_path}")

def compute_centroids_euclidean_distances(centroids, languages, num_layers) -> pd.DataFrame:
    """
    Compute Euclidean distance between all pairs of language centroids for each layer.
    
    Returns a DataFrame with columns [Layer, Language_Pair, Euclidean_Distance].
    """
    results = []
    for layer in range(num_layers):
        for lang1, lang2 in itertools.combinations(languages, 2):
            c1 = centroids[lang1][layer].float()
            c2 = centroids[lang2][layer].float()
            dist = torch.norm(c1 - c2, p=2).item()
            results.append({
                "Layer": layer,
                "Language_Pair": f"{lang1}-{lang2}",
                "Euclidean_Distance": dist
            })
    return pd.DataFrame(results)

def plot_lang_centroids_euclidean_per_language(distance_df, languages, output_dir, num_layers, standardize=False):
    output_dir.mkdir(parents=True, exist_ok=True)

    for base in languages:
        fig, ax = plt.subplots(figsize=(10, 6))
        for pair in distance_df['Language_Pair'].unique():
            if base not in pair:
                continue
            l1, l2 = pair.split('-')
            other = l2 if l1 == base else l1
            data = distance_df[distance_df['Language_Pair'] == pair].sort_values('Layer')
            ax.plot(data['Layer'] + 1, data['Euclidean_Distance'], label=other, marker='o', markersize=4, linewidth=1.5, color=LANGUAGE_COLORS[other])

        ax.set_xlabel('Layer')
        ax.set_ylabel('Euclidean Distance')
        # ax.set_title(f'{base} centroid distances', fontsize=25)
        ax.set_xticks(range(1, num_layers + 1))
        # ax.tick_params(labelsize=25)
        ax.grid(True)

        # legend below
        handles, labels = ax.get_legend_handles_labels()
        fig.subplots_adjust(bottom=0.15, top=0.90)
        ax.legend(handles, labels, title='Language', loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=len(labels), frameon=False)

        suffix = "_stand" if standardize else ""
        out_name = f"{base}_centroid_euclidean_distances{suffix}.png"
        fig.savefig(output_dir / out_name, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved Euclidean plot for {base}: {out_name}")


def cohens_d_per_layer(norms_dict, labels_dict, min_count=2):
    """
    norms_dict: example_id -> array (num_layers,)
    labels_dict: example_id -> 0/1 correctness
    Returns: array shape (num_layers,) of Cohen's d (correct - incorrect)
    """
    example_ids = sorted(set(norms_dict.keys()) & set(labels_dict.keys()))
    if not example_ids:
        return np.array([])
    norms_mat = np.stack([norms_dict[eid] for eid in example_ids], axis=0)  # (N, L)
    labels = np.array([labels_dict[eid] for eid in example_ids], dtype=int)  # (N,)
    N, L = norms_mat.shape
    d = np.zeros(L, dtype=float)
    for l in range(L):
        corr = norms_mat[labels == 1, l]
        inc = norms_mat[labels == 0, l]
        if len(corr) < min_count or len(inc) < min_count:
            continue
        mu_corr = corr.mean()
        mu_inc = inc.mean()
        var_corr = corr.var(ddof=0)
        var_inc = inc.var(ddof=0)
        pooled = np.sqrt((len(corr) * var_corr + len(inc) * var_inc) / (len(corr) + len(inc)))
        d[l] = (mu_corr - mu_inc) / (pooled + 1e-8)
    return d