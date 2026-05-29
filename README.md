# Shared Doubt: Zero-shot Cross-Lingual Confidence Estimation for Language Models

This repository contains the code for our paper on multilingual uncertainty quantification in large language models using lightweight probing.

> **Note:** Some artefacts (extracted hidden states, trained probes) are omitted due to size constraints and can be shared upon request.


## Overview

We investigate whether uncertainty-predictive features in hidden states of multilingual LLMs generalise across languages. We train linear probes on the hidden representations of **Llama 3.1 8B Instruct** and **Qwen3-8B** and evaluate their cross-lingual transfer on six languages: English (`en`), French (`fr`), Spanish (`es`), Polish (`pl`), Russian (`ru`), and Japanese (`ja`).

We benchmark against a range of baselines and perform ablation studies on layer readout and representation.


## Installation

The code requires **Python 3.12** and was run on GPU compute nodes. All dependencies can be installed from `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If running on a cluster that requires loading a Python module first (e.g., an HPC system with Environment Modules), load the appropriate GPU-enabled Python module before creating the virtual environment.


## Folder Structure

```
.
├── data/                 # Dataset files
├── experiments/          # Bash scripts and notebooks for running experiments
├── src/                  # All source code
└── requirements.txt
```


## Pipeline

### 1. Data Preprocessing

We use two datasets:
- **[MKQA](https://aclanthology.org/2021.tacl-1.82/)** — multilingual knowledge questions. We focus on the `entity`, `short_phrase`, `date`, `number`, and `number_with_unit` question types.
- **[Global MMLU](https://huggingface.co/datasets/CohereForAI/Global-MMLU)** — multilingual multiple-choice benchmarks.

**MKQA filtering** is done via:
- `experiments/data_preprocessing.ipynb` — main preprocessing notebook
- `experiments/filter_mkqa_llm_judge.py` — LLM-based answer filtering

**Global MMLU filtering** is done via:
- `experiments/filter_global_mmlu_llm_judge.py` — LLM-based filtering
- `experiments/filter_global_mmlu_regex.py` — regex-based filtering

Processed dataset files are stored in `data/`. Train/validation/test splits use a stratified approach and are stored in `results/{dataset}/{model}/{datasplit}/`.


### 2. Hidden State & Answer Extraction

For each dataset, question type, and language, we prompt each model and extract hidden states from both the prompt-encoding phase (last query token) and the generation phase (last output token).

- **Implementation**: `src/hidden_states.py`
- **Cluster runner**: `experiments/extract_hidden_states.sh`
- **Merging sharded outputs**: `experiments/merge_hidden_states.sh`

Hidden states are stored in HDF5 format under `results/{dataset}/{model}/{datasplit}/{lang}_*.h5`.


### 3. Answer Correctness Assessment

Model answers are assessed for correctness using **GPT-4.1/GPT-4.1-mini as a judge**.

- **Prompting the judge**: `src/assess_answer_correctness.py` (requires `OPENAI_API_KEY`)
- **Computing binary labels**: `src/compute_answer_correctness.py`
- **Judge prompts**: `src/judge_correctness_prompts.yaml`

Output files:
- `results/{dataset}/{model}/{datasplit}/{lang}_judgment.jsonl`
- `results/{dataset}/{model}/{datasplit}/{lang}_correctness.jsonl`


### 4. Uncertainty Quantification Probe

The main probe is `SoftmaxLayerProbe`, implemented in `src/probes.py`. Available architectures:

**Hyperparameter tuning** (Ray Tune + ASHA scheduler):
```bash
python src/htune_probe.py
```

**Training**:
```bash
python src/train_probe.py
```

**Evaluation**:
```bash
python src/eval_probe.py           # standard cross-lingual eval
python src/eval_probe_datasplit.py # per-datasplit eval
```

Trained probes and evaluation results are written to `results/{dataset}/{model}/`.


### 5. Baselines

We compare against the following baselines:

| Baseline | Implementation | Runner |
|---|---|---|
| Length-Normalised Sequence Likelihood (NSL) | `src/nsl_baseline.py`, `src/eval_nsl_baseline.py` | `experiments/run_nsl_baseline.sh` |
| P(True) (Kadavath et al., 2022) | `src/ptrue_baseline.py`, `src/eval_ptrue_baseline.py` | `experiments/run_ptrue_baseline.sh` |
| Verbalised Uncertainty (Tian et al., 2023) | `src/verbal_confidence_baseline.py`, `src/eval_verbalised_confidence_baseline.py` | `experiments/run_verbal_confidence_baseline.sh` |
| Mass-Mean Probe (Marks & Tegmark, 2024) | `src/mass_mean_probe.py` | — |
| Majority / Prior-Probability | `src/classification_baselines.py` | — |

All baselines report the same metrics as the probe: **AUROC**, **Brier Score**, **ECE**, and **AUPR**.


### 6. Ablations

Two ablation studies assess the contribution of individual components:

- **Weight ablation** (sliding-window readout): `src/weight_ablation.py`
- **Representation ablation** (sliding-window input): `src/representation_ablation.py`

Results are visualised with `experiments/eval_ablation_results.ipynb` / `experiments/eval_ablation_results.py`.


## Metrics

All evaluation scripts report:
- **AUROC** — area under the ROC curve
- **Brier Score** — mean squared error of probability estimates
- **ECE** — expected calibration error
- **AUPR** — area under the precision–recall curve

## Configuration

Key constants (model paths, dataset names, language codes, batch sizes) are centralised in `src/constants.py`. Update the path variables to match your local or cluster environment before running any script.