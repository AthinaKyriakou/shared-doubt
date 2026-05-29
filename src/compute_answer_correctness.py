from pathlib import Path
from utils import load_answers, compute_correctness_llm_judge
import json
from collections import defaultdict
from constants import LANGUAGES, DATASPLITS_MKQA, DATASPLITS_GMMLU, LOCAL_RESULTS_DIR, MODELS, DATASETS

def load_answers(lang, results_dir, suffix, dataset):
    file_path = Path(results_dir) / f'{lang}_{suffix}.jsonl'
    answers = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            id_key = 'example_id' if dataset == 'mkqa' else 'sample_id'
            answers[item[id_key]] = item['judgment']
    return answers


def compute_correctness_llm_judge(lang, answers_lang):
    correctness = {}
    for eid, judgment in answers_lang.items():
        normalized = judgment.strip().upper()
        correctness[eid] = 1 if normalized == "YES" else 0
    return correctness


def compute_correctness(languages, ds_results_dir, dataset):
    correctness_dict = defaultdict(dict)
    answers = {}
    for lang in languages:
        try:
            if dataset == 'global_mmlu' and lang != 'fr':
                answers[lang] = load_answers(lang, ds_results_dir, "judgment_mini", dataset)
            else:
                answers[lang] = load_answers(lang, ds_results_dir, "judgment", dataset)
        except FileNotFoundError:
            print(f"[SKIP] judgment file not found for lang='{lang}' in {ds_results_dir}")
            continue
    for lang in languages:
        if lang not in answers:
            continue
        print(f"lang: {lang}")
        answers_lang = answers[lang]
        corr_ds_lang = compute_correctness_llm_judge(lang, answers_lang)
        correctness_dict[lang].update(corr_ds_lang)

    for lang, corr in correctness_dict.items():
        out_path = Path(ds_results_dir) / f"{lang}_correctness.jsonl"
        with out_path.open("w", encoding="utf-8") as fout:
            fout.write(json.dumps(corr, ensure_ascii=False) + "\n")


if __name__ == "__main__":

    MODELS = ['llama_3.1_8B']
    DATASETS = ['mkqa']
    LANGUAGES = ['es']

    for llm in MODELS:
        for dataset in DATASETS:
            datasplits = DATASPLITS_MKQA if dataset == 'mkqa' else DATASPLITS_GMMLU
            for ds in datasplits:
                print(f"ds: {ds}")
                ds_results_dir = f"{LOCAL_RESULTS_DIR}/{dataset}/{llm}/{ds}/"
                compute_correctness(LANGUAGES, ds_results_dir, dataset)