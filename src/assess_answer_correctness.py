from dotenv import load_dotenv
from openai import OpenAI
from itertools import product
import os
import json
import yaml
from constants import MODELS, DATASPLITS_MKQA, DATASPLITS_GMMLU, LOCAL_DATA_DIR, LANGUAGES, LOCAL_RESULTS_DIR
import logging
from tqdm import tqdm

LANGUAGES = ["es"]
MODELS = ["llama_3.1_8B"]
DATASET = "mkqa"
JUDGE_MODEL = "gpt-4.1-mini"
JUDGE_PROMPTS_FILE = "judge_correctness_prompts.yaml"
LANG_YES_NO = {
    "en": ("YES", "NO"),
    "fr": ("OUI", "NON"),
    "es": ("SÍ", "NO"),
    "ru": ("ДА", "НЕТ"),
    "ja": ("はい", "いいえ"),
    "pl": ("TAK", "NIE")
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)

# Get environment variables from .env.
load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Load judge prompts once at module level
with open(JUDGE_PROMPTS_FILE, encoding="utf-8") as f:
    JUDGE_PROMPTS = yaml.safe_load(f)

def ask_llm_judge(correct_variants, reference, system_prompt, user_prompt_template, lang):
    gold_list = "\n".join(f"- {v}" for v in correct_variants)
    user_prompt = user_prompt_template.replace("{gold_list}", gold_list).replace("{reference}", reference)

    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    answer = (resp.choices[0].message.content or "").strip().upper()
    yes_token, no_token = LANG_YES_NO.get(lang, ("YES", "NO"))
    if answer.startswith(yes_token):
        return "YES"
    if answer.startswith(no_token):
        return "NO"
    return answer


def load_ground_truth_mkqa(path, lang):
    """
    Returns a dict mapping example_id -> list of full-answer variants (comma-separated strings).
    It first removes duplicate entries by 'text', then builds all combinations of each entry's text + aliases.
    """
    mapping = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            ex_id = rec.get("example_id")
            # remove duplicate answer entries by primary text
            seen_text = set()
            unique_entries = []
            for entry in rec.get("answers", {}).get(lang, []):
                text = entry.get("text")
                if text and text not in seen_text:
                    seen_text.add(text)
                    unique_entries.append(entry)
            # build list of options per unique entry
            options = []
            for entry in unique_entries:
                texts = [entry.get("text")] + entry.get("aliases", [])
                # dedupe within this entry
                seen_opts = []
                for t in texts:
                    if t and t not in seen_opts:
                        seen_opts.append(t)
                if seen_opts:
                    options.append(seen_opts)
            # cartesian product of options for full variant lists
            variants = []
            for combo in product(*options):
                variants.append(", ".join(combo))
            mapping[ex_id] = variants
    return mapping


def load_ground_truth_global_mmlu(path, lang=None):
    """
    Returns:
      gold_map:  sample_id -> [correct option text]
      score_map: sample_id -> llm_score (or None if missing)
    """
    gold_map = {}
    score_map = {}
    letter_to_field = {"A": "option_a", "B": "option_b", "C": "option_c", "D": "option_d"}

    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            sample_id = rec.get("sample_id")
            if not sample_id:
                continue
            score_map[sample_id] = rec.get("llm_score")
            ans_letter = (rec.get("answer") or "").strip().upper()
            field = letter_to_field.get(ans_letter)
            if not field:
                gold_map[sample_id] = []
                continue
            option_text = rec.get(field)
            if option_text is None:
                gold_map[sample_id] = []
            else:
                option_text = str(option_text).strip()
                gold_map[sample_id] = [option_text] if option_text else []

    return gold_map, score_map


def load_predictions(path):
    """
    Returns a list of (id, predicted_text) tuples.
    Supports both example_id (mkqa) and sample_id (global_mmlu).
    """
    preds = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            ex_id = rec.get("example_id", rec.get("sample_id"))
            preds.append((ex_id, rec["answer"]))
    return preds


def get_already_done(output_file, id_key):
    """Returns a set of example ids already present in the output file."""
    already_done = set()
    if os.path.exists(output_file):
        with open(output_file, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                already_done.add(rec.get(id_key))
    return already_done


def main():
    if DATASET == "mkqa":
        datasplits = DATASPLITS_MKQA
        for llm in MODELS:
            for lang in LANGUAGES:
                for ds in datasplits:
                    ground_truth_file = f"{LOCAL_DATA_DIR}/{DATASET}/{ds}.jsonl"
                    logging.info(f"Processing: {llm} | {ds} | {lang}")

                    system_prompt = JUDGE_PROMPTS["system"][lang]
                    user_prompt_template = JUDGE_PROMPTS["user"][lang]

                    answers_file = f"{LOCAL_RESULTS_DIR}/{DATASET}/{llm}/{ds}/{lang}_answer.jsonl"
                    output_file = f"{LOCAL_RESULTS_DIR}/{DATASET}/{llm}/{ds}/{lang}_judgment.jsonl"

                    gold_map = load_ground_truth_mkqa(ground_truth_file, lang)
                    preds = load_predictions(answers_file)
                    already_done = get_already_done(output_file, "example_id")

                    preds_to_judge = [(ex_id, ref) for ex_id, ref in preds if ex_id not in already_done]
                    logging.info(f"{len(preds_to_judge)} examples to judge | {len(already_done)} already done")

                    with open(output_file, "a", encoding="utf-8") as outf:
                        for ex_id, reference in tqdm(preds_to_judge, desc=f"[{ds}/{lang}]", unit="example"):

                            if ex_id not in gold_map:
                                rec = {
                                    "example_id": ex_id,
                                    "reference": reference,
                                    "gold_variants": None,
                                    "judgment": "ERROR",
                                }
                                outf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                                continue

                            try:
                                gold_variants = gold_map[ex_id]
                                judgment = ask_llm_judge(
                                    gold_variants, reference,
                                    system_prompt, user_prompt_template, lang
                                )
                                rec = {
                                    "example_id": ex_id,
                                    "reference": reference,
                                    "gold_variants": gold_variants,
                                    "judgment": judgment,
                                }
                                outf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            except Exception as e:
                                logging.warning(f"Failed for example {ex_id}: {e}")
                                continue

    else:
        datasplits = DATASPLITS_GMMLU
        for llm in MODELS:
            for lang in LANGUAGES:
                for ds in datasplits:
                    
                    logging.info(f"Processing: {ds} | {lang}")

                    system_prompt = JUDGE_PROMPTS["system"][lang]
                    user_prompt_template = JUDGE_PROMPTS["user"][lang]

                    ground_truth_file = f"{LOCAL_DATA_DIR}/global_mmlu/{lang}_final/{ds}.jsonl"
                    answers_file = f"{LOCAL_RESULTS_DIR}/{DATASET}/{llm}/{ds}/{lang}_answer.jsonl"
                    if not os.path.exists(answers_file):
                        continue
                    output_file = f"{LOCAL_RESULTS_DIR}/{DATASET}/{llm}/{ds}/{lang}_judgment_mini.jsonl"

                    gold_map, score_map = load_ground_truth_global_mmlu(ground_truth_file, lang)
                    preds = load_predictions(answers_file)
                    already_done = get_already_done(output_file, "sample_id")

                    preds_to_judge = [(ex_id, ref) for ex_id, ref in preds if ex_id not in already_done]
                    logging.info(f"{len(preds_to_judge)} examples to judge | {len(already_done)} already done")

                    with open(output_file, "a", encoding="utf-8") as outf:
                        for ex_id, reference in tqdm(preds_to_judge, desc=f"[{ds}/{lang}]", unit="example"):

                            llm_score = score_map.get(ex_id)

                            if ex_id not in gold_map:
                                rec = {
                                    "sample_id": ex_id,
                                    "reference": reference,
                                    "gold_variants": None,
                                    "judgment": "ERROR",
                                    "llm_score": llm_score,
                                }
                                outf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                                continue

                            try:
                                gold_variants = gold_map[ex_id]
                                judgment = ask_llm_judge(
                                    gold_variants, reference,
                                    system_prompt, user_prompt_template, lang
                                )
                                rec = {
                                    "sample_id": ex_id,
                                    "llm_score": llm_score,
                                    "reference": reference,
                                    "gold_variants": gold_variants,
                                    "judgment": judgment,
                                }
                                outf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            except Exception as e:
                                logging.warning(f"Failed for example {ex_id}: {e}")
                                continue


if __name__ == "__main__":
    main()