import os
import re
import pandas as pd
from openai import OpenAI

# config
DATA_SPLIT = "virology.csv" # TODO: update data split name as needed
DIR_IN = "../data/global_mmlu/en_regex_out/"   # input CSV (after regex filtering)

# MODEL = "gpt-4.1-mini"
MODEL = "gpt-4.1"
DIR_OUT = f"../data/global_mmlu/en_{MODEL}_out/"
ERROR_SCORE = -1

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


RUBRIC_PROMPT = """
Your task is to evaluate how well a multiple-choice question or incomplete statement (stem) can be answered without seeing the answer choices. 

Assign a score from 1 to 10 based only on whether the stem can be answered or completed without the choices.

*Scoring Scale (1-10):*

Score 1-2: The stem does not define a clear task and cannot be answered without choices.
Examples:
* "Oxygen is used:" 
* "The element (4, 2) of" 

Score 3-4: The stem defines a task, but answering it requires choosing from an unspecified list or comparison that is only provided by the choices.
Examples:
* "Which set of integers is in order from least to greatest?"
* "The most accurate statement is" 

Score 5: The stem can be answered without choices, but allows many reasonable completions or answers.
Examples:
* "Under which circumstances would you not use a catheter valve?"
* "The advantages of X-rays include"

Score 6-7: 
The stem can be answered without choices and limits the topic, but allows variation in the expected answer.
Examples:
* "What is one function of the liver?"
* "As temperature increases, reaction rate generally"

Score 8-9: The stem can be answered without choices and strongly limits the expected answer.
Examples:
* "What gas do plants primarily absorb during photosynthesis?"
* "The enzymes of glycolysis are located in the"

Score 10: The stem alone specifies a single, exact expected answer.
Examples:
* "What is the first law of thermodynamics in physics?"
* "Calculate 15 × 12."

*Output Instructions:*
You must output exactly one line containing only one integer from 1 to 10, without any additional text.

*Response Format (must match exactly):*
SCORE: X
""".strip()


def llm_score_question(question: str) -> int:
    """
    Ask the LLM to rate the question 1–10 according to the rubric.
    Returns an integer from 1 to 10 (best-effort parsing).
    """
    user_content = f"QUESTION: {question}\nSCORE:"

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": RUBRIC_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        raw = resp.choices[0].message.content.strip()
        print("\n")
        print(question)
        print(raw)
        # Extract first integer 1–10 from the response
        match = re.search(r"\b([1-9]|10)\b", raw)
        if match:
            return int(match.group(1))
        else:
            print(f"[WARN] Could not parse score from response: {raw!r}. Returning response {raw}.")
            return raw
    except Exception as e:
        print(f"[ERROR calling LLM] {e}. Returning error score {ERROR_SCORE}.")
        return ERROR_SCORE


# load data
in_path = os.path.join(DIR_IN, DATA_SPLIT)
df = pd.read_csv(in_path)
base = DATA_SPLIT.rsplit(".csv", 1)[0]
print(f"Loaded {len(df)} rows from {in_path}")

# llm scoring
scores = []
for i, row in df.iterrows():
    q = str(row["question"])
    score = llm_score_question(q)
    scores.append(score)
    if (i + 1) % 50 == 0:
        print(f"Scored {i + 1}/{len(df)} questions... (last score = {score})")

df["llm_score"] = scores

# make sure output dir exists
# save results
os.makedirs(DIR_OUT, exist_ok=True)
# save results as JSONL
out_rated = os.path.join(DIR_OUT, base + ".jsonl")
df.to_json(out_rated, orient="records", lines=True, force_ascii=False)

print("\n=== LLM rating summary ===")
print("Total questions:", len(df))
print("Saved file:", out_rated)