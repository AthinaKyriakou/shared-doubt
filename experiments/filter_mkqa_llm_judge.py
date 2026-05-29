import os
import json
from dotenv import load_dotenv
from openai import OpenAI

# config
DIR_IN = "../data/mkqa/"
DS = "mkqa_answerable_number_with_unit_no_duplicates" # TODO: specify which MKQA data split to filter
INPUT_FILE = os.path.join(DIR_IN, f"{DS}.jsonl")
OUTPUT_DIR = os.path.join(DIR_IN, "time_filtering")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"{DS}.jsonl")

MODEL = "gpt-4.1"
load_dotenv() # get environment variables from .env.
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


FILTERING_PROMPT = """

Your task is to determine whether the following question is TIME-SENSITIVE relative to when the dataset was collected (around 2019).

Assume the question was asked in or before 2019.

A question is TIME-SENSITIVE if its correct answer depends on the real-world time at which the question is asked.

In particular, a question is time-sensitive if its answer depends on:
- The current date, year, or recent events,
- Relative time expressions (e.g., "this year", "currently", "latest", "now", "recent"),
- The current real-world status, role, ranking, statistic, population, record, or version,
- Ongoing or changing real-world situations.

A question is NOT time-sensitive if its answer is fixed by historical facts, completed events, or already-released creative works 
(such as movies, TV series, books, or games), and does not change depending on when the question is asked.

*Output Instructions:*

Answer with exactly one of the following:

- YES — The question is time-sensitive (its answer would differ depending on when it is asked)
- NO — The question is not time-sensitive (its answer would be the same regardless of whether it is asked in 2015, 2019, or 2024)
- UNCERTAIN — The question cannot be reasonably judged

Do not provide any explanation.

*Examples:*

Time-sensitive (YES):
- who is the prime minister of japan
- what is the population of india
- who is the ceo of apple
- what is the latest version of android
- what was the winner of last year’s champions league
- how old is the president of france

Not time-sensitive (NO):
- who wrote the novel 1984
- when was the berlin wall built
- who discovered penicillin
- what is the capital of argentina
- when does prentis come back to criminal minds
- who painted the sistine chapel ceiling

Question:
\"\"\"{question}\"\"\"
""".strip()


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(OUTPUT_FILE, "a", encoding="utf-8") as fout:

        for line in fin:

            data = json.loads(line)

            example_id = data.get("example_id")
            query_en = data.get("query")

            if not query_en:
                verdict = "ERROR"

            else:
                prompt = FILTERING_PROMPT.format(question=query_en)

                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": "You are a careful evaluator."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=5,
                )

                verdict = response.choices[0].message.content.strip().upper()
                print(verdict)

            out = {
                "example_id": example_id,
                "query": query_en,
                "time_sensitive": verdict
            }

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush() # ensure immediate write


if __name__ == "__main__":
    main()