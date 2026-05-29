import pandas as pd
import re

# Configs
DATA_SPLIT = "virology.csv" # TODO: update data split name as needed
DIR_IN = "../data/global_mmlu/en/"
DIR_OUT = "../data/global_mmlu/en_regex_out/"
df = pd.read_csv(DIR_IN + DATA_SPLIT)

# 1) Compile regex for pattern-based questions
pattern = re.compile(
    r"\b("
    r"following|"
    r"options|"
    r"which among them|"
    r"which statement|"
    r"true|"
    r"false|"
    r"correct|"
    r"incorrect|"
    r"best|"
    r"generally speaking|"
    r"which of these combinations|"
    r"likely"
    r")\b",
    re.IGNORECASE
)

# 2) Compile regex for questions containing blanks (3+ underscores)
underscore_pattern = re.compile(r"_{3,}")

# 3) Compile regex for questions with "statement 1" AND "statement 2"
statement1_pattern = re.compile(r"statement\s*1", re.IGNORECASE)
statement2_pattern = re.compile(r"statement\s*2", re.IGNORECASE)

# 4) Compile regex for questions with true/false expressions without the word "following"
truth_words = re.compile(r"\b(true|false|correct|incorrect)\b", re.IGNORECASE)
following_word = re.compile(r"\bfollowing\b", re.IGNORECASE)

# 4) Create masks
mask_pattern = df["question"].str.contains(pattern, na=False)
mask_underscores = df["question"].str.contains(underscore_pattern, na=False)
mask_statement = (df["question"].str.contains(statement1_pattern, na=False) & df["question"].str.contains(statement2_pattern, na=False))

# Questions that have true/false/correct/incorrect BUT do NOT have "following"
mask_truth_no_following = (
    df["question"].str.contains(truth_words, na=False)
    & ~df["question"].str.contains(following_word, na=False)
)

# General removal (pattern or underscores)
mask_general = mask_pattern | mask_underscores

# Total removal mask (either general OR statement 1&2)
remove_mask = mask_general | mask_statement

# 5) Split into groups
# removed by general patterns/underscores, but NOT statement-1&2, NOT truth-no-following
removed_general_df = df[mask_general & ~mask_statement & ~mask_truth_no_following]
# removed because of "statement 1" & "statement 2"
removed_statement_df = df[mask_statement]
# removed because of truth words without "following"
removed_truthvalue_df = df[mask_truth_no_following]
# kept rows (not in any removal mask)
filtered_df = df[~remove_mask]

# 6) Save results
base = DATA_SPLIT.rsplit(".csv", 1)[0]

# Overwrite original with filtered data
filtered_df.to_csv(DIR_OUT + DATA_SPLIT, index=False)

# Save removed groups
removed_general_df.to_csv(DIR_OUT + base + "_removed_general.csv", index=False)
removed_statement_df.to_csv(DIR_OUT + base + "_statements.csv", index=False)
removed_truthvalue_df.to_csv(DIR_OUT + base + "_truthvalue.csv", index=False)

# 7) Print summary
print("Original rows:", len(df))
print("Filtered rows:", len(filtered_df))
print("Removed (general):", len(removed_general_df))
print("Removed (statement 1 & 2):", len(removed_statement_df))
print("Removed (truth w/o 'following'):", len(removed_truthvalue_df))
print("Removed (total):", len(df) - len(filtered_df))

print("\nGeneral removal sample_ids:")
for sid in removed_general_df["sample_id"]:
    print(sid)

print("\nStatement 1 & 2 removal sample_ids:")
for sid in removed_statement_df["sample_id"]:
    print(sid)

print("\nTruth-value (no 'following') removal sample_ids:")
for sid in removed_truthvalue_df["sample_id"]:
    print(sid)