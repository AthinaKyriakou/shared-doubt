import os 

# DATA DETAILS
DATA_DIR = "/path/to/data/"
LOCAL_DATA_DIR = "/path/to/local/data"
DATASETS = ["mkqa", "global_mmlu"]
DATASPLITS_MKQA = [
    "mkqa_answerable_short_phrase", "mkqa_answerable_date_no_duplicates", 
    "mkqa_answerable_number_no_duplicates", "mkqa_answerable_number_with_unit_no_duplicates", "mkqa_answerable_entity_no_duplicates"
]
DATASPLITS_GMMLU = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine", "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics", "global_facts", "high_school_biology", "high_school_chemistry",
    "high_school_computer_science", "high_school_geography", "high_school_government_and_politics", "high_school_macroeconomics", "high_school_mathematics",
    "formal_logic", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics", "human_aging", "human_sexuality", "international_law",
    "jurisprudence", "logical_fallacies", "machine_learning", "management", "marketing", "medical_genetics", "miscellaneous", "moral_disputes", "nutrition",
    "philosophy", "prehistory", "professional_accounting", "professional_law", "professional_medicine", "professional_psychology", "public_relations",
    "security_studies", "sociology", "us_foreign_policy", "virology", "world_religions"
]
DATASPLITS = DATASPLITS_MKQA + DATASPLITS_GMMLU
  
LANGUAGES = ["en", "fr", "es", "pl", "ru", "ja"]

# MODEL DETAILS
MODELS = ("llama_3.1_8B", "qwen3_8B")
LLAMA_MODEL_PATH = "/path/to/models/Meta-Llama-3.1-8B-Instruct"
QWEN_MODEL_PATH = "/path/to/models/Qwen3-8B"
SYS_PROMPT_FILE = "/path/to/src/sys_prompts.json"

# TYPES OF ACTIONS
OPERATIONS = ["extract_hidden_states", "merge_hidden_states"]

# RESULTS
RESULTS_DIR = "/path/to/results/"
LOCAL_RESULTS_DIR = "/path/to/local/results/"
EMISSIONS_DIR = f"{RESULTS_DIR}emissions/"

# CONFIG
BATCH_SIZE = 32
MAX_NEW_TOKENS = 40