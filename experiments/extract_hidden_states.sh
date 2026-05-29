#!/usr/bin/env bash
set -euo pipefail

DATA_SPLITS=(
    # MKQA
    mkqa_answerable_short_phrase
    mkqa_answerable_date_no_duplicates
    mkqa_answerable_number_no_duplicates
    mkqa_answerable_number_with_unit_no_duplicates
    mkqa_answerable_entity_no_duplicates

    # GMMLU
    # abstract_algebra anatomy astronomy business_ethics 
    # clinical_knowledge college_biology college_chemistry
    # college_computer_science 
    # college_mathematics 
    # college_medicine

    # college_physics computer_security conceptual_physics
    # econometrics electrical_engineering elementary_mathematics global_facts high_school_biology 
    # high_school_chemistry 
    # high_school_computer_science high_school_geography high_school_government_and_politics 
    # high_school_mathematics

    # formal_logic 
    # high_school_macroeconomics 
    # high_school_microeconomics high_school_physics high_school_psychology 
    # high_school_statistics
    
    # human_aging human_sexuality international_law
    # jurisprudence logical_fallacies 
    # machine_learning 
    # management 
    # marketing 
    # medical_genetics miscellaneous moral_disputes nutrition
    # philosophy prehistory 

    # professional_accounting 
    # professional_law 
    # professional_medicine 

    # professional_psychology public_relations 
    # security_studies sociology us_foreign_policy virology 
    # world_religions
)

LANGUAGES=(
    # en
    es
    # fr
    # ja
    # ru
    # pl
)

for SPLIT in "${DATA_SPLITS[@]}"; do
    for LANG in "${LANGUAGES[@]}"; do

        echo ""
        echo "========================================"
        echo "  Dataset split : $SPLIT"
        echo "  Language      : $LANG"
        echo "========================================"
        echo ""

        python ../src/hidden_states.py \
            --track-emissions \
            extract_hidden_states \
            --model-name llama_3.1_8B \
            --dataset-name mkqa \
            --datasplit-name "$SPLIT" \
            --lang "$LANG"

    done
done