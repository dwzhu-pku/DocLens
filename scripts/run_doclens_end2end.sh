#!/bin/bash


SCRIPT_DIR=$(cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

DATASET_NAME="MMLongBenchDoc" # [MMLongBenchDoc, FinRAGBench-V, LongDocURL, PaperTab, FetaTab, MMDocRAG, DocBench]
SPLIT_NAME="samples_50" # of any other names of the splits
DATASET_BASE_PATH="${PROJECT_ROOT}/data/${DATASET_NAME}/documents"

PHASE1_INPUT_PAGES="all"
PHASE1_INPUT_MODE="use_ocr"
PHASE1_CANDIDATE_NUM=2

PHASE2_INPUT_PAGES="pgnav_all_located_pages" # [all, pgnav_all_located_pages, evidence_pages]
PHASE2_INPUT_MODE="use_element_localizer" # [use_ocr, use_element_localizer]
PHASE2_CANDIDATE_NUM=2

PHASE_NAME="end2end"
EXP_NAME="end2end_k${PHASE1_CANDIDATE_NUM}_k${PHASE2_CANDIDATE_NUM}_hybrid"

# [claude-sonnet-4@20250514, gemini-2.5-pro, gemini-2.5-flash, qwen3-vl-8b-instruct]
for model_name in "gemini-2.5-pro"; do
    echo "-------------------------------------"
    echo "Running DocLens end-to-end with model: ${model_name}"

    MODEL_NAME="${model_name}"
    
    echo "Running DocLens end-to-end with the following configurations:"
    echo "Dataset Name: ${DATASET_NAME}"
    echo "Split Name: ${SPLIT_NAME}"

    echo "Phase1 Input Mode: ${PHASE1_INPUT_MODE}"
    echo "Phase1 Input Pages: ${PHASE1_INPUT_PAGES}"
    echo "Phase1 Candidate Num: ${PHASE1_CANDIDATE_NUM}"
    echo "Phase2 Input Mode: ${PHASE2_INPUT_MODE}"
    echo "Phase2 Input Pages: ${PHASE2_INPUT_PAGES}"
    echo "Phase2 Candidate Num: ${PHASE2_CANDIDATE_NUM}"

    echo "Model Name: ${MODEL_NAME}"
    echo "Phase Name: ${PHASE_NAME}"
    echo "Experiment Name: ${EXP_NAME}"

    python ${PROJECT_ROOT}/main.py \
        --dataset_name ${DATASET_NAME} \
        --split_name ${SPLIT_NAME} \
        --phase1_input_mode ${PHASE1_INPUT_MODE} \
        --phase1_input_pages ${PHASE1_INPUT_PAGES} \
        --phase1_candidate_num ${PHASE1_CANDIDATE_NUM} \
        --phase2_input_mode ${PHASE2_INPUT_MODE} \
        --phase2_input_pages ${PHASE2_INPUT_PAGES} \
        --phase2_candidate_num ${PHASE2_CANDIDATE_NUM} \
        --model_name ${MODEL_NAME} \
        --phase_name ${PHASE_NAME} \
        --exp_name ${EXP_NAME}
    
    echo "Finished DocLens end-to-end with model: ${model_name}"
    echo "-------------------------------------"
done

