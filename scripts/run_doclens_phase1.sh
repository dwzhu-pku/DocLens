#!/bin/bash


SCRIPT_DIR=$(cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
DATASET_BASE_PATH="${PROJECT_ROOT}/data/${DATASET_NAME}/documents"

DATASET_NAME="FinRAGBench-V" # MMLongBenchDoc, FinRAGBench-V
SPLIT_NAME="samples"

PHASE1_INPUT_PAGES="all"
PHASE1_INPUT_MODE="use_ocr"
PHASE1_CANDIDATE_NUM=8

PHASE_NAME="phase1"
MODEL_NAME="gemini-2.5-flash" # claude-sonnet-4@20250514, gemini-2.5-pro, gemini-2.5-flash
EXP_NAME="phase1"

echo "Running DocLens Phase 1 with the following configurations:"
echo "Dataset Name: ${DATASET_NAME}"
echo "Split Name: ${SPLIT_NAME}"

echo "Phase1 Input Mode: ${PHASE1_INPUT_MODE}"
echo "Phase1 Input Pages: ${PHASE1_INPUT_PAGES}"
echo "Phase1 Candidate Num: ${PHASE1_CANDIDATE_NUM}"

echo "Model Name: ${MODEL_NAME}"
echo "Phase Name: ${PHASE_NAME}"
echo "Experiment Name: ${EXP_NAME}"

python ${PROJECT_ROOT}/main.py \
    --dataset_name ${DATASET_NAME} \
    --split_name ${SPLIT_NAME} \
    --phase1_input_mode ${PHASE1_INPUT_MODE} \
    --phase1_input_pages ${PHASE1_INPUT_PAGES} \
    --phase1_candidate_num ${PHASE1_CANDIDATE_NUM} \
    --model_name ${MODEL_NAME} \
    --phase_name ${PHASE_NAME} \
    --exp_name ${EXP_NAME}

