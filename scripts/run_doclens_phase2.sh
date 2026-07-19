#!/bin/bash


SCRIPT_DIR=$(cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
DATASET_BASE_PATH="${PROJECT_ROOT}/data/${DATASET_NAME}/documents"

DATASET_NAME="MMLongBenchDoc"
SPLIT_NAME="samples"
INPUT_MODE="located_evidence_image_zoom"
PHASE_NAME="phase2"
MODEL_NAME="gemini-2.5-pro"
CANDIDATE_NUM=8
EXP_NAME="phase2"

echo "Running DocLens Phase 2 with the following configurations:"
echo "Dataset Name: ${DATASET_NAME}"
echo "Split Name: ${SPLIT_NAME}"
echo "Input Mode: ${INPUT_MODE}"
echo "Model Name: ${MODEL_NAME}"
echo "Candidate Num: ${CANDIDATE_NUM}"
echo "Phase Name: ${PHASE_NAME}"
echo "Experiment Name: ${EXP_NAME}"

python ${PROJECT_ROOT}/main.py \
    --dataset_name ${DATASET_NAME} \
    --split_name ${SPLIT_NAME} \
    --input_mode ${INPUT_MODE} \
    --model_name ${MODEL_NAME} \
    --candidate_num ${CANDIDATE_NUM} \
    --phase_name ${PHASE_NAME} \
    --exp_name ${EXP_NAME}

