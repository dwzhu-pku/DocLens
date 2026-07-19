#!/bin/bash


SCRIPT_DIR=$(cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

DATASET_NAME="MMLongBenchDoc" # [MMLongBenchDoc, FinRAGBench-V, LongDocURL, PaperTab, FetaTab, MMDocRAG, DocBench]
SPLIT_NAME="samples_100" # of any other names of the splits
DATASET_BASE_PATH="${PROJECT_ROOT}/data/${DATASET_NAME}/documents"

PHASE_NAME="baseline"
EXP_NAME=${PHASE_NAME}

# [claude-sonnet-4@20250514, gemini-2.5-pro, gemini-2.5-flash, qwen3-vl-8b-instruct]
for model_name in "gemini-2.5-pro"; do

    MODEL_NAME="${model_name}"
    echo "-------------------------------------"
    echo "Running DocLens baseline with the following configurations:"
    echo "Dataset Name: ${DATASET_NAME}"
    echo "Split Name: ${SPLIT_NAME}"
    echo "Phase Name: ${PHASE_NAME}"
    echo "Model Name: ${MODEL_NAME}"
    echo "Experiment Name: ${EXP_NAME}"

    python ${PROJECT_ROOT}/main.py \
        --dataset_name ${DATASET_NAME} \
        --split_name ${SPLIT_NAME} \
        --model_name ${MODEL_NAME} \
        --phase_name ${PHASE_NAME} \
        --exp_name ${EXP_NAME}

    echo "Finished DocLens baseline mode with model: ${model_name}"
    echo "-------------------------------------"
done

