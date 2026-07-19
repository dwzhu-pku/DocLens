#!/bin/bash

# -- SCRIPT CONFIGURATION --
NUM_GPUS=8
MAX_WORKERS=16

if [ -z "$1" ]; then
    echo "Please specify the dataset name as the first argument (e.g., 'MMLongBenchDoc' or 'FinRAGBench-V' or 'LongDocURL' or 'PaperTab' or 'FetaTab' or 'MMDocRAG' or 'DocBench')."
    exit 1
fi

if [[ "$1" == "MMLongBenchDoc" || "$1" == "FinRAGBench-V" || "$1" == "LongDocURL" || "$1" == "PaperTab" || "$1" == "FetaTab" || "$1" == "MMDocRAG" || "$1" == "DocBench" ]]; then
    DATASET_NAME="$1"
else
    echo "Error: Unknown dataset '$1'. Please use 'MMLongBenchDoc', 'FinRAGBench-V', 'LongDocURL', 'PaperTab', 'FetaTab', 'MMDocRAG', or 'DocBench'."
    exit 1
fi

SCRIPT_DIR=$(cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
DATASET_BASE_PATH="${PROJECT_ROOT}/data/${DATASET_NAME}/documents"


# -- SCRIPT EXECUTION --
echo "Starting document processing..."
echo "Dataset Path: ${DATASET_BASE_PATH}"
echo "Number of GPUs: ${NUM_GPUS}"
echo "Max Workers: ${MAX_WORKERS}"
echo "-------------------------------------"

python ${PROJECT_ROOT}/preprocess/doc_parse_parallel_mineru.py \
    --dataset_folder "${DATASET_BASE_PATH}" \
    --num_gpus ${NUM_GPUS} \
    --max_workers ${MAX_WORKERS}

echo "-------------------------------------"
echo "Script finished."