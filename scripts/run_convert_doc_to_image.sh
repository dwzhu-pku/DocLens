#!/bin/bash

# -- SCRIPT CONFIGURATION --
IMAGE_DPI=200
IMAGE_FORMAT="jpeg"

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
PDF_SOURCE_DIRECTORY="${PROJECT_ROOT}/data/${DATASET_NAME}/documents"


# -- SCRIPT EXECUTION --
echo "Starting PDF to image conversion process..."
echo "Source Directory: ${PDF_SOURCE_DIRECTORY}"
echo "Image DPI: ${IMAGE_DPI}"
echo "Output Format: ${IMAGE_FORMAT}"
echo "-------------------------------------"

python ${PROJECT_ROOT}/preprocess/pdf_to_images.py \
    --pdf_directory "${PDF_SOURCE_DIRECTORY}" \
    --dpi ${IMAGE_DPI} \
    --format ${IMAGE_FORMAT}

echo "-------------------------------------"
echo "Conversion script finished."