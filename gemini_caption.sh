#!/bin/bash
DATA_PATH="/media02/nthuy/SnapUGC/SnapUGC_0"
JSON_PATH="/media02/nthuy/SnapUGC/SnapUGC_0/snapugc0_test.json"
OUTPUT_FNAME="snapugc0_test_gemini_rawcap.json"
MODEL_NAME="gemini-2.0-flash"
PROMPT="Please describe the video in detail, start with the description right away and do not include prefixes such as 'Here is the description of the video: '."

python gemini_caption.py \
    --data_path "$DATA_PATH" \
    --json_path "$JSON_PATH" \
    --output_path "$DATA_PATH" \
    --output_fname "$OUTPUT_FNAME" \
    --model_name "$MODEL_NAME" \
    --prompt "$PROMPT" \
    --logging_steps 100