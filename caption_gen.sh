#!/bin/bash
MODEL_PATH="/media02/nthuy/Thesis_engagement_expl/checkpoints/longvu_qwen"
DATA_PATH="/media02/nthuy/SnapUGC/SnapUGC_0"
JSON_PATH="/media02/nthuy/SnapUGC/SnapUGC_0/snapugc0_test.json"

torchrun --nproc_per_node=1 --master_port=29502 caption_gen.py \
    --model_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --json_path $JSON_PATH \
    --output_path $DATA_PATH \
    --ckpt_num_steps 10 \
    --per_device_batch_size 1 \
