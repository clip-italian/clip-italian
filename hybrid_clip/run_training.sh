#!/bin/bash

SCRIPT_DIR=.
MODEL_DIR=~/models/cc_wit_coco_v2_resumed_from_20_on_12julynoon

IMAGE_ENCODER="openai/clip-vit-base-patch32"
TEXT_ENCODER="dbmdz/bert-base-italian-xxl-uncased"


python ${SCRIPT_DIR}/run_hybrid_clip.py \
    --output_dir ${MODEL_DIR} \
    --overwrite_output_dir \
    --tokenizer_name=${TEXT_ENCODER} \
    --train_file="../data/train_dataset_v2.json" \
    --validation_file="../data/valid_dataset_v2.json" \
    --do_train --do_eval \
    --num_train_epochs="80" --max_seq_length 96 \
    --per_device_train_batch_size="128" \
    --per_device_eval_batch_size="128" \
    --learning_rate="0.00008" --warmup_ratio 0.1 --weight_decay 0.0 \
    --preprocessing_num_workers 32 \
    --log_comet \
    --eval_when 10 \
    --text_model_name_or_path=${TEXT_ENCODER} \
    --vision_model_name_or_path=${IMAGE_ENCODER} # should be ignored
    
#    --run_from_checkpoint /home/giuseppe/models/cc_wit_coco_v2_2_resumed/20 \
#    --push_to_hub

