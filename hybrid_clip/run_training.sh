#!/bin/bash

SCRIPT_DIR=.
MODEL_DIR=~/models/from_frozen_backbones_checkpoint_2

IMAGE_ENCODER="openai/clip-vit-base-patch32"
TEXT_ENCODER="dbmdz/bert-base-italian-xxl-uncased"


python ${SCRIPT_DIR}/run_hybrid_clip.py \
    --output_dir ${MODEL_DIR} \
    --overwrite_output_dir \
    --tokenizer_name=${TEXT_ENCODER} \
    --train_file="../data/train_dataset_v4.json" \
    --validation_file="../data/valid_dataset_v4.json" \
    --do_train --do_eval \
    --num_train_epochs="17" --max_seq_length 96 \
    --per_device_train_batch_size="128" \
    --per_device_eval_batch_size="128" \
    --learning_rate="0.00001" --warmup_ratio 0.1 --weight_decay 0.0 \
    --preprocessing_num_workers 32 \
    --log_comet \
    --exp_name from_frozen_backbones_checkpoint_2 \
    --eval_when 1 \
    --text_model_name_or_path=${TEXT_ENCODER} \
    --vision_model_name_or_path=${IMAGE_ENCODER} \
    --run_from_checkpoint /home/giuseppe/models/frozen_backbones/50

#    --freeze_backbones    # freezes both models except from the reprojections layers
#    --run_from_checkpoint /home/giuseppe/models/cc_wit_coco_v2_2_resumed/20 \
#    --push_to_hub

