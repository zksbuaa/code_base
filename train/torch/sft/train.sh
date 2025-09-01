#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

MODEL=$1
METHOD=$2
DFT=$3

CONSOLE_LOG_PATH=/mnt/public/code/zks/DFT/logs/train/${MODEL}_1epoch/
RUNNING_LOG_PATH=/mnt/public/code/zks/DFT/logs/run/${MODEL}_1epoch/
OUTPUT_DIR=/mnt/public/code/zks/DFT/ckpt/${MODEL}_1epoch/${METHOD}

mkdir -p $CONSOLE_LOG_PATH
mkdir -p $RUNNING_LOG_PATH
mkdir -p $TENSORBOARD_LOG_PATH
mkdir -p $OUTPUT_DIR


deepspeed --num_gpus=8 train.py \
    --model_name_or_path /mnt/public/model/zks/${MODEL} \
    --torch_dtype bfloat16 \
    --bf16 True \
    --trust_remote_code True \
    --dataset_path /mnt/public/data/zks/NuminaMath-CoT \
    --preprocessing_num_workers 16 \
    --enable_gradient_checkpointing True \
    --reduce_logging True \
    --use_dft ${DFT} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 5e-5 \
    --lr_scheduler_type cosine \
    --weight_decay 0.01 \
    --gradient_accumulation_steps 10 \
    --train_sample_num 800000 \
    --eval_strategy steps \
    --eval_steps 200 \
    --save_strategy steps \
    --save_only_model 1 \
    --save_steps 1000 \
    --logging_steps 10 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --output_dir ${OUTPUT_DIR} \
    --dist_logger ${RUNNING_LOG_PATH}/${METHOD} \
    --save_safetensors True \
    --remove_unused_columns False \
    --disable_tqdm False \
    --sample_seed 819 \
    --report_to none \
    2>&1 | tee ${CONSOLE_LOG_PATH}/${METHOD}.log