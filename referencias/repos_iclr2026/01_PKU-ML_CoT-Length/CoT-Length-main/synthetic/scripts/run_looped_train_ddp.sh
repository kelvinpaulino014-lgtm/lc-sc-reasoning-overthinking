#!/bin/bash

# Simple 2-GPU DDP training for looped GPT-2

# Optional: restrict to 2 GPUs
# export CUDA_VISIBLE_DEVICES=0,1

MODEL_SIZE=${MODEL_SIZE:-6}
N_LOOP=${N_LOOP:-9}
T_VAL=${T_VAL:-80}
t_val=${t_val:-12}
ITERS=${ITERS:-25000}
BATCH_SIZE=${BATCH_SIZE:-256} # interpreted as GLOBAL batch size across GPUs

echo "Launching DDP training on 2 GPUs: model_size=${MODEL_SIZE}, n_loop=${N_LOOP}, T=${T_VAL}, t=${t_val}, global_batch_size=${BATCH_SIZE}"

torchrun --standalone --nproc_per_node=2 synthetic/train.py \
  --model_size=${MODEL_SIZE} \
  --device='cuda:0' \
  --iter=${ITERS} \
  --T=${T_VAL} \
  --t=${t_val} \
  --model_type=looped_gpt2 \
  --n_loop=${N_LOOP} \
  --batch_size=${BATCH_SIZE}
