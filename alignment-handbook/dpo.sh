#!/bin/bash

# CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml  --main_process_port 7897 --num_processes 4 scripts/run_dpo.py recipes/biomistral_7b/dpo/config_full.yaml
# CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml  --main_process_port 7897 --num_processes 4 scripts/run_dpo.py recipes/mistral_7b/dpo/config_full.yaml
# CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml  --main_process_port 7897 --num_processes 4 scripts/run_dpo.py recipes/llama2_7b/dpo/config_full.yaml
# CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml  --main_process_port 7897 --num_processes 4 scripts/run_dpo.py recipes/meditron_7b/dpo/config_full.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml  --main_process_port 7897 --num_processes 4 scripts/run_dpo.py recipes/selfbiorag_7b/dpo/config_full.yaml