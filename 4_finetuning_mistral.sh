#!/usr/bin/env bash

mkdir -p /tmp/ahfroet1/huggingface/datasets
mkdir -p /tmp/ahfroet1/huggingface/hub
mkdir -p /tmp/ahfroet1/tmp
mkdir -p /tmp/ahfroet1/xdg-cache

export HF_HOME=/tmp/ahfroet1/huggingface
export HF_DATASETS_CACHE=/tmp/ahfroet1/huggingface/datasets
export HF_HUB_CACHE=/tmp/ahfroet1/huggingface/hub
export TMPDIR=/tmp/ahfroet1/tmp
export XDG_CACHE_HOME=/tmp/ahfroet1/xdg-cache

export CUDA_VISIBLE_DEVICES=0

torchrun \
	--standalone \
	--nnodes=1 \
	--nproc_per_node=1 \
	/mnt/storage5/users/ahfroet1/LLM_Depot/run_local_finetuning.py \
    --model /mnt/storage5/users/ahfroet1/LLM_Depot/llm_models/Mistral3.2-24B \
    --system_instruction /mnt/storage5/users/ahfroet1/OncoClass_AI/Prompt/System_Instruction_Finetuning.txt \
    --train /mnt/storage5/users/ahfroet1/OncoClass_AI/Input/OncoClass_train_final.json \
    --validation /mnt/storage5/users/ahfroet1/OncoClass_AI/Input/OncoClass_val_final.json \
    --output /mnt/storage5/users/ahfroet1/LLM_Depot/finetune_adapters/Mistral-3.2-24B-OncoClass-final \
    --epochs 4 \
    --learning_rate 2e-5
