#!/usr/bin/env bash

# source /mnt/storage5/users/ahboced1/tools/python_environments/vllm-0.25.1_cu130_python3.12.3_amd64/bin/activate

export CUDA_VISIBLE_DEVICES=1

torchrun \
	--standalone \
	--nnodes=1 \
	--nproc_per_node=1 \
	/mnt/storage5/users/ahfroet1/LLM_Depot/run_local_finetuning.py \
    --model /mnt/storage5/users/ahfroet1/LLM_Depot/llm_models/Gemma-4-31B-it \
    --system_instruction /mnt/storage5/users/ahfroet1/OncoClass_AI/Prompt/System_Instruction_Finetuning.txt \
    --train /mnt/storage5/users/ahfroet1/OncoClass_AI/Input/OncoClass_train_final.json \
    --validation /mnt/storage5/users/ahfroet1/OncoClass_AI/Input/OncoClass_val_final.json \
    --output /mnt/storage5/users/ahfroet1/LLM_Depot/finetune_adapters/Gemma-4-31B-it-OncoClass-final \
    --epochs 4 \
    --learning_rate 2e-5
