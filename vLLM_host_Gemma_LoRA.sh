#!/usr/bin/env bash

set -e

# source /mnt/storage5/users/ahboced1/tools/python_environments/vllm-0.25.1_cu130_python3.12.3_arm64/bin/activate

# these three exports are needed on the DGX Spark since those have a dynamic IP but on the normal server they are probably also a good idea
export VLLM_HOST_IP=127.0.0.1
export GLOO_SOCKET_IFNAME=lo
export NCCL_SOCKET_IFNAME=lo
export CUDA_VISIBLE_DEVICES=1

MODEL_NAME="Base_Gemma_Model"
MODEL_PATH=/mnt/storage5/users/ahfroet1/LLM_Depot/llm_models/Gemma-4-31B-it/

LORA_NAME="OncoClass_Gemma_LoRA_Adapter"
LORA_PATH=/mnt/storage5/users/ahfroet1/LLM_Depot/finetune_adapters/Gemma-4-31B-it-OncoClass-final/

LORA_RANK=64

# We set the max model length to 64k to prevent gpu memory issues and we usually don't use more context length
MAX_MODEL_LENGTH=64000
GPU_MEMORY_UTILIZATION=0.9
TENSOR_PARALLEL_SIZE=1

vllm serve $MODEL_PATH \
	--served-model-name $MODEL_NAME \
	--host 127.0.0.1 \
	--port 8002 \
	--max-num-seqs 1 \
	--tensor-parallel-size $TENSOR_PARALLEL_SIZE \
	--max-model-len $MAX_MODEL_LENGTH \
	--gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
	--tokenizer_mode auto \
	--config_format auto \
	--load_format auto \
	--dtype bfloat16 \
	--enable-lora \
	--lora-modules $LORA_NAME=$LORA_PATH \
	--max-lora-rank=$LORA_RANK \
	--seed 42
	
	
# need to adapt
	--lora-target-modules q_proj k_proj v_proj \
	--language-model-only \