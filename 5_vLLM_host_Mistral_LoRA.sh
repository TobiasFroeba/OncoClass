#!/usr/bin/env bash

set -e

# source /mnt/storage5/users/ahboced1/tools/python_environments/vllm-0.25.1_cu130_python3.12.3_amd64/bin/activate

# magic export variables needed..
export CUDA_HOME=/usr/local/cuda-13.0
export CUDA_PATH=$CUDA_HOME
export CUDACXX=$CUDA_HOME/bin/nvcc

export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# set version specific cache paths
mkdir -p /tmp/$USER/torch_extensions_pt211_cuda130_cache
mkdir -p /tmp/$USER/triton360_pt211_cuda130_cache
mkdir -p /tmp/$USER/vllm_0.25.1_cache

export TORCH_EXTENSIONS_DIR=/tmp/$USER/torch_extensions_pt211_cuda130_cache
export TRITON_CACHE_DIR=/tmp/$USER/triton360_pt211_cuda130_cache
export VLLM_CACHE_ROOT=/tmp/$USER/vllm_0.25.1_cache

# these three exports are needed on the DGX Spark since those have a dynamic IP but on the normal server they are probably also a good idea
export VLLM_HOST_IP=127.0.0.1
export GLOO_SOCKET_IFNAME=lo
export NCCL_SOCKET_IFNAME=lo
export CUDA_VISIBLE_DEVICES=0

MODEL_NAME="Base_Mistral_Model"
MODEL_PATH=/mnt/storage5/users/ahfroet1/LLM_Depot/llm_models/Mistral3.2-24B/

LORA_NAME="OncoClass_Mistral_LoRA_Adapter"
LORA_PATH=/mnt/storage5/users/ahfroet1/LLM_Depot/finetune_adapters/Mistral-3.2-24B-OncoClass-final/

LORA_RANK=64

# We set the max model length to 64k to prevent gpu memory issues and we usually don't use more context length
MAX_MODEL_LENGTH=64000
GPU_MEMORY_UTILIZATION=0.9
TENSOR_PARALLEL_SIZE=1

vllm serve $MODEL_PATH \
	--served-model-name $MODEL_NAME \
	--host 127.0.0.1 \
	--port 8001 \
	--max-num-seqs 1 \
	--tensor-parallel-size $TENSOR_PARALLEL_SIZE \
	--max-model-len $MAX_MODEL_LENGTH \
	--gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
	--tokenizer_mode mistral \
	--config_format mistral \
	--load_format mistral \
	--dtype bfloat16 \
	--enable-lora \
	--lora-modules $LORA_NAME=$LORA_PATH \
	--max-lora-rank=$LORA_RANK \
	--seed 42
