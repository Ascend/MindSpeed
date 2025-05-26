#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
source "tests_extend/system_tests/env_npu.sh"

export TASK_QUEUE_ENABLE=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export CLOSE_MATMUL_K_SHIFT=1


GPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=52824
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH="/home/dataset/llama2/alpaca_text_document"
TOKENIZER_MODEL="/home/dataset/model/llama-2-7b-hf/tokenizer.model"
TP=1
PP=1

declare -A ARGS
handle_arg() {
    local arg_name=$1
    local arg_value=$2
    ARGS[$arg_name]=$arg_value
}
layerzero_flag=true
ARGS["layerzero"]=${layerzero_flag}
for arg in "$@"; do
    if [[ $arg =~ ^--([a-zA-Z0-9_]+)=(.*)$ ]]; then
        handle_arg "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"
    else
        echo "Invalid argument: $arg"
    fi
done

ZERO_ARGS=""
if ${ARGS["layerzero"]}; then
    ZERO_ARGS="
        --layerzero \
        --layerzero-config layerzero.yml \
    "
    echo "using ZERO_ARGS"
fi

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --sequence-parallel \
    --num-layers 28 \
    --hidden-size 4096 \
    --ffn-hidden-size 11008 \
    --num-attention-heads 32 \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --micro-batch-size 1 \
    --global-batch-size 16 \
    --make-vocab-size-divisible-by 1 \
    --lr 1.25e-6 \
    --train-iters 2000 \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --use-fused-rmsnorm \
    --swiglu \
    --use-flash-attn \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --min-lr 1.25e-7 \
    --weight-decay 0.00 \
    --clip-grad 0.0 \
    --adam-beta1 0.9 \
    --initial-loss-scale 1 \
    --adam-beta2 0.95 \
    --use-fused-rotary-pos-emb \
    --no-gradient-accumulation-fusion \
    --use-distributed-optimizer \
    --fp16 \
    --no-load-optim \
    --no-load-rng \
    --transformer-impl local \
    --npu-deterministic \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 5 \
    --eval-interval 10000 \
    --eval-iters 0 \
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt_v2.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $ZERO_ARGS \
    --distributed-backend nccl

set + x
