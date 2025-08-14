#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
source "tests_extend/system_tests/env_npu.sh"
export STREAMS_PER_DEVICE=32

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))


ITER=10
CKPT_FORMAT="torch_dist"
SAVE_INTERVAL=$((ITER/2))
CKPT_DIR=./ckpt_llama
DATA_PATH="/home/dataset/llama2/alpaca_text_document"
TOKENIZER_MODEL="/home/dataset/model/llama-2-7b-hf/tokenizer.model"

TP=2
PP=2
EP=2
CP=2

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

MOE_ARGS_1="
    --num-experts 4 \
    --moe-token-dispatcher-type alltoall_seq \
    --disable-bias-linear \
    --sequence-parallel \
    --expert-model-parallel-size ${EP} \
    --moe-grouped-gemm \
    --bf16\
    --moe-router-topk 2 \
    --moe-aux-loss-coeff 0.02 \
    --moe-expert-capacity-factor 2 \
"

MOE_ARGS_2="
    --num-experts 4 \
    --moe-token-dispatcher-type allgather \
    --disable-bias-linear \
    --sequence-parallel \
    --expert-model-parallel-size ${EP} \
    --moe-grouped-gemm \
    --bf16\
    --moe-router-topk 2 \
    --moe-aux-loss-coeff 0.02 \
    --moe-expert-capacity-factor 2 \
"

GPT_ARGS="
    --num-layers-per-virtual-pipeline-stage 1 \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --npu-deterministic \
    --transformer-impl local \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --no-gradient-accumulation-fusion \
    --lr 1.0e-6 \
    --train-iters $ITER \
    --micro-batch-size 1 \
    --global-batch-size 16 \
    --max-position-embeddings 1024 \
    --seq-length 1024 \
    --num-attention-heads 8 \
    --hidden-size 3072 \
    --num-layers 4 \
    --log-throughput \
    --log-interval 1 \
    --save-interval $SAVE_INTERVAL \
    --eval-interval 10000 \
    --eval-iters 10 \
    --data-path $DATA_PATH \
    --split 100,0,0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.9 \
    --adam-eps 1e-5 \
    --add-qkv-bias \
    --attention-dropout 0.1 \
    --attention-softmax-in-fp32 \
    --clip-grad 1.0 \
    --distributed-timeout-minutes 10 \
    --ffn-hidden-size 3072 \
    --hidden-dropout 0.0 \
    --init-method-std 0.01 \
    --lr 5.0e-7 \
    --lr-decay-style cosine \
    --lr-warmup-fraction 0.01 \
    --min-lr 1e-8 \
    --make-vocab-size-divisible-by 1 \
    --manual-gc \
    --manual-gc-interval 50 \
    --no-bias-dropout-fusion \
    --no-bias-gelu-fusion \
    --no-bias-swiglu-fusion \
    --no-gradient-accumulation-fusion \
    --no-masked-softmax-fusion \
    --normalization RMSNorm \
    --norm-epsilon 1.0e-5 \
    --no-rope-fusion \
    --num-query-groups 8 \
    --num-workers 4 \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --use-distributed-optimizer \
    --position-embedding-type rope \
    --seed 1234 \
    --rotary-base 100000 \
    --use-cpu-initialization \
    --untie-embeddings-and-output-weights \
    --use-rotary-position-embeddings \
    --weight-decay 1e-2 \
    --end-weight-decay 1e-2 \
    --start-weight-decay 1e-3 \
    --weight-decay-incr-style linear \
    --initial-loss-scale 65536 \
    --exit-on-missing-checkpoint \
    --group-query-attention \
    --auto-detect-ckpt-format \
    --reset-attention-mask \
    --context-parallel-size ${CP} \
    --use-flash-attn \
    --attention-mask-type general \
    --distributed-backend nccl \
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    --transformer-impl local \
    --npu-deterministic \
    --ckpt-format ${CKPT_FORMAT} \
    --save ${CKPT_DIR} \
    --exit-interval ${SAVE_INTERVAL} \
    $MOE_ARGS_1 \
    $GPT_ARGS

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    --transformer-impl local \
    --npu-deterministic \
    --ckpt-format ${CKPT_FORMAT} \
    --load ${CKPT_DIR} \
    $MOE_ARGS_1 \
    $GPT_ARGS

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    --transformer-impl local \
    --npu-deterministic \
    --ckpt-format ${CKPT_FORMAT} \
    --save ${CKPT_DIR} \
    --exit-interval ${SAVE_INTERVAL} \
    $MOE_ARGS_2 \
    $GPT_ARGS

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    --transformer-impl local \
    --npu-deterministic \
    --ckpt-format ${CKPT_FORMAT} \
    --load ${CKPT_DIR} \
    $MOE_ARGS_2 \
    $GPT_ARGS