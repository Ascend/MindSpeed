#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))
CKPT_DIR=./ckpt
VOCAB_FILE=<Specify path to file>/vocab.json
MERGE_FILE=<Specify path to file>/merges.txt
DATA_PATH=<Specify path and file prefix>_text_document
TP=2
PP=2
CP=1
EP=1

DISTRIBUTED_ARGS="
    --master_addr $MASTER_ADDR \
    --node_rank $NODE_RANK \
    --worker_num $WORLD_SIZE \
    --local_worker_num $NPUS_PER_NODE \
    --master_port $MASTER_PORT \
    --log_dir=msrun_log \
    --join=False \
    --cluster_time_out=300 \
    --bind_core=True \
"
GPT_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --num-layers-per-virtual-pipeline-stage 1 \
    --num-layers 8 \
    --hidden-size 4096 \
    --ffn-hidden-size 14336 \
    --num-attention-heads 64 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --micro-batch-size 1 \
    --global-batch-size 16 \
    --make-vocab-size-divisible-by 1 \
    --lr 1.0e-6 \
    --train-iters 1000 \
    --init-method-std 0.01 \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --min-lr 1.0e-7 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --initial-loss-scale 4096.0 \
    --disable-bias-linear \
    --lr-warmup-fraction 0.01 \
    --fp16
"
DATA_ARGS="
    --split 990,5,5
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
"
OUTPUT_ARGS="
    --log-throughput \
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 10000 \
    --eval-iters 10 \
"
msrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --ai-franmework mindspore \
    --distributed-backend nccl \
    