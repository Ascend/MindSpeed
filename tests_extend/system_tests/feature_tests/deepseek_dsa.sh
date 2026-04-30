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

CKPT_DIR=./ckpt_llama
TOKENIZER_MODEL="/home/dataset/model/llama-2-7b-hf/tokenizer.model"
VOCAB_FILE=/home/dataset/enwiki/gpt2-vocab.json
MERGE_FILE=/home/dataset/enwiki/gpt2-merges.txt
DATA_PATH=/home/dataset/enwiki/my-t5_text_sentence

TP=2
PP=2
CP=2

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

RECOMPUTE_ARGS="
    --recompute-granularity full \
    --recompute-method block \
    --recompute-num-layers 1 \
"

DSA_ARGS="
    --multi-latent-attention \
    --experimental-attention-variant dsa \
    --dsa-indexer-n-heads 64 \
    --dsa-indexer-head-dim 128 \
    --dsa-indexer-topk 2048 \
    --qk-layernorm \
    --qk-head-dim 128 \
    --qk-pos-emb-head-dim 64 \
    --q-lora-rank 1536 \
    --kv-lora-rank 512 \
    --v-head-dim 128 \
    --dsa-indexer-loss-coeff 1 \
    --dsa-indexer-use-sparse-loss \
    --use-dsa-absorb \
    --use-fused-lightning-indexer \
    --use-fused-sparse-flash-attention \
    --use-fused-lightning-indexer-kl-loss \
"

GPT_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --context-parallel-algo kvallgather_cp_algo \
    --use-flash-attn \
    --use-fused-rotary-pos-emb \
    --sequence-parallel \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --num-layers 2 \
    --hidden-size 6144 \
    --num-attention-heads 64 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --train-iters 2000 \
    --micro-batch-size 1 \
    --global-batch-size 4 \
    --lr 5.0e-7 \
    --lr-decay-style cosine \
    --lr-decay-iters 32000 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.006 \
    --position-embedding-type rope \
    --disable-bias-linear \
    --no-bias-gelu-fusion \
    --no-bias-dropout-fusion \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --npu-deterministic \
    --normalization RMSNorm \
    --bf16 \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --vocab-size 50257 \
    --num-workers 4 \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-throughput \
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 10000 \
    --eval-iters 10 \
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $RECOMPUTE_ARGS \
    $DSA_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \

set +x
