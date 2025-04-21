
#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
source "tests_extend/system_tests/env_npu.sh"

NPUS_PER_NODE=16
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

CKPT_DIR=./ckpt_llama
DATA_PATH="/home/dataset/llama2/alpaca_text_document"
TOKENIZER_MODEL="/home/dataset/model/llama-2-7b-hf/tokenizer.model"

TP=2
PP=4
EP=2

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

DUALPIPE_ARGS="
    --moe-unperm2-mem-optim-swap \
    --moe-fb-overlap \
    --schedules-method dualpipev \
    --moe-zero-memory level0 \
    --moe-tp-extend-ep \
"

MOE_ARGS="
    --expert-model-parallel-size ${EP} \
    --moe-token-dispatcher-type alltoall \
    --moe-permutation-async-comm \
    -use-fused-moe-token-permute-and-unpermute \
    --moe-grouped-gemm \
    --n-shared-experts 1 \
    --num-experts 32 \
    --moe-router-topk 8 \
    --moe-aux-loss-coeff 0.02 \
    --moe-tp-extend-ep \
    --recompute-norm \
    --recompute-activation-function \
"

GPT_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --use-flash-attn \
    --use-fused-rotary-pos-emb \
    --use-fused-swiglu \
    --use-fused-rmsnorm \
    --sequence-parallel \
    --use-distributed-optimizer \
    --num-layers 8 \
    --noop-layers 7 \
    --manual-gc \
    --manual-gc-interval 50 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --train-iters 50 \
    --hidden-size 5120 \
    --num-attention-heads 128 \
    --ffn-hidden-size 1536 \
    --make-vocab-size-divisible-by 128 \
    --vocab-size 126464 \
    --micro-batch-size 1 \
    --global-batch-size 32 \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --disable-bias-linear \
    --lr-decay-style linear \
    --lr-warmup-iters 0 \
    --short-seq-prob 0.0 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --untie-embeddings-and-output-weights \
    --init-method-std 0.006 \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --swiglu \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --no-gradient-accumulation-fusion \
    --bf16 \
    --group-query-attention \
    --num-query-groups 8 \
    --lr 2.0e-4 \
    --min-lr 2.0e-4 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --rotary-base 100000 \
    --norm-epsilon 1.0e-5 \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 995,5,0
"

OUTPUT_ARGS="
    --log-throughput \
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 10000 \
    --eval-iters 0 \
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    --npu-deterministic \
    $GPT_ARGS \
    $MOE_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $DUALPIPE_ARGS \
    2>&1 | tee -i base-deter-nofuse.log

