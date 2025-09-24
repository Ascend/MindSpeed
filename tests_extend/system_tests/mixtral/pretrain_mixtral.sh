#!/bin/bash

source "tests_extend/system_tests/env_npu.sh"
export CUDA_DEVICE_MAX_CONNECTIONS=1

IPs=('$master_ip' '$other_ips')
LOCAL_HOST=`hostname -I|awk -F " " '{print$1}'`
NPUS_PER_NODE=8
MASTER_ADDR=${IPs[0]}
MASTER_PORT=6000
NNODES=${#IPs[@]}
NODE_RANK=""
for i in "${!IPs[@]}";
do
    echo "${IPs[$i]}"
    if [ "$LOCAL_HOST" == "${IPs[$i]}" ];
    then
        NODE_RANK=$i
        echo "============NODE_RANK:${NODE_RANK}============="
        break
    fi
done
if [[ $NODE_RANK == "" ]]; then
    echo "[Error] para \"NODE_RANK\" must be configured"
    exit 1
fi
WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

echo "NODE_RANK ${NODE_RANK}"

DATA_PATH="your data path"
TOKENIZER_MODEL="your tokenizer path"
CKPT_SAVE_DIR="your model save ckpt path"
CKPT_LOAD_DIR="your model ckpt path"


TP=2
PP=4
EP=2
NUM_LAYERS=32

# Set parameter moe-aux-loss-coeff to 0 when verifying accuracy
# Set moe-model-type to deepspeed_moe when train deepspeed moe
MOE_ARGS="
    --num-experts 8 \
    --expert-model-parallel-size ${EP} \
    --moe-router-topk 2 \
    --moe-router-load-balancing-type aux_loss \
    --moe-aux-loss-coeff 0.01 \
    --moe-train-capacity-factor 1.1 \
    --noisy-gate-policy RSample \
    --moe-model-type deepspeed_moe
"

GPT_ARGS="
    --use-legacy-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --sequence-parallel \
    --recompute-method block \
    --recompute-granularity full \
    --recompute-num-layers ${NUM_LAYERS} \
    --num-layers ${NUM_LAYERS} \
    --hidden-size 4096 \
    --ffn-hidden-size 14336 \
    --num-attention-heads 32 \
    --group-query-attention \
    --num-query-groups 8 \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --seq-length 4096 \
    --max-position-embeddings 32768 \
    --micro-batch-size 1 \
    --global-batch-size 128 \
    --make-vocab-size-divisible-by 1 \
    --lr 1.25e-6 \
    --train-iters 1000 \
    --lr-decay-style cosine \
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
    --weight-decay 1e-1 \
    --lr-warmup-fraction 0.01 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --initial-loss-scale 65536 \
    --adam-beta2 0.95 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --load ${CKPT_LOAD_DIR} \
    --save ${CKPT_SAVE_DIR} \
    --transformer-impl local \
    --bf16
"

DATA_ARGS="
    --data-path $DATA_PATH  \
    --split 100,0,0 \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 1000 \
    --eval-interval 1000 \
    --eval-iters 0 \
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
  $MOE_ARGS \
  $GPT_ARGS \
  $DATA_ARGS \
  $OUTPUT_ARGS \
  --distributed-backend nccl

set +x
