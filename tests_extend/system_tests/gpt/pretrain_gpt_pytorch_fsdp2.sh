#!/bin/bash

export ASCEND_LAUNCH_BLOCKING=1
export CUDA_DEVICE_MAX_CONNECTIONS=2

IPs=('$master_ip' '$other_ips')
IPS_STR="${IPs[*]}"

LOCAL_HOST=`hostname -I|awk -F " " '{print$1}'`
NPUS_PER_NODE=8
MASTER_ADDR=${IPs[0]}
MASTER_PORT=6088
NNODES=${#IPs[@]}
NODE_RANK=""

WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))
HIDDEN_SIZE=4096
((FFN_SIZE=4*${HIDDEN_SIZE}))
SEQ_LEN=1024

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

TOKENIZER_MODEL="/home/dataset/model/llama-2-7b-hf"
DATA_PATH="/home/dataset/llama2-7b-hf-eod/alpaca_text_document"

TP=2
PP=1
EP=1
CP=2
NUM_LAYERS=2

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --context-parallel-algo megatron_cp_algo \
    --sequence-parallel \
    --num-layers 8 \
    --hidden-size ${HIDDEN_SIZE} \
    --ffn-hidden-size ${FFN_SIZE} \
    --num-attention-heads 8 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_MODEL} \
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings ${SEQ_LEN} \
    --micro-batch-size 1 \
    --global-batch-size 8 \
    --make-vocab-size-divisible-by 1 \
    --lr 1.25e-6 \
    --train-iters 20 \
    --lr-decay-style cosine \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --use-fused-rmsnorm \
    --swiglu \
    --use-fused-swiglu \
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
    --transformer-impl local \
    --bf16 \
    --untie-embeddings-and-output-weights \
    --use-torch-fsdp2 \
    --fsdp2-config-path tests_extend/system_tests/gpt/fsdp2_config.yaml \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

OUTPUT_ARGS="
    --save output_save \
    --log-interval 1 \
    --save-interval 10 \
    --no-save-optim \
    --no-save-rng \
    --load output_save \
    --no-load-optim \
    --no-load-rng \
    --eval-interval 1000 \
    --eval-iters 0 \
"

PROFILE_ARGS="
    --profile \
    --profile-step-start 2 \
    --profile-step-end 3 \
    --profile-level level2 \
    --profile-with-cpu \
    --profile-with-stack \
    --profile-with-memory \
    --save-save-path ./profile_dir \
    --profile-ranks 0 \
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl | tee logs/torch_fsdp2.log

set +x
