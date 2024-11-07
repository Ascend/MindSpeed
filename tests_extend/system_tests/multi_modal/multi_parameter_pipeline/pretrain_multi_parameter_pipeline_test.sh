# 本测试脚本，只支持以下情况
# (1) world_size=1, 不开启并行特性
# (2) world_size=4, PP=4, 同时打开--use-multiparameter-pipeline-model-parallel开关， 进行PP测试
# (3) world_size=4, PP=4, 同时打开--use-multiparameter-pipeline-model-parallel \
# 与 --num-layers-per-virtual-pipeline-stage 1， 进行VPP测试
# 三者精度应当对齐

source "/MindSpeed/tests_extend/system_tests/env_npu.sh"

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=1
MASTER_ADDR=localhost
MASTER_PORT=6900
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CKPT_DIR=./ckpt_llama
TOKENIZER_MODEL=""
MINDSPEED_PATH=""

TP=1
PP=1
CP=1
EP=1

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --sequence-parallel \
    --use-distributed-optimizer \
    --num-layers 8 \
    --hidden-size 2048 \
    --ffn-hidden-size 28672 \
    --num-attention-heads 64 \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --micro-batch-size 2 \
    --global-batch-size 16 \
    --make-vocab-size-divisible-by 1 \
    --lr 1.0e-5 \
    --train-iters 1000 \
    --lr-decay-style cosine \
    --untie-embeddings-and-output-weights \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --normalization RMSNorm \
    --swiglu \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --no-gradient-accumulation-fusion \
    --min-lr 1.0e-5 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --initial-loss-scale 4096.0 \
    --adam-beta2 0.95 \
    --adam-eps 1e-5 \
    --disable-bias-linear \
    --group-query-attention \
    --num-query-groups 8 \
    --lr-warmup-fraction 0.01 \
    --mock-data
"

OUTPUT_ARGS="
    --log-throughput \
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 10000 \
    --eval-iters 1 \
"

torchrun $DISTRIBUTED_ARGS $MINDSPEED_PATH/tests_extend/system_tests/multi_modal/multi_parameter_pipeline/pretrain_multi_parameter_pipeline_test.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl

set +x