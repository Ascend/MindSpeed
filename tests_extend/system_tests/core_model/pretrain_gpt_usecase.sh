#!/bin/bash
# entry file:  pretrain_gpt_usecase.py
echo "------ARGUMENTS LIST --------"
for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
   echo "$KEY=$VALUE"
done
echo "---------------------------------"


set -exo pipefail
if [[ -z $TRAINING_DTYPE ]]; then TRAINING_DTYPE=bf16 ; fi
if [[ "${USE_MCORE}" == 'False' ]]; then
       echo "Running using megatron core"
       USE_LEGACY=1
fi

if [[ -z $EXTRA_ARGS ]]; then
  echo "no additional params"
else
  ADDITIONAL_PARAMS=$EXTRA_ARGS ;
fi

if [[ "${TRAINING_DTYPE}" == "fp16" ]]; then
      ADDITIONAL_PARAMS+=" --apply-query-key-layer-scaling"
fi

if [[ $CHECKPOINT_RESUME_TEST -eq 1 ]]; then
       echo "Running checkpoint resume test..."
       __SAVE_INTERVAL=20
       ADDITIONAL_PARAMS+=" --use-checkpoint-opt_param-scheduler"
       if [[ $MAX_STEPS -ne 50 ]]; then
         echo "Overriding MAX_STEPS=50"
         MAX_STEPS=50
       fi
else
       __SAVE_INTERVAL=${SAVE_INTERVAL:-10000}  # inf
       if [[ -z $CHECKPOINT_PATH ]]; then
         echo "The checkpoint_path is not set. Please set it."
         exit
        else
          echo "rm -rf ${CHECKPOINT_PATH}"
          rm -rf ${CHECKPOINT_PATH}
      fi
fi


set +x

echo "use_legacy: ${USE_LEGACY}, ${USE_LEGACY:+--use-legacy-models}"
echo "vp_size: ${VP_SIZE}, ${VP_SIZE:+--num-layers-per-virtual-pipeline-stage "$VP_SIZE"}"
echo "add_params : ${ADDITIONAL_PARAMS}, ${ADDITIONAL_PARAMS:+$ADDITIONAL_PARAMS}"

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

export CUDA_DEVICE_MAX_CONNECTIONS=1


DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"


torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
       --num-layers 8 \
       --hidden-size 512 \
       --num-attention-heads 8 \
       --log-params-norm \
       --log-num-zeros-in-grad \
       --micro-batch-size ${MBS:-4} \
       --global-batch-size ${GBS:-32} \
       --seq-length ${SEQ_LEN:-1024} \
       --max-position-embeddings ${SEQ_LEN:-1024} \
       --train-iters ${MAX_STEPS:-50} \
       --timing-log-level 2 \
       --save ${CHECKPOINT_PATH} \
       --load ${CHECKPOINT_PATH} \
       --data-path ${DATA_PATH} \
       --vocab-file ${VOCAB_FILE} \
       --merge-file ${MERGE_FILE} \
       --split 100,0,0 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction 0.01 \
       --log-interval 1 \
       --save-interval $__SAVE_INTERVAL \
       --eval-interval 15 \
       --eval-iters 0 \
       --transformer-impl local \
       --tensor-model-parallel-size ${TP_SIZE} \
       --pipeline-model-parallel-size ${PP_SIZE} \
       --no-bias-swiglu-fusion \
       --no-gradient-accumulation-fusion \
       ${VP_SIZE:+--num-layers-per-virtual-pipeline-stage "$VP_SIZE"} \
       ${ADDITIONAL_PARAMS:+$ADDITIONAL_PARAMS} \
       ${USE_LEGACY:+--use-legacy-models} \
       --${TRAINING_DTYPE} 

