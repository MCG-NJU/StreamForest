#!/bin/bash
STREAMFOREST_ROOT_PATH="/your_local_path_to/StreamForest"
cd $STREAMFOREST_ROOT_PATH
export PYTHONPATH=$STREAMFOREST_ROOT_PATH
export OMP_NUM_THREADS=1
export DISABLE_ADDMM_CUDA_LT=1
export TORCH_CUDNN_USE_HEURISTIC_MODE_B=1
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_0,mlx5_2
# export NCCL_DEBUG="INFO"

############### Offline PT ################
DATA_VERSION="anno/data_list/stage1_init_connector_iv1m.yaml"       #Download from https://huggingface.co/datasets/MCG-NJU/StreamForest-Annodata/tree/main/data_list
DATA_VERSION_CLEAN=$(basename "$DATA_VERSION" .yaml)

VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN=$(basename "$VISION_MODEL_VERSION")

LLM_VERSION="Qwen/Qwen2-7B-Instruct"
LLM_VERSION_CLEAN=$(basename "$LLM_VERSION")

mm_projector_type=tome64_mlp
PROMPT_VERSION=plain

BASE_RUN_NAME=stage1-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-${mm_projector_type}-pretrain_${DATA_VERSION_CLEAN}_${PROMPT_VERSION}_$(date +"%Y%m%d_%H%M%S")
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

PARTITION='video'
JOB_NAME=$(basename "$0" .sh)_$(date +"%Y%m%d_%H%M%S")

OUTPUT_DIR=ckpt/stage1-init_connector/${BASE_RUN_NAME}
mkdir -p ${OUTPUT_DIR}/runs

srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --ntasks=8 \
    --gres=gpu:8 \
    --ntasks-per-node=8 \
    --cpus-per-task=16 \
    --kill-on-bad-exit=1 \
    python -u llava/train/train_mem.py \
    --deepspeed scripts/deepspeed/zero3.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path ${DATA_VERSION} \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_tunable_parts="mm_mlp_adapter" \
    --mm_vision_select_layer -2 \
    --mm_projector_type ${mm_projector_type} \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 50000 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --run_name $BASE_RUN_NAME \
    --attn_implementation sdpa \
    --frames_upbound 4 \
    --time_msg short \
    --local_num_frames 1 \
    --sample_type middle \
    --mm_local_num_frames 1 \
    2>&1 | tee ${OUTPUT_DIR}/runs/${MID_RUN_NAME}.log