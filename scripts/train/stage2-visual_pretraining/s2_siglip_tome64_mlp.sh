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

############### Pretrain ################
DATA_VERSION="anno/data_list/stage2_short_pretrain_iv6m.yaml"
DATA_VERSION_CLEAN=$(basename "$DATA_VERSION" .yaml)

VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN=$(basename "$VISION_MODEL_VERSION")

LLM_VERSION="Qwen/Qwen2-7B-Instruct"
LLM_VERSION_CLEAN=Qwen2-7B-Instruct

PROJECTOR_PATH="ckpt/stage1-init_connector/stage1-siglip-so400m-patch14-384-Qwen2-7B-Instruct-tome64_mlp-pretrain_sft_4debug.yaml_plain_20250812_222825/mm_projector.bin"

mm_projector_type=tome64_mlp

PROMPT_VERSION="qwen_2"

MID_RUN_NAME=stage2-${mm_projector_type}_${DATA_VERSION_CLEAN}_$(basename "$0" .sh)_$(date +"%Y%m%d_%H%M%S")
echo "MID_RUN_NAME: ${MID_RUN_NAME}"

PARTITION='video'
JOB_NAME=$(basename "$0" .sh)_$(date +"%Y%m%d_%H%M%S")

OUTPUT_DIR=ckpt/stage2-visual_pretraining/${MID_RUN_NAME}
mkdir -p ${OUTPUT_DIR}/runs

srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --ntasks=16 \
    --gres=gpu:8 \
    --preempt \
    --ntasks-per-node=8 \
    --cpus-per-task=16 \
    --kill-on-bad-exit=1 \
    python -u llava/train/train_mem.py \
    --deepspeed scripts/deepspeed/zero1.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path ${DATA_VERSION} \
    --vision_tower ${VISION_MODEL_VERSION} \
    --pretrain_mm_mlp_adapter ${PROJECTOR_PATH} \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --mm_vision_select_layer -2 \
    --mm_projector_type ${mm_projector_type} \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --bf16 True \
    --run_name $MID_RUN_NAME \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --attn_implementation sdpa \
    --frames_upbound 8 \
    --time_msg short \
    --local_num_frames 2 \
    --sample_type dynamic_fps1 \
    --vision_encode_type image_video_new \
    --mm_pos_num_frames 2 \
    --mm_num_compress_latents 128 \
    --mm_num_compress_query_type pooling \
    --mm_close_init True \
    --mm_local_num_frames 2 \
    2>&1 | tee ${OUTPUT_DIR}/runs/${MID_RUN_NAME}.log