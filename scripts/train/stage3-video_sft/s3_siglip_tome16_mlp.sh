#!/bin/bash
STREAMFOREST_ROOT_PATH="/your_local_path_to/StreamForest"
cd $STREAMFOREST_ROOT_PATH
export PYTHONPATH=$STREAMFOREST_ROOT_PATH
export OMP_NUM_THREADS=1
export DISABLE_ADDMM_CUDA_LT=1
export TORCH_CUDNN_USE_HEURISTIC_MODE_B=1
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_0,mlx5_2
export TRITON_CACHE_DIR="/tmp/triton3"
export NCCL_P2P_LEVEL=NVL
# export NCCL_DEBUG="INFO"
mkdir -p $TRITON_CACHE_DIR

############### Offline PT ################
DATA_VERSION="anno/data_list/stage3_short-long_mix_sft.yaml"    #Download from https://huggingface.co/datasets/MCG-NJU/StreamForest-Annodata/tree/main/data_list
DATA_VERSION_CLEAN=$(basename "$DATA_VERSION" .yaml)

VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN=$(basename "$VISION_MODEL_VERSION")

LLM_VERSION="ckpt/stage2-visual_pretraining/path_of_your_stage2_ckpt_here"  #your stage2 ckpt
LLM_VERSION_CLEAN=$(basename "$LLM_VERSION")

mm_projector_type=tome16_mlp

PROMPT_VERSION="qwen_2"

MID_RUN_NAME=stage3-${mm_projector_type}_${DATA_VERSION_CLEAN}_$(basename "$0" .sh)_$(date +"%Y%m%d_%H%M%S")
echo "MID_RUN_NAME: ${MID_RUN_NAME}"


PARTITION='video5'
JOB_NAME=$(basename "$0" .sh)_$(date +"%Y%m%d_%H%M%S")

OUTPUT_DIR=ckpt/stage3-video_sft/${MID_RUN_NAME}
mkdir -p ${OUTPUT_DIR}/runs

srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --ntasks=32 \
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
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --frames_upbound 256 \
    --frames_lowbound 64 \
    --time_msg short \
    --local_num_frames 8 \
    --vision_encode_type image_video_new \
    --sample_type dynamic_fps1 \
    --mm_pos_num_frames 8 \
    --mm_num_compress_latents 128 \
    --mm_num_compress_query_type pooling \
    --mm_close_init True \
    --mm_local_num_frames 8 \
    2>&1 | tee ${OUTPUT_DIR}/runs/${MID_RUN_NAME}.log
