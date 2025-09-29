MAX_NUM_FRAMES="512"
MODEL_NAME="streamforest"
TIME_MSG="short_online_v2"
REPLACE_PROJECTOR="ablation_woSTFW_PEMF"
CKPT_PATH="/your_local_path_to/StreamForest/ckpt/StreamForest-Qwen2-7B_Siglip_ablation_woPEMF+FSTW"

TASK="ovbench"

root_path="/your_local_path_to/StreamForest"
export PYTHONPATH=$root_path
export HF_DATASETS_OFFLINE=1
MASTER_PORT=$((18000 + $RANDOM % 100))
NUM_GPUS=8
CONV_TEMPLATE=qwen_2
TASK_SUFFIX="${TASK//,/_}"
mkdir ${CKPT_PATH}/eval
JOB_NAME=$(basename "$0" .sh)_$(date +"%Y%m%d_%H%M")

echo "检查点路径: $CKPT_PATH"
echo "最大帧数: $MAX_NUM_FRAMES"
echo "任务: $TASK"
echo "模型名称: $MODEL_NAME"
echo "提示词类型: $TIME_MSG"
echo "记忆类型: $REPLACE_PROJECTOR"


RESULT_DIR="${CKPT_PATH}/eval/${MAX_NUM_FRAMES}_${TASK}"

if [ ! -d "${RESULT_DIR}" ] && [ -d "${CKPT_PATH}" ]; then
  mkdir -p ${RESULT_DIR}
  echo "Created directory: ${RESULT_DIR}"
else
    echo "Directory ${RESULT_DIR} already exists or ${CKPT_PATH} not exists."
fi

srun -p videopp1 \
    --job-name=${JOB_NAME} \
    --ntasks=1 \
    --gres=gpu:8 \
    --ntasks-per-node=1 \
    --cpus-per-task=16 \
    --kill-on-bad-exit=1 \
    accelerate launch --num_processes ${NUM_GPUS} --main_process_port 10078 -m lmms_eval \
        --model ${MODEL_NAME} \
        --model_args pretrained=$CKPT_PATH,conv_template=$CONV_TEMPLATE,max_frames_num=$MAX_NUM_FRAMES,time_msg=$TIME_MSG,mm_projector_type=$REPLACE_PROJECTOR \
        --tasks $TASK \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix $TASK_SUFFIX \
        --output_path ${RESULT_DIR}/response__${TASK}_${MODEL_NAME}_F${MAX_NUM_FRAMES}_${JOB_NAME} \
        2>&1 | tee ${RESULT_DIR}/log_${TASK}_${MODEL_NAME}_F${MAX_NUM_FRAMES}_${JOB_NAME}.log
