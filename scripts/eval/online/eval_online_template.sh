CKPT_PATH=""
MAX_NUM_FRAMES=""
TASK=""
MODEL_NAME=""
TIME_MSG="short_online"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt_path) CKPT_PATH="$2"; shift 2 ;;
    --max_frames) MAX_NUM_FRAMES="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --model_name) MODEL_NAME="$2"; shift 2 ;;
    --time_msg) TIME_MSG="$2"; shift 2 ;;  # 新增 time_msg 参数
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done



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

srun -p videop1 \
    --job-name=${JOB_NAME} \
    --ntasks=1 \
    --gres=gpu:${NUM_GPUS} \
    --ntasks-per-node=1 \
    --cpus-per-task=16 \
    --kill-on-bad-exit=1 \
    accelerate launch --num_processes ${NUM_GPUS} --main_process_port 10078 -m lmms_eval \
        --model ${MODEL_NAME} \
        --model_args pretrained=$CKPT_PATH,conv_template=$CONV_TEMPLATE,max_frames_num=$MAX_NUM_FRAMES,time_msg=$TIME_MSG \
        --tasks $TASK \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix $TASK_SUFFIX \
        --output_path ${CKPT_PATH}/eval/response__${TASK}_${MODEL_NAME}_F${MAX_NUM_FRAMES}_${JOB_NAME} \
        2>&1 | tee ${CKPT_PATH}/eval/log_${TASK}_${MODEL_NAME}_F${MAX_NUM_FRAMES}_${JOB_NAME}.log



