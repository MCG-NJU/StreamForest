STREAMFOREST_ROOT_PATH="/your_local_path_to/StreamForest"
cd $STREAMFOREST_ROOT_PATH
export PYTHONPATH=$STREAMFOREST_ROOT_PATH

MAX_FRAMES=2048
TIME_MSG=short_online_v2
MODEL_NAME=streamforest
CKPT_PATH=ckpt/StreamForest-Qwen2-7B_Siglip
# CKPT_PATH=ckpt/StreamForest-Qwen2-7B_Siglip_drive

TASKS=(
    "odvbench"
    "streamingbench"
    "ovbench"
    "ovobench"
    "videomme"
    "mlvu_mc"
    "mvbench"
    "perceptiontest_val_mc"
)


for TASK in "${TASKS[@]}"; do
    echo "============================"
    echo "Running benchmark: $TASK"
    echo "============================"

    bash scripts/eval/online/eval_online_template.sh \
        --ckpt_path $CKPT_PATH \
        --max_frames $MAX_FRAMES \
        --model_name $MODEL_NAME \
        --time_msg $TIME_MSG \
        --task "$TASK"
done
