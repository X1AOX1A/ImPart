MODEL_PTH=$1
MY_DTYPE=$2
WORK_PTH="${MODEL_PTH}/0.ifeval_$(date "+%Y-%m-%d_%H:%M:%S")"
mkdir "$WORK_PTH"

if [ "$MY_DTYPE" = "fp16" ]; then
    MY_DTYPE="float16"
else
    MY_DTYPE="bfloat16"
fi

conda run --no-capture-output -n update \
lm_eval --model vllm \
    --model_args pretrained="$MODEL_PTH",dtype="$MY_DTYPE",tensor_parallel_size=2 \
    --tasks ifeval \
    --batch_size 512 \
    --output_path "${WORK_PTH}" \
    --log_samples \
    >> "${WORK_PTH}/eval_log.txt" 2>&1

echo "model: ${MODEL_PTH}"
echo "Log in: ${LOG_PTH}"