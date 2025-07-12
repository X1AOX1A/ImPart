PARALLEL_SIZE=$1
BSZ=$2
# export CUDA_VISIBLE_DEVICES=$3
cuda=$3
MODEL_PTH=$4
MY_DTYPE=$5
WORK_PTH="${MODEL_PTH}/0.ifeval_$(date "+%Y-%m-%d_%H:%M:%S")"
mkdir "$WORK_PTH"

if [ "$MY_DTYPE" = "fp16" ]; then
    MY_DTYPE="float16"
else
    MY_DTYPE="bfloat16"
fi

conda run --no-capture-output -n update \
lm_eval --model vllm \
    --model_args pretrained="$MODEL_PTH",dtype="$MY_DTYPE",tensor_parallel_size="${PARALLEL_SIZE}" \
    --tasks ifeval \
    --batch_size "$BSZ" \
    --output_path "${WORK_PTH}" \
    --log_samples \
    >> "${WORK_PTH}/eval_inference_log.txt" 2>&1

#lm_eval --model vllm \
#    --model_args pretrained="/data1/yany/dare_col/chat_0.5",dtype="float16",tensor_parallel_size="1" \
#    --tasks ifeval \
#    --batch_size "512"
echo "model: ${MODEL_PTH}"
echo "Log in: ${LOG_PTH}"
# e.g. ts -G 1 bash my_eval/script/eval_ifeval.sh 1 512 0 /data/yany3/DARE_Quant/chat/13b/dare_1/dare_1_bitdelta fp16
# e.g. ts -G 4 bash my_eval/script/eval_ifeval.sh 4 512 0 /data1/yany2/new_mask/chat/dare_delta_32_magnitude_mag fp16
# e.g. ts -G 1 bash my_eval/script/eval_ifeval.sh 1 512 0/data/yany3/DARE_Quant/llama3/8b/dare_16/dare_quant_8bit bf16