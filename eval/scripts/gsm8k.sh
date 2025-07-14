MODEL_PTH=$1
MY_DTYPE=$2

TOKENIZER_PTH="${MODEL_PTH}"
WORK_PTH="${MODEL_PTH}/0.gsm8k_$(date "+%Y-%m-%d_%H:%M:%S")"
mkdir "$WORK_PTH"
# conda run --no-capture-output -n modelc \
python my_eval/eval_code/eval_gsm8k_new.py \
  --model_pth "$MODEL_PTH" \
  --tokenizer_pth "$TOKENIZER_PTH" \
  --save_gen_results_folder "$WORK_PTH" \
  --my_dtype "$MY_DTYPE" \
  >> "${WORK_PTH}/eval_log.txt" 2>&1

echo "model: ${MODEL_PTH}"
echo "Log in: ${WORK_PTH}"

# e.g. ts -G 1 bash my_eval/script/0-gsm8k.sh /data1/yany/optimal_merge fp16
# e.g. ts -G 2 bash my_eval/script/0-gsm8k.sh /data1/yany2/new_mask/math/dare_delta_32_magnitude_rescale bf16