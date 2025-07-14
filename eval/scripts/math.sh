MODEL_PTH=$1
MY_DTYPE=$2

TOKENIZER_PTH="${MODEL_PTH}"
WORK_PTH="${MODEL_PTH}/0.math_$(date "+%Y-%m-%d_%H:%M:%S")"
mkdir "$WORK_PTH"

conda run --no-capture-output -n modelc \
python my_eval/eval_code/eval_math.py \
  --model_pth "$MODEL_PTH" \
  --tokenizer_pth "$TOKENIZER_PTH" \
  --tensor_parallel_size $PARALLEL_SIZE \
  --batch_size $BSZ \
  --save_gen_results_folder "$WORK_PTH" \
  --my_dtype "$MY_DTYPE" \
  >> "${WORK_PTH}/eval_log.txt" 2>&1

echo "Model: ${MODEL_PTH}"
echo "Log in: ${WORK_PTH}"