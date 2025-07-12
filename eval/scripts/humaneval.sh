MODEL_PTH=$1
MY_DTYPE=$2
TOKENIZER_PTH="${MODEL_PTH}"
WORK_PTH="${MODEL_PTH}/0.humaneval_$(date "+%Y-%m-%d_%H:%M:%S")"
mkdir "$WORK_PTH"
# conda run --no-capture-output -n "modelc" \
python my_eval/eval_code/humaneval.py \
  --model_pth "$MODEL_PTH" \
  --tokenizer_pth "$TOKENIZER_PTH" \
  --save_gen_results_folder "$WORK_PTH" \
  --my_dtype "$MY_DTYPE" \
  >> "${WORK_PTH}/eval_log.txt" 2>&1

evaluate_functional_correctness "${WORK_PTH}/processed.jsonl" >> "${WORK_PTH}/results.txt"
python my_eval/eval_code/eval_humaneval_new.py --input_jsonl "$WORK_PTH/gen.jsonl" >> "$WORK_PTH/extract_log.txt"
evaluate_functional_correctness "$WORK_PTH/processed_new.jsonl" >> "$WORK_PTH/results_new.txt"

# e.g. ts -G 1 bash my_eval/script/1-humaneval.sh /data1/yany2/MergeWithScale/metamath_distill_mse bf16
# e.g. ts -G 2 bash my_eval/script/1-humaneval.sh /data1/yany2/new_mask/math/dare_delta_32_magnitude_norescale bf16



