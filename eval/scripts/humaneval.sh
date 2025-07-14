MODEL_PTH=$1
MY_DTYPE=$2

TOKENIZER_PTH="${MODEL_PTH}"
WORK_PTH="${MODEL_PTH}/0.humaneval_$(date "+%Y-%m-%d_%H:%M:%S")"
mkdir "$WORK_PTH"

python eval/eval_humaneval.py \
  --model_pth "$MODEL_PTH" \
  --tokenizer_pth "$TOKENIZER_PTH" \
  --save_gen_results_folder "$WORK_PTH" \
  --my_dtype "$MY_DTYPE" \
  >> "${WORK_PTH}/eval_log.txt" 2>&1

evaluate_functional_correctness "${WORK_PTH}/processed.jsonl" >> "${WORK_PTH}/results.txt"
python eval/extract_humaneval.py --input_jsonl "$WORK_PTH/gen.jsonl" >> "$WORK_PTH/extract_log.txt"
evaluate_functional_correctness "$WORK_PTH/processed_new.jsonl" >> "$WORK_PTH/results_new.txt"



