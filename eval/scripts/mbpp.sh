MODEL_PTH=$1
MY_DTYPE=$2

WORK_PTH="${MODEL_PTH}/0.mbpp_evalplus_$(date "+%Y-%m-%d_%H:%M:%S")"
mkdir "$WORK_PTH"

conda info --envs | grep '*' >> "$WORK_PTH/eval_log.txt"

evalplus.evaluate \
  --model "$MODEL_PTH" \
  --dataset mbpp \
  --backend vllm \
  --i_just_wanna_run True \
  --output_file "$WORK_PTH/gen.jsonl" \
  --greedy \
  >> "$WORK_PTH/eval_log.txt"

echo "model: ${MODEL_PTH}"
echo "Log in: ${WORK_PTH}"