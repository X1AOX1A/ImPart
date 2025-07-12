TASK=$1
DTYPE=$2
SIZE=$3
RATE=$4


mkdir "$WORKSPACE"
echo "$DELTA"

# e.g. 开始 Quant
echo "Start DeltaCoMe Quant"
export PYTHONPATH=/home/yany/0.project/delta-quant/gptq-delta
conda run --no-capture-output -n bowen \
python ./gptq-delta/llama.py \
  --model "$FTM" \
  --dataset c4 \
  --my_dtype "$DTYPE" \
  --config "$CONFIG"\
  --saved_delta_path "$DELTA" \
  --save_compressed_delta_dir "$WORKSPACE/832_quant.pt" \
  >> "$WORKSPACE/832_quant_log.txt" 2>&1

# Load quantized Delta Weight to pretrained model
echo "Load Quantized DeltaW"
conda run --no-capture-output -n modelc \
python ./gptq-delta/load_delta.py \
  --merge \
  --finetuned_model "$FTM" \
  --delta_path "$WORKSPACE/832_quant.pt" \
  --dtype "$DTYPE" \
  --save_path "$WORKSPACE" \
  >> "$WORKSPACE/load_delta_log.txt" 2>&1

