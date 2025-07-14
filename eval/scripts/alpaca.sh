set -e

MODEL_NAME_OR_PATH=$1
TEMPLATE=$2 # llama2, llama3, mistral, openchat

# MODEL_NAME_OR_PATH=$(grep "^save_dir:" $CONFIG | awk '{print $2}')
# OUTPUT_DIR=$(grep "^log_dir:" $CONFIG | awk '{print $2}')
OUTPUT_DIR="$MODEL_NAME_OR_PATH"
BATCH_SIZE=512
GPT_ANNOTATOR="weighted_alpaca_eval_gpt-4o-2024-08-06"

OUTPUT_DIR=${OUTPUT_DIR//\"/}/alpaca_eval
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH//\"/}
echo "MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "GPT_ANNOTATOR=${GPT_ANNOTATOR}"

echo "Generating predictions..."
python eval/eval_alpacaeval.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --output_file $OUTPUT_DIR/file_upload.json \
    --batch_size $BATCH_SIZE \
    --template $TEMPLATE

echo "###### GPT-Annotator: $GPT_ANNOTATOR ######"
mkdir -p $OUTPUT_DIR/$GPT_ANNOTATOR
echo "log file will be saved in $OUTPUT_DIR/$GPT_ANNOTATOR/metrics.txt"
alpaca_eval --model_outputs $OUTPUT_DIR/file_upload.json \
    --annotators_config $GPT_ANNOTATOR \
    --caching_path $OUTPUT_DIR/$GPT_ANNOTATOR/annotations_cache.json > $OUTPUT_DIR/$GPT_ANNOTATOR/metrics.txt 2>&1
    # set caching_path to none to allow multiple runs at the same time

cat $OUTPUT_DIR/$GPT_ANNOTATOR/metrics.txt
echo "Metrics are saved in $OUTPUT_DIR/$GPT_ANNOTATOR/metrics.txt"
