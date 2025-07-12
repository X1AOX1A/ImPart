TASK=$1
DTYPE=$2
SIZE=$3
RATE=$4

if [ "$TASK" == "chat" ]; then
  if [ "$SIZE" == "13b" ]; then
    WORKSPACE="/data/yany3/Quant/llamachat_13b/deltacome_${RATE}"
    FTM="/nfsdata/yany/download/merge/from_llama2_13b/meta-llama/Llama-2-13b-chat-hf"
    CONFIG="/home/yany/0.project/delta-quant/gptq-delta/config/used/chat_13b.yaml"
    if [ "$RATE" == "16" ]; then
      IDX="36"
    elif [ "$RATE" == "64" ]; then
      IDX="37"
    elif [ "$RATE" == "128" ]; then
      IDX="38"
    fi
      DELTA="/nfsdata/lyx/CACHE/svd_drop/saves/ckpt/direct_mask/direct_mask_1.3_llama2_chat_13b/config${IDX}/svd_delta.pt"
    if [ "$RATE" == "32" ]; then
      DELTA="/nfsdata/lyx/CACHE/svd_drop/saves/ckpt/direct_mask/direct_mask_1.3_llama2_chat_13b/config11/svd_delta.pt"
    fi
  else
    WORKSPACE="/data/yany3/Quant/llamachat_7b/deltacome"
    FTM="/nfsdata/yany/download/merge/from_llama2_7b/meta-llama/Llama-2-7b-chat-hf"
    CONFIG="/home/yany/0.project/delta-quant/gptq-delta/config/used/chat_7b.yaml"
    DELTA="/data1/lyx/CACHE/svd_drop/saves/ckpt/direct_mask/direct_mask_1.3_llama2_chat_7b/config11/svd_delta.pt"
  fi
elif [ "$TASK" == "math" ]; then
  if [ "$SIZE" == "13b" ]; then
    WORKSPACE="/data/yany3/Quant/wizardmath_13b/deltacome_${RATE}"
    FTM="/data1/yany/download/merge/from_llama2_13b/WizardMath-13B-V1.0"
    CONFIG="/home/yany/0.project/delta-quant/gptq-delta/config/used/math_13b.yaml"
    if [ "$RATE" == "32" ]; then
      IDX="3"
    elif [ "$RATE" == "48" ]; then
      IDX="6"
    elif [ "$RATE" == "64" ]; then
      IDX="9"
    elif [ "$RATE" == "128" ]; then
      IDX="12"
    else
      IDX="nonono"
    fi
    DELTA="/data1/lyx/CACHE/svd_drop/saves/ckpt/direct_mask/direct_mask_1.3/config${IDX}/svd_delta.pt"
    if [ "$RATE" == "16" ]; then
      DELTA="/data/lyx/CACHE/svd_drop/saves/ckpt/direct_mask/direct_mask_1.3/config59/svd_delta.pt"
    fi
  else
    WORKSPACE="/data/yany3/Quant/wizardmath_7b/deltacome"
    FTM="/nfsdata/yany/download/merge/from_llama2_7b/WizardLMTeam/WizardMath-7B-V1.0"
    CONFIG="/home/yany/0.project/delta-quant/gptq-delta/config/used/math_7b.yaml"
    DELTA="/data1/lyx/CACHE/svd_drop/saves/ckpt/direct_mask/direct_mask_1.3_wizardmath_7b/config14/svd_delta.pt"
  fi

elif [ "$TASK" == "code" ]; then
  if [ "$SIZE" == "13b" ]; then
    WORKSPACE="/data/yany3/Quant/wizardcoder_13b/deltacome_${RATE}"
    FTM="/data1/yany/download/merge/from_codellama_13b/WizardLMTeam/WizardCoder-Python-13B-V1.0"
    CONFIG="/home/yany/0.project/delta-quant/gptq-delta/config/used/code_13b.yaml"
    if [ "$RATE" == "16" ]; then
      IDX="28"
    elif [ "$RATE" == "32" ]; then
      IDX="11"
    elif [ "$RATE" == "64" ]; then
      IDX="29"
    elif [ "$RATE" == "128" ]; then
      IDX="30"
    else
      IDX="nonono"
    fi
    #DELTA="/data1/lyx/CACHE/svd_drop/saves/ckpt/direct_mask/direct_mask_1.3_wizardcoder_13b/config${IDX}/svd_delta.pt"
    DELTA="/nfsdata/lyx/CACHE/svd_drop/saves/ckpt/direct_mask/direct_mask_1.3_wizardcoder_13b/config${IDX}/svd_delta.pt"
  else
    WORKSPACE="/data/yany3/Quant/magicoder_7b/deltacome"
    FTM="/nfsdata/yany/download/merge/from_codellama_7b/ise-uiuc/Magicoder-S-CL-7B"
    CONFIG="/home/yany/0.project/delta-quant/gptq-delta/config/used/code_7b.yaml"
    DELTA="/data1/lyx/CACHE/svd_drop/saves/ckpt/direct_mask/direct_mask_1.3_magicoder_7b/config11/svd_delta.pt"
  fi
elif [ "$TASK" == "code1" ]; then
  if [ "$SIZE" == "13b" ]; then
    WORKSPACE="/data/yany3/Quant/magicoder_13b/deltacome_${RATE}"
    FTM="/nfsdata/yany/download/merge/from_llama2_13b/code/magicoder-llama-2-13b-fp16"
    CONFIG="/home/yany/0.project/delta-quant/gptq-delta/config/used/code1_13b.yaml"
    if [ "$RATE" == "16" ]; then
      IDX="26"
    elif [ "$RATE" == "64" ]; then
      IDX="27"
    elif [ "$RATE" == "128" ]; then
      IDX="28"
    else
      IDX="nonono"
    fi
    DELTA="/nfsdata/lyx/CACHE/svd_drop/saves/ckpt/direct_mask/direct_mask_1.3_magicoder_13b/config${IDX}/svd_delta.pt"
    if [ "$RATE" == "32" ]; then
      DELTA="/data/lyx/CACHE/svd_drop/saves/ckpt/direct_mask/direct_mask_1.3_magicoder_13b/config7/svd_delta.pt"
    fi
  else
    WORKSPACE=""
    FTM=""
  fi
else
    WORKSPACE=""
    FTM=""
    echo "Missing Task"
fi
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

# e.g. 加载 Quant 后的 DeltaW
echo "Load Quantized DeltaW"
conda run --no-capture-output -n modelc \
python ./gptq-delta/load_delta.py \
  --merge \
  --finetuned_model "$FTM" \
  --delta_path "$WORKSPACE/832_quant.pt" \
  --dtype "$DTYPE" \
  --save_path "$WORKSPACE" \
  >> "$WORKSPACE/load_delta_log.txt" 2>&1

#echo "Start Evaluating"
#cd /home/yany/0.project/EvalLLM/ || exit 1
#if [ "$TASK" == "chat" ]; then
#  # e.g. 测试 IFEval
#  ts -G 1 bash my_eval/script/eval_ifeval.sh 1 512 0 "$WORKSPACE" "$DTYPE"
#elif [ "$TASK" == "math" ]; then
#  # e.g. 测试 GSM8K
#  ts -G 1 bash my_eval/script/eval_gsm8k.sh 1 512 0 "$WORKSPACE" "$DTYPE"
#  # e.g. 测试 MATH
#  ts -G 1 bash my_eval/script/eval_math.sh 1 512 0 "$WORKSPACE" "$DTYPE"
#elif [ "$TASK" == "code" ]; then
#  # e.g. 测试 HumanEval
#  ts -G 1 bash my_eval/script/eval_humaneval.sh 1 512 0 "$WORKSPACE" "$DTYPE" WizardCoder
#  # e.g. 测试 MBPP
#  ts -G 1 bash my_eval/script/eval_mbpp.sh 1 512 0 "$WORKSPACE" "$DTYPE" WizardCoder
#else
#    echo "Missing Task"
#fi
# e.g. ts -G 1 bash gptq-delta/yy/scripts/run_deltacome.sh math bf16 13b 16
# e.g. ts -G 1 bash gptq-delta/yy/scripts/run_deltacome.sh code1 fp16 13b 16
# e.g. ts -G 1 bash gptq-delta/yy/scripts/run_deltacome.sh code fp16 13b 16
