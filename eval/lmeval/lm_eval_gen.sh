#!/bin/bash
set -exuo pipefail
exec > >(tee -a /mnt/public2/code/zks/tmp/output.log) 2>&1

# ─── 参数设置 ─────────────────────────────────────────────────────────────────────

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <model_path_1> [<model_path_2> ...]"
    exit 1
fi

MODEL_PATHS=("$@")

#CONVERT_SCRIPT="/mnt/public/code/ssc/dynamic_train/Megatron-DeepSpeed-dynamic/tools/convert_checkpoint/deepspeed_ckpt_qwen2.5_to_hf.py"

# ─── 模型转换 ─────────────────────────────────────────────────────────────────────

# python $CONVERT_SCRIPT --input $ORIGINAL_MODEL --output $HF_MODEL

# ─── 环境准备 ─────────────────────────────────────────────────────────────────────

export HF_ENDPOINT=https://hf-mirror.com
export HUGGING_FACE_HUB_TOKEN=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
export HF_TOKEN=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
export PATH=/opt/conda/bin:/opt/conda/condabin:$PATH
# ln -s /opt/maca/tools/cu-bridge/bin/cucc /opt/maca/tools/cu-bridge/bin/nvcc


cd /mnt/public/code/data_mixture/lm-evaluation-harness
# pip install peft
pip install transformers==4.51.3
pip install -e .[math]
pip install langdetect immutabledict --no-deps

export HF_ALLOW_CODE_EVAL=1

# ─── 模型评估 ─────────────────────────────────────────────────────────────────────
TASK="agieval_gaokao_mathcloze,agieval_math,gsm8k,gsm8k_cot,humaneval,humaneval_64,mbpp"
OUTPUT_DIR="/mnt/public/code/zks/lmeval_res/sft_ckpt/3B_gen/0615"
# LOG_FILE="/mnt/public/code/zks/eval_log.txt"

for model_path in "${MODEL_PATHS[@]}"; do
    echo "Running evaluation for model: $model_path"

    accelerate launch -m lm_eval --model hf \
        --model_args="pretrained=$model_path,dtype=bfloat16" \
        --tasks=$TASK \
        --batch_size=auto \
        --output_path=$OUTPUT_DIR \
        --log_samples \
        --trust_remote_code \
        --confirm_run_unsafe_code \
        --num_fewshot 0 \
        --apply_chat_template
done