#!/bin/bash
set -exuo pipefail
exec > >(tee -a /mnt/public2/code/zks/tmp/output.log) 2>&1

# ─── 参数设置 ─────────────────────────────────────────────────────────────────────
MODEL_1="/mnt/public/code/zks/tmp/loss_diff_step190700"
MODEL_2="/mnt/public/code/zks/tmp/baseline_step152550"
#MODEL_3="/mnt/public/model/zks/sft_ckpt/full/0609_lossdiff_1/checkpoint-2000"
#MODEL_4="/mnt/public/model/zks/sft_ckpt/full/0610_baseline_1/checkpoint-2000"
#MODEL_5="/mnt/public/model/zks/sft_ckpt/full/0609_lossdiff_1/checkpoint-4000"
#MODEL_6="/mnt/public/model/zks/sft_ckpt/full/0610_baseline_1/checkpoint-4000"
#MODEL_3="/mnt/public/code/zzy/dynamic_train/expt/0414-1-baseline/ckpt/global_step19060_hf"
#MODEL_4="/mnt/public/code/zzy/dynamic_train/expt/0424-1-decision-tree/ckpt/global_step19060_hf"
#MODEL_5="/mnt/public/model/zks/Qwen2.5-0.5B"
#MODEL_6="/mnt/public/model/zks/Qwen2.5-7B"
#MODEL_7="/mnt/public/code/zzy/wzh/dynamic_train/expt/0427-doremi-main_0.5B/ckpt/global_step19060_hf"

#CONVERT_SCRIPT="/mnt/public/code/ssc/dynamic_train/Megatron-DeepSpeed-dynamic/tools/convert_checkpoint/deepspeed_ckpt_qwen2.5_to_hf.py"

# ─── 模型转换 ─────────────────────────────────────────────────────────────────────

# python $CONVERT_SCRIPT --input $ORIGINAL_MODEL --output $HF_MODEL

# ─── 环境准备 ─────────────────────────────────────────────────────────────────────
apt update
apt install -y ca-certificates
update-ca-certificates

export HF_ENDPOINT=https://hf-mirror.com
export HUGGING_FACE_HUB_TOKEN=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
export HF_TOKEN=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
export PATH=/opt/conda/bin:/opt/conda/condabin:$PATH
# ln -s /opt/maca/tools/cu-bridge/bin/cucc /opt/maca/tools/cu-bridge/bin/nvcc


cd /mnt/public/code/data_mixture/lm-evaluation-harness
# pip install peft
pip install transformers==4.51.3
pip install -e '.[math]'
pip install langdetect immutabledict --no-deps

export HF_ALLOW_CODE_EVAL=1

# ─── 模型评估 ─────────────────────────────────────────────────────────────────────
TASK1="zks_mmlu,zks_arc_easy,zks_arc_challenge,zks_truthfulqa,zks_winogrande,zks_hellaswag,zks_cmmlu,zks_ceval,zks_clue_c3,zks_gaokao_mathqa"
FEW_SHOT=0

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --few_shot) FEW_SHOT="$2"; shift ;;
        # 如果需要支持更多参数，可以在这里添加更多的case语句
    esac
    shift
done

echo "The value of few_shot is: $FEW_SHOT"
OUTPUT_DIR="/mnt/public/code/zks/lmeval_res/zks/3B_mc_{$FEW_SHOT}shot"

for i in 1 2
do
    # 定义当前模型变量名
    model_var="MODEL_$i"
    
    # 使用间接扩展获取实际模型路径
    model_path=${!model_var}
    
    # 输出当前使用的模型路径，可选
    echo "Running evaluation for model: $model_path"
    
    # 执行accelerate命令，替换--model_args中的pretrained参数
    accelerate launch -m lm_eval --model hf \
        --model_args="pretrained=$model_path,dtype=bfloat16" \
        --tasks=$TASK1 \
        --batch_size=auto \
        --output_path=$OUTPUT_DIR \
        --log_samples \
        --trust_remote_code \
        --confirm_run_unsafe_code \
        --num_fewshot=$FEW_SHOT
done