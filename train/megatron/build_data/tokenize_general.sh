# 制作 mmap 数据
#!/bin/bash
set -exuo pipefail

cd /mnt/public2/code/zks/build_data
exec &> >(tee -a /mnt/public2/data/zks/build_data/logs/tokenize.log)
start_time=$(date +%s.%3N)


# ─── 环境准备和参数 ──────────────────────────────────────────────────────────────────

#pip install nltk --no-deps
#bash tools/config_nltk.sh

TOKENIZER_PATH="/mnt/public/model/huggingface/Qwen2.5-3B"
PREFIXES=(
    "01_en_web_1" 
    # "02_en_web_2" "05_en_kno" "13_en_best"
)
declare -A PREFIX_2_JSONL_KEY=(
    ["01_en_web_1"]="corpus_content"
    ["02_en_web_2"]="corpus_content"
    ["05_en_kno"]="corpus_content"
    ["13_en_best"]="corpus_content"
)
WORKERS=64
INPUT_QUEUE_SIZE=2000
CHUNK_SIZE=2000


# ─── 生成每个源的 MMAP 文件 ───────────────────────────────────────────────────────────

for prefix in "${PREFIXES[@]}"; do
    json_key=${PREFIX_2_JSONL_KEY[$prefix]}
    
    python /mnt/public2/data/ssc/pt-prepare/Megatron-DeepSpeed/tools/preprocess_data_many_cores.py \
        --input "/mnt/public2/data/zks/build_data/general/jsonl/${prefix}.jsonl" \
        --json-key "$json_key" \
        --output-prefix "/mnt/public2/data/zks/build_data/general/mmaps/${prefix}" \
        --dataset-impl mmap \
        --tokenizer-type PretrainedFromHF \
        --tokenizer-name-or-path "$TOKENIZER_PATH" \
        --append-eod \
        --workers "$WORKERS" \
        --input-queue-size "$INPUT_QUEUE_SIZE" \
        --chunk-size "$CHUNK_SIZE"
done

# ─── 合并所有源 ────────────────────────────────────────────────────────────────────
data_dir="/mnt/public2/data/zks/build_data/general/mmaps"

python tools/merge_datasets.py \
    --input ${data_dir} \
    --output-prefix ${data_dir}/general_corpora

# ─── 计算每个源的 Token 数量 ──────────────────────────────────────────────────────────
stats_file="${data_dir}/stats.txt"

python tools/get_tokens.py --dataset-path ${data_dir}/general_corpora    --sample-index -1 --tokenizer-path /mnt/public/model/huggingface/Qwen2.5-3B >> ${stats_file}
python tools/get_tokens.py --dataset-path ${data_dir}/01_en_web_1_corpus_content_document   --sample-index -1 --tokenizer-path /mnt/public/model/huggingface/Qwen2.5-3B >> ${stats_file}
python tools/get_tokens.py --dataset-path ${data_dir}/02_en_web_2_corpus_content_document    --sample-index -1 --tokenizer-path /mnt/public/model/huggingface/Qwen2.5-3B >> ${stats_file}
python tools/get_tokens.py --dataset-path ${data_dir}/05_en_kno_corpus_content_document    --sample-index -1 --tokenizer-path /mnt/public/model/huggingface/Qwen2.5-3B >> ${stats_file}
python tools/get_tokens.py --dataset-path ${data_dir}/13_en_best_corpus_content_document    --sample-index -1 --tokenizer-path /mnt/public/model/huggingface/Qwen2.5-3B >> ${stats_file}



# ─── 执行完毕 ─────────────────────────────────────────────────────────────────────

end_time=$(date +%s.%3N)
elapsed=$(echo "$end_time - $start_time" | bc)  # 计算耗时
printf "执行耗时: %.3f 秒\n" "$elapsed"
echo "脚本执行完毕"
