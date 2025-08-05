# 制作 mmap 数据
#!/bin/bash
set -exuo pipefail

cd /mnt/public2/code/zks/build_data
exec &> >(tee -a /mnt/public2/data/zks/build_data/logs/tokenize.log)
start_time=$(date +%s.%3N)


# ─── 环境准备和参数 ──────────────────────────────────────────────────────────────────

pip install nltk --no-deps
bash tools/config_nltk.sh

TOKENIZER_PATH="/mnt/public/model/huggingface/Qwen2.5-3B"
PREFIXES=(
    #"26k_books" "guidelines" "pubmed" "statpearls" "textbooks"
)
declare -A PREFIX_2_JSONL_KEY=(
    ["26k_books"]="text"
    ["guidelines"]="text"
    ["pubmed"]="text"
    ["statpearls"]="text"
    ["textbooks"]="text"
)
WORKERS=64
INPUT_QUEUE_SIZE=2000
CHUNK_SIZE=2000


# ─── 生成每个源的 MMAP 文件 ───────────────────────────────────────────────────────────

for prefix in "${PREFIXES[@]}"; do
    json_key=${PREFIX_2_JSONL_KEY[$prefix]}
    
    python /mnt/public2/data/ssc/pt-prepare/Megatron-DeepSpeed/tools/preprocess_data_many_cores.py \
        --input "/mnt/h_public/lh/medical_tlp/data_organized/pretrain2048/${prefix}/chunk.jsonl" \
        --json-key "$json_key" \
        --output-prefix "/mnt/public2/data/zks/build_data/data/mmaps/${prefix}" \
        --dataset-impl mmap \
        --tokenizer-type PretrainedFromHF \
        --tokenizer-name-or-path "$TOKENIZER_PATH" \
        --append-eod \
        --workers "$WORKERS" \
        --input-queue-size "$INPUT_QUEUE_SIZE" \
        --chunk-size "$CHUNK_SIZE"
done

# ─── 合并所有源 ────────────────────────────────────────────────────────────────────

python tools/merge_datasets.py \
    --input /mnt/public2/data/zks/build_data/data/mmaps \
    --output-prefix /mnt/public2/data/zks/build_data/data/mmaps/medical_corpora

# ─── 计算每个源的 Token 数量 ──────────────────────────────────────────────────────────

python tools/get_tokens.py --dataset-path /mnt/public2/data/zks/build_data/data/mmaps/medical_corpora    --sample-index -1 --tokenizer-path /mnt/public/model/huggingface/Qwen2.5-3B >> /mnt/public2/data/zks/build_data/data/mmaps/stats.txt
python tools/get_tokens.py --dataset-path /mnt/public2/data/zks/build_data/data/mmaps/26k_books_text_document    --sample-index -1 --tokenizer-path /mnt/public/model/huggingface/Qwen2.5-3B >> /mnt/public2/data/zks/build_data/data/mmaps/stats.txt
python tools/get_tokens.py --dataset-path /mnt/public2/data/zks/build_data/data/mmaps/guidelines_text_document    --sample-index -1 --tokenizer-path /mnt/public/model/huggingface/Qwen2.5-3B >> /mnt/public2/data/zks/build_data/data/mmaps/stats.txt
python tools/get_tokens.py --dataset-path /mnt/public2/data/zks/build_data/data/mmaps/pubmed_text_document    --sample-index -1 --tokenizer-path /mnt/public/model/huggingface/Qwen2.5-3B >> /mnt/public2/data/zks/build_data/data/mmaps/stats.txt
python tools/get_tokens.py --dataset-path /mnt/public2/data/zks/build_data/data/mmaps/statpearls_text_document    --sample-index -1 --tokenizer-path /mnt/public/model/huggingface/Qwen2.5-3B >> /mnt/public2/data/zks/build_data/data/mmaps/stats.txt
python tools/get_tokens.py --dataset-path /mnt/public2/data/zks/build_data/data/mmaps/textbooks_text_document      --sample-index -1 --tokenizer-path /mnt/public/model/huggingface/Qwen2.5-3B >> /mnt/public2/data/zks/build_data/data/mmaps/stats.txt



# ─── 执行完毕 ─────────────────────────────────────────────────────────────────────

end_time=$(date +%s.%3N)
elapsed=$(echo "$end_time - $start_time" | bc)  # 计算耗时
printf "执行耗时: %.3f 秒\n" "$elapsed"
echo "脚本执行完毕"
