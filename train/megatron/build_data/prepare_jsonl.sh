# 制作 jsonl 数据
#!/bin/bash
set -exuo pipefail
apt install bc

cd /mnt/public2/code/zks/build_data
exec &> >(tee -a /mnt/public2/data/zks/build_data/logs/jsonl.log)
start_time=$(date +%s.%3N)


# ─── 计算每个数据源需要的 Gb 大小 ─────────────────────────────────────────────────────────

train_tokens_in_billion=12  # 训练数据的 token 数量
expansion_ratio=1.2  # 防止取的数据不够
train_tokens_in_billion=$(echo "$train_tokens_in_billion * $expansion_ratio" | bc)

data_mixture=(0.1 0.05 0.1 0.05 0.1 0.1 0.15 0.15 0.05355 0.01291 0.02654 0.04026 0.06674)  # 归一化的 13 个源的混合比例
gb_to_b_ratios=(4.6 4.5 4.7 4.7 3.6 4.1 4.0 3.2 1.8 3.3 5.2 4.7 4.2)  # 每个源 JSONL GB 大小和对应的 tokens 数量比值（估计值）
required_gbs=()
for i in "${!data_mixture[@]}"; do
    required_gb=$(echo "${train_tokens_in_billion} * ${data_mixture[$i]} * ${gb_to_b_ratios[$i]} / 1" | bc)
    required_gbs+=("$required_gb")
done
echo "每个源所需的 GB 大小: ${required_gbs[@]}"


# ─── 生成每个源的 Jsonl 文件 ──────────────────────────────────────────────────────────
# 基本上不用修改

bash   tools/any2jsonl.sh \
    --source /mnt/h_public/PreTrain-Data/EN_data_segmentation/score_1_2,/mnt/h_public/PreTrain-Data/EN_data_segmentation/score_2_3,/mnt/h_public/PreTrain-Data/EN_data_segmentation/score_3_4,/mnt/h_public/PreTrain-Data/EN_data_segmentation/score_4_5,/mnt/h_public/PreTrain-Data/EN_data_segmentation/score_5_6 \
    --depth 1 \
    --suffixes jsonl.zst \
    --method random \
    --output /mnt/public2/data/zks/build_data/general/jsonl/01_en_web_1.jsonl \
    --size-gb ${required_gbs[$((1-1))]}

bash   tools/any2jsonl.sh \
    --source /mnt/h_public/PreTrain-Data/EN_data_segmentation/score_-2_-1,/mnt/h_public/PreTrain-Data/EN_data_segmentation/score_-1_0,/mnt/h_public/PreTrain-Data/EN_data_segmentation/score_0_1 \
    --depth 1 \
    --suffixes jsonl.zst \
    --method random \
    --output /mnt/public2/data/zks/build_data/general/jsonl/02_en_web_2.jsonl \
    --size-gb ${required_gbs[$((2-1))]}

bash   tools/any2jsonl.sh \
    --source /mnt/h_public/PreTrain-Data/CN_data_segmentation/score_1_2,/mnt/h_public/PreTrain-Data/CN_data_segmentation/score_2_3,/mnt/h_public/PreTrain-Data/CN_data_segmentation/score_3_4,/mnt/h_public/PreTrain-Data/CN_data_segmentation/score_4_5 \
    --depth 1 \
    --suffixes jsonl.zst \
    --method random \
    --output /mnt/public2/data/zks/build_data/general/jsonl/03_cn_web_1.jsonl \
    --size-gb ${required_gbs[$((3-1))]}

bash   tools/any2jsonl.sh \
    --source /mnt/h_public/PreTrain-Data/CN_data_segmentation/score_-2_-1,/mnt/h_public/PreTrain-Data/CN_data_segmentation/score_-1_0,/mnt/h_public/PreTrain-Data/CN_data_segmentation/score_0_1 \
    --depth 1 \
    --suffixes jsonl.zst \
    --method random \
    --output /mnt/public2/data/zks/build_data/general/jsonl/04_cn_web_2.jsonl \
    --size-gb ${required_gbs[$((4-1))]}

bash   tools/any2jsonl.sh \
    --source /mnt/h_public/PreTrain-Data/scihub_jsonl \
    --depth 2 \
    --suffixes jsonl \
    --method random \
    --output /mnt/public2/data/zks/build_data/general/jsonl/05_en_kno.jsonl \
    --size-gb ${required_gbs[$((5-1))]}

bash   tools/any2jsonl.sh \
    --source /mnt/h_public/PreTrain-Data/C-Ebook,/mnt/h_public/PreTrain-Data/CN/corpus_source=IndustryCorpus2 \
    --depth 1 \
    --suffixes jsonl.zst,parquet \
    --no-ext \
    --default-ext parquet \
    --method size_desc \
    --output /mnt/public2/data/zks/build_data/general/jsonl/06_cn_kno.jsonl \
    --size-gb ${required_gbs[$((6-1))]}

python   tools/convert_jsonl_keys.py \
    --input /mnt/public2/data/zks/build_data/general/jsonl/06_cn_kno.jsonl \
    --keys text content corpus_content \
    --new-key corpus_content

bash   tools/any2jsonl.sh \
    --source /mnt/public/data/the-stack-v2-train-smol-ids \
    --depth 2 \
    --suffixes jsonl \
    --method random \
    --output /mnt/public2/data/zks/build_data/general/jsonl/07_code.jsonl \
    --size-gb ${required_gbs[$((7-1))]}

bash   tools/any2jsonl.sh \
    --source /mnt/h_public/PreTrain-Data/Math \
    --depth 10 \
    --suffixes parquet,jsonl.zst \
    --method random \
    --output /mnt/public2/data/zks/build_data/general/jsonl/08_math.jsonl \
    --size-gb ${required_gbs[$((8-1))]}

python   tools/convert_jsonl_keys.py \
    --input /mnt/public2/data/zks/build_data/general/jsonl/08_math.jsonl \
    --keys input output problem generated_solution text \
    --new-key corpus_content


# cp /mnt/h_public/ssc/data_mixture/data2/data_en_pile_wo_web.jsonl /mnt/public2/data/zks/build_data/general/jsonl/09_en_pile_wo_web.jsonl

# cp /mnt/h_public/ssc/data_mixture/data2/data_wiki.jsonl /mnt/public2/data/zks/build_data/general/jsonl/10_wiki.jsonl

# cp /mnt/h_public/ssc/data_mixture/data2/data_base_all.jsonl /mnt/public2/data/zks/build_data/general/jsonl/11_cn_base.jsonl

bash   tools/any2jsonl.sh \
    --source /mnt/h_public/ssc/data_mixture/data2/data_en_pile_wo_web.jsonl \
    --depth 1 \
    --suffixes jsonl \
    --method random \
    --output /mnt/public2/data/zks/build_data/general/jsonl/09_en_pile_wo_web.jsonl \
    --size-gb ${required_gbs[$((9-1))]}

bash   tools/any2jsonl.sh \
    --source /mnt/h_public/ssc/data_mixture/data2/data_wiki.jsonl \
    --depth 1 \
    --suffixes jsonl \
    --method random \
    --output /mnt/public2/data/zks/build_data/general/jsonl/10_wiki.jsonl \
    --size-gb ${required_gbs[$((10-1))]}

bash   tools/any2jsonl.sh \
    --source /mnt/h_public/ssc/data_mixture/data2/data_base_all.jsonl \
    --depth 1 \
    --suffixes jsonl \
    --method random \
    --output /mnt/public2/data/zks/build_data/general/jsonl/11_cn_base.jsonl \
    --size-gb ${required_gbs[$((11-1))]}


bash   tools/any2jsonl.sh \
    --source /mnt/h_public/PreTrain-Data/CN_data_segmentation/score_5_6,/mnt/h_public/PreTrain-Data/CN_data_segmentation/score_6_7,/mnt/h_public/PreTrain-Data/CN_data_segmentation/score_7_8,/mnt/h_public/PreTrain-Data/CN_data_segmentation/score_8_9 \
    --depth 1 \
    --suffixes jsonl.zst \
    --method random \
    --output /mnt/public2/data/zks/build_data/general/jsonl/12_cn_best.jsonl \
    --size-gb ${required_gbs[$((12-1))]}

bash   tools/any2jsonl.sh \
    --source /mnt/h_public/PreTrain-Data/EN_data_segmentation/score_6_7 \
    --depth 1 \
    --suffixes jsonl.zst \
    --method random \
    --output /mnt/public2/data/zks/build_data/general/jsonl/13_en_best.jsonl \
    --size-gb ${required_gbs[$((13-1))]}


# ─── 执行完毕 ─────────────────────────────────────────────────────────────────────

end_time=$(date +%s.%3N)
elapsed=$(echo "$end_time - $start_time" | bc)  # 计算耗时
printf "执行耗时: %.3f 秒\n" "$elapsed"
echo "脚本执行完毕"
