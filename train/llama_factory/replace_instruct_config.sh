#!/bin/bash

src_dir="/mnt/public/model/zks/Qwen2.5-3B-Instruct"

files=(
  "config.json"
  "generation_config.json"
  "tokenizer_config.json"
  "tokenizer.json"
  "vocab.json"
)

# 目标目录列表 b
target_dirs=(
  "/mnt/public/model/zks/sft_ckpt/full/0609_lossdiff_1/checkpoint-6000"
  "/mnt/public/model/zks/sft_ckpt/full/0609_lossdiff_1/checkpoint-8000"
  "/mnt/public/model/zks/sft_ckpt/full/0610_baseline_1/checkpoint-6000"
  "/mnt/public/model/zks/sft_ckpt/full/0610_baseline_1/checkpoint-8000"
)

# 遍历每个目标目录
for target_dir in "${target_dirs[@]}"; do
  # 拷贝每个文件，覆盖同名文件
  for file in "${files[@]}"; do
    src_file="$src_dir/$file"
    if [[ -f "$src_file" ]]; then
      cp -f "$src_file" "$target_dir/"
      echo "Copied $file to $target_dir"
    else
      echo "[Warning] $file not found in $src_dir"
    fi
  done
done
