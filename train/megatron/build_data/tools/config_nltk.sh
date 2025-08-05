#!/bin/bash
set -euxo pipefail

apt update
apt install git -y
apt install unzip -y

# 明确使用 root 目录
NLTK_DIR="/root/nltk_data"
mkdir -p $NLTK_DIR

# 克隆仓库到临时目录
TMP_DIR=$(mktemp -d)
git clone https://gitee.com/ki_seki_admin/nltk_data.git $TMP_DIR

# 处理压缩包（根据实际仓库结构调整）
cd $TMP_DIR

# 如果仓库直接将 zip 文件放在根目录
find . -name "*.zip" -exec unzip -o {} -d "$NLTK_DIR" \;

# 如果仓库内有 packages 目录
if [ -d "packages" ]; then
    find packages/ -name "*.zip" -exec unzip -o {} -d "$NLTK_DIR" \;
fi

# 清理临时文件
rm -rf $TMP_DIR

# 设置 NLTK 数据路径（可选）
echo "export NLTK_DATA=$NLTK_DIR" >> /root/.bashrc