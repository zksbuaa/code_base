from datasets import load_dataset
import json
from tqdm import tqdm

# 配置部分
dataset_name = "/mnt/public/data/zks/benchmarks/humaneval"
#config_list = ['main']
split_list = ["test"]  # 根据实际数据集的分割情况调整   
output_file = "/mnt/public2/data/zks/train_benchmark/humaneval.jsonl"

# 可选：用于字段映射（需根据实际数据集格式修改）
input_field = "prompt"        # 或 "question", "prompt"
output_field = "canonical_solution"            # 或 "response", "answer"

data = []

# 加载数据集
#for config in config_list:
for split in split_list:
    dataset = load_dataset(dataset_name, split=split)
    for item in dataset:
        data.append(item)

# 转换为 ShareGPT 格式
with open(output_file, "w", encoding="utf-8") as fout:
    for example in tqdm(data):
        conversations = []
        # 添加用户输入
        user_input = example.get(input_field, "").strip()
        if user_input:
            conversations.append({
                "from": "human",
                "value": user_input
            })

        # 添加助手回复
        assistant_output = example.get(output_field, "").strip()
        if assistant_output:
            conversations.append({
                "from": "gpt",
                "value": '    ' + assistant_output
            })

        # 写入到 JSONL 文件
        fout.write(json.dumps({"conversations": conversations}, ensure_ascii=False) + "\n")
