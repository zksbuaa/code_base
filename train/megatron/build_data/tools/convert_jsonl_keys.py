#!/usr/bin/env python3
import json
import argparse
import os
from tempfile import NamedTemporaryFile
from typing import List
from tqdm import tqdm

def safe_convert_keys(
    input_path: str,
    old_keys: List[str],
    new_key: str,
    separator: str = " "
) -> None:
    """
    跨文件系统安全的原地修改版本
    """
    # 在目标文件同目录创建临时文件
    target_dir = os.path.dirname(os.path.abspath(input_path))
    with NamedTemporaryFile('w', encoding='utf-8', 
                          dir=target_dir,  # 关键修改：与目标文件同目录
                          delete=False) as tmpfile:
        temp_path = tmpfile.name
        try:
            with open(input_path, 'r', encoding='utf-8') as infile:
                for line_num, line in tqdm(enumerate(infile, 1)):
                    try:
                        data = json.loads(line.strip())
                        values = [
                            str(data[key]).strip() 
                            for key in old_keys 
                            if key in data and str(data[key]).strip()
                        ]
                        new_data = {new_key: separator.join(values)} if values else {}
                        if new_data:
                            tmpfile.write(json.dumps(new_data, ensure_ascii=False) + '\n')
                    except json.JSONDecodeError:
                        print(f"Warning: 跳过第 {line_num} 行 (无效的 JSON 格式)")
                    except Exception as e:
                        print(f"Error: 处理第 {line_num} 行时发生错误 - {str(e)}")

            # 原子替换（同文件系统）
            os.replace(temp_path, input_path)
        except Exception as e:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise RuntimeError(f"文件处理失败: {str(e)}") from e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JSONL 原地转换工具 (跨文件系统安全版)")
    parser.add_argument('-i', '--input', required=True, help="需要修改的文件路径")
    parser.add_argument('-k', '--keys', nargs='+', required=True, help="需要合并的原始 keys")
    parser.add_argument('-n', '--new-key', required=True, help="合并后的新 key 名称")
    parser.add_argument('-s', '--separator', default=' ', help="合并值使用的分隔符")

    args = parser.parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"输入文件不存在: {args.input}")

    print(f"开始安全处理文件...")
    try:
        safe_convert_keys(
            input_path=args.input,
            old_keys=args.keys,
            new_key=args.new_key,
            separator=args.separator
        )
        print(f"成功更新文件: {args.input}")
    except Exception as e:
        print(f"处理失败: {str(e)}")
        exit(1)
