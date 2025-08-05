import pandas as pd
import ujson

res_file_list = [
    '/mnt/public/code/zks/lmeval_res/zks/3B_mc_{5}shot/__mnt__public__code__zks__tmp__loss_diff_step190700/results_2025-06-18T11-40-35.583835.json',
    '/mnt/public/code/zks/lmeval_res/zks/3B_mc_{5}shot/__mnt__public__code__zks__tmp__baseline_step152550/results_2025-06-18T12-00-23.384293.json'
]
model_name_list = ['lossdiff', 'baseline']
                  

'''
key_list = [
    #["agieval_gaokao_mathcloze", "acc,none"],
    ["agieval_math", "acc,none"],
    ["gsm8k", "exact_match,flexible-extract"],
    ["gsm8k_cot", "exact_match,flexible-extract"],
    ["humaneval", "pass@1,create_test"],
    #["humaneval_64", "pass@64,create_test"],
    ["mbpp", "pass_at_1,none"]
]
'''
key_list = [
    ["zks_mmlu", "acc_norm,none"],
    ["zks_arc_easy", "acc,none"],
    ["zks_arc_challenge", "acc_norm,none"],
    ["zks_truthfulqa", "acc_norm,none"],
    ["zks_hellaswag", "acc_norm,none"],
    ["zks_winogrande", "acc_norm,none"],
    ["zks_cmmlu", "acc_norm,none"],
    ["zks_ceval", "acc_norm,none"],
    ["zks_clue_c3", "acc_norm,none"],
    ["zks_gaokao_mathqa", "acc_norm,none"],
]


def create_dataframe_from_json_files():
    data = {}
    
    for i, file_path in enumerate(res_file_list):
        try:
            model_name = model_name_list[i]
            
            # 读取JSON文件
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = ujson.load(f)
            
            # 提取所需的键值对
            model_data = {}
            for (key1, key2) in key_list:
                model_data[key1[4:]] = json_data['results'][key1][key2]
                
            # 计算平均值
            value_list = [v for v in model_data.values() if isinstance(v, (int, float))]
            model_data['mean'] = sum(value_list) / len(value_list) 
            
            data[model_name] = model_data
            
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
            continue
    
    # 创建DataFrame并转置，使模型名作为行索引，keys作为列
    df = pd.DataFrame.from_dict(data, orient='index')
    
    return df

if __name__ == "__main__":
    df = create_dataframe_from_json_files()
    df.to_csv('/mnt/public/code/zks/lmeval_res/zks/3B_mc_{5}shot/res.csv', index=True)
    print(df)