"""
Take in a YAML, and output all other splits with this YAML
"""

import argparse
import os

import yaml
from tqdm import tqdm

#from lm_eval.utils import eval_logger


mmlu_subjects = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions"
]



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_yaml_path", required=True)
    parser.add_argument("--save_prefix_path", default="zks_mmlu")
    parser.add_argument("--cot_prompt_path", default=None)
    parser.add_argument("--task_prefix", default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # get filename of base_yaml so we can `"include": ` it in our other YAMLs.
    base_yaml_name = os.path.split(args.base_yaml_path)[-1]
    with open(args.base_yaml_path, encoding="utf-8") as f:
        base_yaml = yaml.full_load(f)

    if args.cot_prompt_path is not None:
        import json

        with open(args.cot_prompt_path, encoding="utf-8") as f:
            cot_file = json.load(f)

    for subject in tqdm(mmlu_subjects):
        if args.cot_prompt_path is not None:
            description = cot_file[subject_eng]
        else:
            description = (
                f"The following are multiple-choice questions on {subject}. Please select the correct answer.\n\n"
            )

        yaml_dict = {
            "include": base_yaml_name,
            "task": f"zks_mmlu_{args.task_prefix}_{subject}"
            if args.task_prefix != ""
            else f"zks_mmlu_{subject}",
            "dataset_name": subject,
            "description": description,
        }

        file_save_path = args.save_prefix_path + f"_{subject}.yaml"
        #eval_logger.info(f"Saving yaml for subset {subject_eng} to {file_save_path}")
        with open(file_save_path, "w", encoding="utf-8") as yaml_file:
            yaml.dump(
                yaml_dict,
                yaml_file,
                width=float("inf"),
                allow_unicode=True,
                default_style='"',
            )

    # write group config out

    group_yaml_dict = {
        "group": "zks_mmlu",
        "task": [f"zks_mmlu_{task_name}" for task_name in mmlu_subjects],
        "aggregate_metric_list": [
            {"metric": "acc", "aggregation": "mean", "weight_by_size": True},
            {"metric": "acc_norm", "aggregation": "mean", "weight_by_size": True},
        ],
        "metadata": {"version": 2.0},
    }

    file_save_path = "_" + args.save_prefix_path + ".yaml"

    with open(file_save_path, "w", encoding="utf-8") as group_yaml_file:
        yaml.dump(
            group_yaml_dict,
            group_yaml_file,
            width=float("inf"),
            allow_unicode=True,
            default_style='"',
        )
