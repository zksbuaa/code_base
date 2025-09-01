from logger import DistributedLogger
from DFTTrainer import DFTTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, HfArgumentParser, Trainer
from DataProcessor import convert_text_to_qwen_input, MaxPaddingCollator
from dataclasses import dataclass, field
from typing import List, Optional
from datasets import load_dataset, Dataset
from functools import partial
import os
import traceback
import torch
import random



@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Path to pretrained model"})
    torch_dtype: str = field(default="bfloat16")
    trust_remote_code: bool = field(default=True)

@dataclass
class DataArguments:
    dataset_path: str = field(metadata={"help": "dataset"})
    train_sample_num: int = field(default=100000)
    eval_sample_num: int = field(default=500)
    max_length: int = field(default=8192)
    preprocessing_num_workers: int = field(default=8)
    sample_seed: int = field(default=42)

@dataclass
class DFTArguments:
    enable_gradient_checkpointing: bool = field(default=True)
    use_dft: bool = field(default=True)

@dataclass 
class LoggingArguments:
    dist_logger: Optional[str] = field(default=None)
    reduce_logging: bool = field(default=True)

def setup_model_and_tokenizer(model_args: ModelArguments):
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, 
        trust_remote_code=model_args.trust_remote_code,
        use_fast=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 对于训练，使用left padding以避免Flash Attention问题
    tokenizer.padding_side = 'left'
    
    dtype_mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32
    }
    torch_dtype = dtype_mapping.get(model_args.torch_dtype, torch.bfloat16)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=model_args.trust_remote_code
    )
    
    return model, tokenizer

def setup_datasets(data_args: DataArguments, tokenizer, training_args: TrainingArguments):
    process_func = partial(convert_text_to_qwen_input, tokenizer=tokenizer, max_len=data_args.max_length)
    
    dataset = load_dataset(data_args.dataset_path)['train']

    random.seed(data_args.sample_seed)
    random_samples_for_train = random.sample(range(len(dataset)), data_args.train_sample_num)
    train_dataset = dataset.select(random_samples_for_train).map(process_func, num_proc=data_args.preprocessing_num_workers)

    random_samples_for_eval = random.sample(range(len(dataset)), data_args.eval_sample_num)
    eval_dataset = dataset.select(random_samples_for_eval).map(process_func, num_proc=data_args.preprocessing_num_workers)

    return train_dataset, eval_dataset



def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, DFTArguments, LoggingArguments, TrainingArguments))
    model_args, data_args, dft_args, logging_args, training_args = parser.parse_args_into_dataclasses()

    dist_logger = DistributedLogger(logging_args.dist_logger, f'{logging_args.dist_logger}.log' if logging_args.dist_logger else None)

    if training_args.local_rank <= 0:
       os.makedirs(training_args.output_dir, exist_ok=True)

    try:
        dist_logger.info(f"加载模型: {model_args.model_name_or_path}")
        model, tokenizer = setup_model_and_tokenizer(model_args)
        
        if dft_args.enable_gradient_checkpointing:
            model.gradient_checkpointing_enable()
            dist_logger.info("启用梯度检查点")
        
        train_dataset, eval_dataset = setup_datasets(data_args, tokenizer, training_args)
        
        training_args.group_by_length = False
        
        if training_args.local_rank > 0:
            training_args.report_to = []

        if dft_args.use_dft:
            dist_logger.info("使用DFT训练")
            trainer = DFTTrainer(
                model=model,
                tokenizer=tokenizer,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=MaxPaddingCollator()
            )

        else:
            dist_logger.info("使用SFT训练")
            trainer = Trainer(
                model=model,
                tokenizer=tokenizer,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=MaxPaddingCollator()
            )

        dist_logger.info(f"开始训练")
        trainer.train()
        
        trainer.save_model(training_args.output_dir)
        dist_logger.info(f"模型已保存: {training_args.output_dir}")
        
    except Exception as e:
        dist_logger.error(f"训练失败: {e}")
        dist_logger.error(f"详情: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    main()