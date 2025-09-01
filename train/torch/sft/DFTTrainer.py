from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    logging as hf_logging,
)
import os
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss


class DFTTrainer(Trainer):
    def __init__(self, log_metrics_steps: int = 100, use_simple_dft: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_metrics_steps = log_metrics_steps
        self.use_simple_dft = use_simple_dft
        self.is_main_process = self.args.local_rank <= 0
        # 统一保存策略：使用 transformers 内置 TrainingArguments.save_only_model
        self.save_only_model = bool(getattr(self.args, "save_only_model", False))
        
        if not self.is_main_process:
            hf_logging.set_verbosity_error()

    def _save_checkpoint(self, model, trial, metrics=None):
        """如果设置为仅保存模型，则在每个 checkpoint 只保存模型与分词器。"""
        if not self.save_only_model:
            return super()._save_checkpoint(model, trial, metrics)
        
        # 仅保存模型权重
        checkpoint_folder = f"checkpoint-{self.state.global_step}"
        output_dir = os.path.join(self.args.output_dir, checkpoint_folder)
        os.makedirs(output_dir, exist_ok=True)
        
        # 使用内部 _save，确保兼容 deepspeed/fp16 等场景
        self._save(output_dir)
        
        # 保存 tokenizer
        if self.tokenizer is not None:
            try:
                self.tokenizer.save_pretrained(output_dir)
            except Exception:
                pass
        
        '''
        # 轮转旧 checkpoint
        try:
            # 不同版本签名不同，优先无参调用
            self._rotate_checkpoints()
        except TypeError:
            try:
                self._rotate_checkpoints(use_mtime=False)
            except Exception:
                pass
        '''

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # 获取vocab size
        vocab_size = logits.shape[-1]
        
        # Shift操作：确保logits和labels对齐
        # logits: [batch, seq_len, vocab] -> [batch, seq_len-1, vocab]
        # labels: [batch, seq_len] -> [batch, seq_len-1]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # 展平用于loss计算
        # shift_logits: [batch * (seq_len-1), vocab]
        # shift_labels: [batch * (seq_len-1)]
        shift_logits_flat = shift_logits.view(-1, vocab_size)
        shift_labels_flat = shift_labels.view(-1)
        
        # 计算基础loss
        loss_fct = CrossEntropyLoss(reduction='none')
        loss_flat = loss_fct(shift_logits_flat, shift_labels_flat)
        
        if self.use_simple_dft:
            # DFT: 按预测概率加权
            with torch.no_grad():
                probs = F.softmax(shift_logits_flat, dim=-1)
                # 获取正确token的概率
                valid_mask = shift_labels_flat != -100
                gather_labels = shift_labels_flat.clone()
                gather_labels[~valid_mask] = 0  # 将padding位置设为0以避免gather错误
                
                p_correct = probs.gather(1, gather_labels.unsqueeze(-1)).squeeze(-1)
                dft_weight = p_correct * valid_mask.float()  # mask掉padding
            
            # 应用DFT权重
            loss_flat = loss_flat * dft_weight.detach()
        
        # 计算GA平均loss
        if num_items_in_batch is not None and num_items_in_batch > 0:
            loss = loss_flat.sum() / num_items_in_batch
        else:
            valid_tokens = (shift_labels_flat != -100).sum()
            loss = loss_flat.sum() / valid_tokens

        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes
        
        # 记录指标
        if model.training and self.state.global_step % self.log_metrics_steps == 0 and self.is_main_process:
            with torch.no_grad():
                if self.use_simple_dft and num_items_in_batch > 0 and 'dft_weight' in locals():
                    avg_dft_weight = dft_weight[valid_mask].mean().item()
                    self.log({
                        "train/avg_dft_weight": avg_dft_weight,
                    })
        
        return (loss, outputs) if return_outputs else loss