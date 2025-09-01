from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import DataCollator



def convert_text_to_qwen_input(item, tokenizer, max_len):

    messages = [
        {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
        {"role": "user", "content": item['problem']},
        {"role": "assistant", "content": item['solution']}
    ]

    text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False
    )
    prefix = text[:text.rfind('<|im_start|>assistant')+len('<|im_start|>assistant\n')]

    # Tokenize the text input
    input_ids = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_len)['input_ids'][0]
    prefix_ids = tokenizer(prefix, return_tensors="pt", padding=True, truncation=True, max_length=max_len)['input_ids'][0]

    # Initialize attention_mask and labels
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)

    # Create labels, initially copy input_ids
    labels = input_ids.clone()

    # Set labels to -100 where attention_mask is 0 (those tokens will be ignored during loss computation)
    labels[:prefix_ids.shape[0]] = -100

    item['input_ids'] = input_ids
    item['attention_mask'] = attention_mask
    item['labels'] = labels

    return item




class MaxPaddingCollator():
    def __init__(self):
        super().__init__()

    def __call__(self, examples):
        # 获取 batch 中每个样本的 input_ids, attention_mask, labels
        input_ids = [example['input_ids'] for example in examples]
        attention_masks = [example['attention_mask'] for example in examples]
        labels = [example['labels'] for example in examples]
        
        # 找到最长的 input_ids 长度
        max_len = max([len(input_id) for input_id in input_ids])

        # 对每个样本进行 padding
        padded_input_ids = torch.stack([self._pad_tensor(input_id, max_len) for input_id in input_ids])
        padded_attention_masks = torch.stack([self._pad_tensor(att_mask, max_len, pad_value=0) for att_mask in attention_masks])
        padded_labels = torch.stack([self._pad_tensor(label, max_len, pad_value=-100) for label in labels])

        # 返回 padding 后的批次数据
        return {
            'input_ids': padded_input_ids.long(),
            'attention_mask': padded_attention_masks.long(),
            'labels': padded_labels.long()
        }

    def _pad_tensor(self, sequence, max_len, pad_value=0):
        """
        Helper function to pad tensors to the max_len
        """
        padding_length = max_len - len(sequence)
        return torch.cat([torch.tensor(sequence), torch.tensor([pad_value] * padding_length)])