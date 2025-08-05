import argparse
from tqdm import tqdm
from indexed_dataset import MMapIndexedDataset, make_builder
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, default=None, help='input path')
    parser.add_argument('--output-path', type=str, default=None, help='output path')
    parser.add_argument('--n-tokens', type=int, default=None, help='number of tokens to save')
    parser.add_argument('--vocab-size', type=int, default=151936, help='vocab size')
    parser.add_argument('--tokenizer-path', type=str, default="/mnt/public/code/data_mixture/models/Qwen2.5-0.5B", help='tokenizer path')
    return parser.parse_args()

def save_first_n_tokens(input_path, output_path, n_tokens, vocab_size):
    output_bin = output_path + ".bin"
    output_idx = output_path + ".idx"

    i_dataset = MMapIndexedDataset(input_path, skip_warmup=True)
    o_builder = make_builder(output_bin, impl="mmap", vocab_size=vocab_size)

    i = 0
    total_tokens = 0
    pbar = tqdm(total=n_tokens, desc="Processing tokens")
    while True:
        doc = i_dataset[i % len(i_dataset)]

        if total_tokens + len(doc) > n_tokens:
            doc = doc[:n_tokens - total_tokens]
            o_builder.add_doc(doc, [len(doc)])
            total_tokens += len(doc)
            pbar.update(len(doc))
            break
        else:
            o_builder.add_doc(doc, [len(doc)])
            total_tokens += len(doc)
            pbar.update(len(doc))
        i += 1

    o_builder.finalize(output_idx)
    pbar.close()

def decode_sentence(sentence, tokenizer_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side="right", use_fast=False)
    return tokenizer.decode(sentence)

if __name__ == "__main__":
    args = parse_args()
    print(args)
    save_first_n_tokens(args.input_path, args.output_path, args.n_tokens, args.vocab_size)
