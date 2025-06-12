import os
import json
import requests
import tiktoken
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel

# Folder where downloaded data will go
data_folder = os.path.join(os.path.dirname(__file__), "hellaswag")

# Download helper
def get_file_from_url(link, out_file, part_size=1024):
    reply = requests.get(link, stream=True)
    total_size = int(reply.headers.get("content-length", 0))
    with open(out_file, "wb") as f, tqdm(
        desc=out_file,
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in reply.iter_content(chunk_size=part_size):
            written = f.write(chunk)
            bar.update(written)

# Links to all three splits
hellaswag_links = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

# Tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# Download one split
def get_split(split_name):
    os.makedirs(data_folder, exist_ok=True)
    url = hellaswag_links[split_name]
    out_file = os.path.join(data_folder, f"hellaswag_{split_name}.jsonl")
    if not os.path.exists(out_file):
        print(f"Downloading {url} to {out_file}...")
        get_file_from_url(url, out_file)

# Convert a sample into tensors
def convert_sample(sample):
    ctx_text = sample["ctx"]
    right_answer = sample["label"]
    choices = sample["endings"]

    info = {
        "label": right_answer,
        "ctx_tokens": None,
        "ending_tokens": [],
    }

    # Tokenize context and endings
    ctx_tokens = tokenizer.encode(ctx_text)
    info["ctx_tokens"] = ctx_tokens
    choice_rows = []
    mask_rows = []

    for c in choices:
        c_tokens = tokenizer.encode(" " + c)  # note the space!
        choice_rows.append(ctx_tokens + c_tokens)
        mask_rows.append([0]*len(ctx_tokens) + [1]*len(c_tokens))
        info["ending_tokens"].append(c_tokens)

    # Pad all sequences to same length
    max_len = max(len(r) for r in choice_rows)
    tokens_tensor = torch.zeros((4, max_len), dtype=torch.long)
    mask_tensor = torch.zeros((4, max_len), dtype=torch.long)

    for i, (row, mask) in enumerate(zip(choice_rows, mask_rows)):
        tokens_tensor[i, :len(row)] = torch.tensor(row)
        mask_tensor[i, :len(mask)] = torch.tensor(mask)

    return info, tokens_tensor, mask_tensor, right_answer

# Loop through all samples
def load_samples(split_name):
    get_split(split_name)
    path = os.path.join(data_folder, f"hellaswag_{split_name}.jsonl")
    with open(path, "r") as f:
        for line in f:
            sample = json.loads(line)
            yield sample

# Evaluation function
@torch.no_grad()
def run_eval(model_name, device_type):

    torch.set_float32_matmul_precision('high')
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.to(device_type)
    # model = torch.compile(model)  # optional

    correct = 0
    correct_norm = 0
    total = 0

    for sample in load_samples("val"):
        info, tokens, mask, label = convert_sample(sample)
        tokens = tokens.to(device_type)
        mask = mask.to(device_type)

        out_logits = model(tokens).logits
        shifted_logits = out_logits[..., :-1, :].contiguous()
        shifted_tokens = tokens[..., 1:].contiguous()
        flat_logits = shifted_logits.view(-1, shifted_logits.size(-1))
        flat_tokens = shifted_tokens.view(-1)

        loss_all = F.cross_entropy(flat_logits, flat_tokens, reduction='none')
        loss_all = loss_all.view(tokens.size(0), -1)

        shifted_mask = mask[..., 1:].contiguous()
        loss_only_end = loss_all * shifted_mask
        total_loss = loss_only_end.sum(dim=1)
        avg_loss = total_loss / shifted_mask.sum(dim=1)

        guess = total_loss.argmin().item()
        guess_norm = avg_loss.argmin().item()

        total += 1
        correct += int(guess == label)
        correct_norm += int(guess_norm == label)

        print(f"{total} acc_norm: {correct_norm}/{total} = {correct_norm/total:.4f}")

        # Print some samples
        if total < 10:
            print("---")
            print(f"Context:\n{sample['ctx']}")
            print("Choices:")
            for i, c in enumerate(sample["endings"]):
                print(f"{i} (loss: {avg_loss[i].item():.4f}) {c}")
            print(f"chosen: {guess_norm}, correct: {label}")

# Run from command line
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_type", type=str, default="gpt2", help="which model to load")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="which device to use")
    args = parser.parse_args()
    run_eval(args.model_type, args.device)