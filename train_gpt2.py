import torch
import torch.nn as nn
import torch.nn.functional as F

# This class handles the attention part of the transformer
class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_head
        self.hidden_size = config.n_embd

        # One big linear layer to get Q, K, V
        self.qkv = nn.Linear(self.hidden_size, 3 * self.hidden_size)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, x):
        batch_size, seq_len, hidden_size = x.size()

        # Split into query, key, value
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=2)

        # Reshape for multi-head attention
        head_dim = hidden_size // self.n_heads
        q = q.view(batch_size, seq_len, self.n_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, head_dim).transpose(1, 2)

        # Scaled dot-product attention (causal)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Bring heads back together
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        return self.out_proj(out)
    



class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x




class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.n_embd)
        self.attn = Attention(config)
        self.norm2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x




from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768




import inspect

# A simple version of the GPT model class
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Build the main transformer structure
        self.tok_embd = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_embd = nn.Embedding(config.block_size, config.n_embd)

        self.blocks = nn.ModuleList([
            Block(config) for _ in range(config.n_layer)
        ])

        self.layer_norm = nn.LayerNorm(config.n_embd)
        self.out_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Share weights between token embedding and output head
        self.out_head.weight = self.tok_embd.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx shape: (batch_size, sequence_length)
        B, T = idx.size()
        assert T <= self.config.block_size, "Input too long!"

        # Get embeddings
        pos = torch.arange(T, device=idx.device)
        tok_emb = self.tok_embd(idx)
        pos_emb = self.pos_embd(pos)
        x = tok_emb + pos_emb  # combine token + position info

        # Run through each transformer block
        for block in self.blocks:
            x = block(x)

        # Final normalization and get predictions
        x = self.layer_norm(x)
        logits = self.out_head(x)

        # If targets are given, compute loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Load pretrained weights from Hugging Face for GPT-2 models."""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel

        print(f"Loading pretrained weights for {model_type}...")

        # Choose config based on model type
        config_map = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),
        }
        
        config_args = config_map[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        config = GPTConfig(**config_args)
        model = cls(config)

        # Get state dicts
        my_weights = model.state_dict()
        hf_model = GPT2LMHeadModel.from_pretrained(model_type)
        hf_weights = hf_model.state_dict()

        # Ignore unnecessary buffers
        hf_keys = [k for k in hf_weights if not k.endswith(('.attn.bias', '.attn.masked_bias'))]
        my_keys = [k for k in my_weights if not k.endswith('.attn.bias')]

        assert len(hf_keys) == len(my_keys), "Mismatch in keys between models!"

        # Copy over weights
        transposed_keys = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        for k_hf, k_my in zip(hf_keys, my_keys):
            with torch.no_grad():
                if any(k_hf.endswith(name) for name in transposed_keys):
                    my_weights[k_my].copy_(hf_weights[k_hf].t())
                else:
                    my_weights[k_my].copy_(hf_weights[k_hf])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # Gather trainable parameters
        all_params = {name: p for name, p in self.named_parameters() if p.requires_grad}

        # Split into decayed and non-decayed groups
        decay_params = [p for name, p in all_params.items() if p.dim() >= 2]
        no_decay_params = [p for name, p in all_params.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]

        # Info for debugging
        if master_process:
            print(f"Decay params: {len(decay_params)}, total: {sum(p.numel() for p in decay_params):,}")
            print(f"No decay params: {len(no_decay_params)}, total: {sum(p.numel() for p in no_decay_params):,}")

        # Use fused AdamW if available on GPU
        fused_ok = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_ok and device_type == "cuda"
        if master_process:
            print(f"Using fused AdamW: {use_fused}")

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer







import os
import time
import math
import numpy as np
import tiktoken

# Load token data from the token file
def read_tok(path):
    arr = np.load(path).astype(np.int32)
    return torch.tensor(arr, dtype=torch.long)

# A simpler data loader for handling training/validation data
class DataLoader:
    def __init__(self, batch_size, seq_len, rnk_id, total_rnks, data_type):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.rnk_id = rnk_id
        self.total_rnks = total_rnks
        assert data_type in {'train', 'val'}

        folder = "edu_fineweb10B"
        files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if data_type in f])
        assert files, f"No data found for {data_type}"
        if master_is_running:
            print(f"{len(files)} files found for {data_type}")
        self.files = files
        self.reset()

    def reset(self):
        self.shard_ind = 0
        self.data = read_tok(self.files[self.shard_ind])
        self.curr_pos = self.batch_size * self.seq_len * self.rnk_id

    def get_batch(self):
        B, T = self.batch_size, self.seq_len
        data_chunk = self.data[self.curr_pos : self.curr_pos + B * T + 1]
        inputs = data_chunk[:-1].view(B, T)
        targets = data_chunk[1:].view(B, T)
        self.curr_pos += B * T * self.total_rnks

        if self.curr_pos + (B * T * self.total_rnks + 1) > len(self.data):
            self.shard_ind = (self.shard_ind + 1) % len(self.files)
            self.data = read_tok(self.files[self.shard_ind])
            self.curr_pos = B * T * self.rnk_id
        return inputs, targets

# Function to get which row (out of 4) has the lowest loss
def pick_best_answer(input_ids, input_mask, mod_logits):
    pred_logits = mod_logits[..., :-1, :].contiguous()
    actual_tokens = input_ids[..., 1:].contiguous()
    flat_logits = pred_logits.view(-1, pred_logits.size(-1))
    flat_targets = actual_tokens.view(-1)
    losses = F.cross_entropy(flat_logits, flat_targets, reduction='none').view(input_ids.size(0), -1)
    mask = input_mask[..., 1:]
    total_loss = (losses * mask).sum(dim=1)
    avg_loss = total_loss / mask.sum(dim=1)
    return avg_loss.argmin().item()

# Distributed training setup
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

is_distributed = int(os.environ.get('RANK', -1)) != -1
if is_distributed:
    assert torch.cuda.is_available(), "Need CUDA for distributed mode"
    init_process_group(backend='nccl')
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{local_rank}'
    torch.cuda.set_device(device)
    master_is_running = rank == 0
else:
    rank = local_rank = 0
    world_size = 1
    master_is_running = True
    device = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# Encoding and model setup
enc = tiktoken.get_encoding("gpt2")

# Training setup
total_tokens = 2**19
batch = 64
seq = 1024
assert total_tokens % (batch * seq * world_size) == 0
grad_steps = total_tokens // (batch * seq * world_size)

if master_is_running:
    print(f"Batch size: {total_tokens}")
    print(f"Steps to accumulate grads: {grad_steps}")

train_data = DataLoader(batch, seq, rank, world_size, "train")
val_data = DataLoader(batch, seq, rank, world_size, "val")

torch.set_float32_matmul_precision('high')

model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
if is_distributed:
    model = DDP(model, device_ids=[local_rank])
raw_model = model.module if is_distributed else model

use_compile = False
if use_compile:
    model = torch.compile(model)

# Learning rate schedule
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup = 715
total_steps = 19073
def learning_rate(step):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step > total_steps:
        return min_lr
    decay = (step - warmup) / (total_steps - warmup)
    return min_lr + 0.5 * (1 + math.cos(math.pi * decay)) * (max_lr - min_lr)

optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, device_type=device_type)

# Logs and checkpoints
log_path = "log"
os.makedirs(log_path, exist_ok=True)
log_file = os.path.join(log_path, "log.txt")
with open(log_file, "w"): pass

for step in range(total_steps):
    start_time = time.time()
    is_last = step == total_steps - 1

    if step % 250 == 0 or is_last:
        model.eval()
        val_data.reset()
        total_loss = 0.0
        with torch.no_grad():
            for _ in range(20):
                x, y = val_data.get_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    _, loss = model(x, y)
                total_loss += loss / 20
        if is_distributed:
            dist.all_reduce(total_loss, op=dist.ReduceOp.AVG)
        if master_is_running:
            print(f"Validation loss: {total_loss.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {total_loss.item():.4f}\n")
            if step > 0 and (step % 5000 == 0 or is_last):
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': total_loss.item()
                }
                torch.save(checkpoint, os.path.join(log_path, f"model_{step:05d}.pt"))

    if (step % 250 == 0 or is_last) and not use_compile:
        correct = total = 0
        for i, ex in enumerate(iterate_examples("val")):
            if i % world_size != rank:
                continue
            _, toks, msk, lbl = render_example(ex)
            toks, msk = toks.to(device), msk.to(device)
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    pred_logits, _ = model(toks)
                pred = pick_best_answer(toks, msk, pred_logits)
            total += 1
            correct += int(pred == lbl)
        if is_distributed:
            total = torch.tensor(total, dtype=torch.long, device=device)
            correct = torch.tensor(correct, dtype=torch.long, device=device)
            dist.all_reduce(total, op=dist.ReduceOp.SUM)
            dist.all_reduce(correct, op=dist.ReduceOp.SUM)
            total = total.item()
            correct = correct.item()
        if master_is_running:
            acc = correct / total
            print(f"HellaSwag accuracy: {correct}/{total} = {acc:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc:.4f}\n")

    if ((step > 0 and step % 250 == 0) or is_last) and not use_compile:
        model.eval()
        prompt = enc.encode("Hello, I'm a language model,")
        prompt = torch.tensor(prompt, dtype=torch.long).unsqueeze(0).repeat(4, 1).to(device)
        rng = torch.Generator(device=device).manual_seed(42 + rank)
        while prompt.size(1) < 32:
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, _ = model(prompt)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                top_probs, top_ids = torch.topk(probs, 50, dim=-1)
                next_tok = torch.gather(top_ids, -1, torch.multinomial(top_probs, 1, generator=rng))
                prompt = torch.cat((prompt, next_tok), dim=1)
        for i in range(4):
            decoded = enc.decode(prompt[i, :32].tolist())
            print(f"rank {rank} sample {i}: {decoded}")

    model.train()
    optimizer.zero_grad()
    loss_sum = 0.0
    for _ in range(grad_steps):
        xb, yb = train_data.get_batch()
        xb, yb = xb.to(device), yb.to(device)
        if is_distributed:
            model.require_backward_grad_sync = (_ == grad_steps - 1)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            _, loss = model(xb, yb)
        loss = loss / grad_steps
        loss_sum += loss.detach()
        loss.backward()
    if is_distributed:
        dist.all_reduce(loss_sum, op=dist.ReduceOp.AVG)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    curr_lr = learning_rate(step)
    for group in optimizer.param_groups:
        group['lr'] = curr_lr
    optimizer.step()
    if device_type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start_time
    tokens_done = batch * seq * grad_steps * world_size
    if master_is_running:
        print(f"step {step:5d} | loss: {loss_sum.item():.6f} | lr {curr_lr:.4e} | norm: {grad_norm:.4f} | time: {elapsed*1000:.2f}ms | tok/s: {tokens_done/elapsed:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_sum.item():.6f}\n")

if is_distributed:
    destroy_process_group()