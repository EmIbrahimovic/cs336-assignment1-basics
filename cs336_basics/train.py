import os
import time
import typing

import numpy as np
import torch
from jaxtyping import Integer, Float
from typing import Any, Optional

from torch import Tensor, nn
from tqdm import tqdm
import wandb

from cs336_basics.optimizer import AdamW, cross_entropy, gradient_clipping
from cs336_basics.modules import TransformerLM, softmax
from cs336_basics.stupidargsmain import main
from cs336_basics.tokenizer import train_bpt_tokenizer, Tokenizer


def data_loader(x: Integer[np.ndarray, "n"], batch_size: int, context_length: int, device: str) -> tuple[Tensor, Tensor]:
    """
    Pair of PyTorch Tensors, both of shape batch_size, context_length, containing token IDs.
    Both are placed on the requested device.
    """
    start_indices = np.random.randint(low=0, high=x.shape[0] - context_length, size=(batch_size,))
    token_indices = start_indices[:, np.newaxis] + np.arange(context_length) # unsqueeze start_indices; (n, .) (conlen); (n, .) (1, conlen)
    target_indices = token_indices + 1

    all_tokens = torch.from_numpy(x[token_indices]).to(device)
    all_targets = torch.from_numpy(x[target_indices]).to(device)

    return all_tokens, all_targets


def save_checkpoint(model: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    iteration: int,
                    out: str|os.PathLike|typing.BinaryIO|typing.IO[bytes]):
    """
    Dump all the state from the model, optimizer and iteration into the file-like object out.
    """
    obj = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iteration': iteration
    }

    torch.save(obj, out)


def load_checkpoint(src: str|os.PathLike|typing.BinaryIO|typing.IO[bytes],
                    model: nn.Module,
                    optimizer: torch.optim.Optimizer):
    obj = torch.load(src)
    model.load_state_dict(obj['model'])
    optimizer.load_state_dict(obj['optimizer'])
    return obj['iteration']

def train(model_args: dict[str, Any],
          optimizer_args: dict[str, Any],
          train_file,
          num_steps: int,
          device: str,
          wandb_entity: str,
          wandb_project: str,
          checkpoint_args: Optional[dict[str, Any]]=None):

    # Model arguments
    vocab_size = model_args.get('vocab_size', 1000)
    batch_size = model_args.get('batch_size', 8)
    context_length = model_args.get('context_length', 250)
    num_layers = model_args.get('num_layers', 2)
    d_model = model_args.get('d_model', 128)
    num_heads = model_args.get('num_heads', 4)
    rope_theta = model_args.get('rope_theta', 10000)
    d_ff = model_args.get('d_ff', 4*d_model)

    betas = optimizer_args.get('betas', (0.9, 0.999))
    learning_rate = optimizer_args.get('learning_rate', 1e-4)
    weight_decay = optimizer_args.get('weight_decay', 0.01)

    checkpoint_path = checkpoint_args.get('path', './checkpoints')
    checkpoint_rate = checkpoint_args.get('rate', num_steps//2)

    trainset = np.load(train_file, mmap_mode='r')
    model = TransformerLM(d_model, num_heads, d_ff, vocab_size, context_length, num_layers, rope_theta, device)
    optimizer = AdamW(model.parameters(), learning_rate, weight_decay, betas)

    wandb.login()
    run = wandb.init(
        entity=wandb_entity,# Set the wandb entity where your project will be logged (generally your team name).
        project=wandb_project,# Set the wandb project where this run will be logged.
        config={
            "num_steps": num_steps,
            **model_args,
            **optimizer_args
        },
    )

    num_periods = 8
    steps_per_period = num_steps//num_periods

    start_time = time.time()

    for period in range(num_periods):
        for step in tqdm(range(steps_per_period), desc=f"Period {period + 1}/{num_periods} - [{period*steps_per_period + 1}, {(period+1)*steps_per_period}]/{num_steps}"):
            inputs, targets = data_loader(trainset, batch_size, context_length, device=device)
            current_lr = learning_rate # learning_rate_schedule(step, alpha_min=0, alpha_max=0.1, Tc=10, Tw=80)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = cross_entropy(outputs, targets)
            loss.backward()

            gradient_clipping(model.parameters(), max_norm=1.0)

            optimizer.step()

            if checkpoint_args is not None and (step + 1) % checkpoint_rate == 0:
                save_checkpoint(model, optimizer, step, out=os.path.join(checkpoint_path, f"_{step}"))

            run.log({"loss": loss.item(), "learning rate": current_lr, "clock_time": time.time() - start_time})

        print(f"Step {(period+1)*steps_per_period}/{num_steps} ~ Loss {loss.item():.4f} - LR: {current_lr:.2f} - Time: {time.time() - start_time}")

    run.finish()
    print("Training completed!")
    print("Running validation...")
    return model

# TODO test this one
def decode(model: nn.Module,
           eot_token: int,
           user_input: Float[Tensor, "seq_len"],
           max_tokens: Optional[int]=None,
           temperature: float=1,
           p_threshold: int=0) -> Float[Tensor, "output_seq_len"]:

    next_token = None

    next_input = user_input
    while next_token != eot_token and not (max_tokens is not None and next_input.shape[0] >= max_tokens):
        logits = model(next_input)

        q = softmax(logits/temperature, dim=-1)
        q_sorted = q.sort(descending=True)
        q_summed = q.cumsum(dim=-1)
        q_masked = torch.where(q_summed <= p_threshold, q_sorted.values, 0)

        p = q_masked / (q_masked.sum())

        next_token = np.random.choice(q_sorted.indices, p=p)

        next_input = torch.cat(next_input, next_token)

    return next_input


def small_test():
    train_file = '../data/TinyStoriesV2-GPT4-train.txt'
    shortened_train_file = '../data/TinyStoriesV2-GPT4-train_50k.txt'
    eot_token = "<|endoftext|>"
    with open(train_file, 'r') as f:
        train_text = f.read()
        shortened_train_text = train_text[:50000] + eot_token
    with open(shortened_train_file, 'w') as f:
        f.write(shortened_train_text)

    num_steps = 1000
    model_args = {
        'vocab_size': 1000,
        'batch_size': 8,
        'context_length': 250,
        'num_layers': 2,
        'd_model': 128,
        'num_heads': 4,
        'rope_theta': 10000,
        'd_ff': 4 * 128
    }

    vocab, merges = train_bpt_tokenizer(shortened_train_file, 1000, [eot_token])
    tokenizer = Tokenizer(vio)
    train(
        model_args, {},
    )

if __name__ == "__main__":
    raise SystemExit(main())