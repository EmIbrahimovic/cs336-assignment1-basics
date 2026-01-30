import argparse
import os
from typing import Optional

import torch

from cs336_basics.train import train


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train a custom TransformerLM with a custom optimizer/training loop.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required / core
    parser.add_argument("--train-file", type=str, required=True, help="Path to a .npy file of token ids.")
    parser.add_argument("--num-steps", type=int, required=True, help="Total training steps.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # W&B
    parser.add_argument("--wandb-entity", type=str, default=os.getenv("WANDB_ENTITY", None))
    parser.add_argument("--wandb-project", type=str, default=os.getenv("WANDB_PROJECT", None))
    parser.add_argument(
        "--wandb-mode",
        type=str,
        choices=("online", "offline", "disabled"),
        default=os.getenv("WANDB_MODE", "online"),
        help="W&B mode. Use 'disabled' to avoid logging.",
    )

    # Model hyperparameters
    g_model = parser.add_argument_group("model")
    g_model.add_argument("--vocab-size", type=int, default=50257)
    g_model.add_argument("--batch-size", type=int, default=8)
    g_model.add_argument("--context-length", type=int, default=1024)
    g_model.add_argument("--num-layers", type=int, default=48)
    g_model.add_argument("--d-model", type=int, default=1600)
    g_model.add_argument("--num-heads", type=int, default=25)
    g_model.add_argument("--rope-theta", type=float, default=10000.0)
    g_model.add_argument("--d-ff", type=int, default=None, help="Defaults to 4*d_model if not set.")

    # Optimizer hyperparameters
    g_opt = parser.add_argument_group("optimizer")
    g_opt.add_argument("--learning-rate", type=float, default=1e-4)
    g_opt.add_argument("--weight-decay", type=float, default=0.01)
    g_opt.add_argument("--betas", type=_parse_betas, default=(0.9, 0.999))

    # Checkpointing
    g_ckpt = parser.add_argument_group("checkpoint")
    g_ckpt.add_argument("--checkpoint-path", type=str, default="./checkpoints")
    g_ckpt.add_argument(
        "--checkpoint-rate",
        type=int,
        default=None,
        help="Save every N steps. If omitted, defaults to num_steps//2 inside train().",
    )
    g_ckpt.add_argument("--no-checkpoints", action="store_true", help="Disable checkpoint saving entirely.")

    args = parser.parse_args(argv)

    # Make W&B mode effective before train() calls wandb.login/init.
    os.environ["WANDB_MODE"] = args.wandb_mode

    if not args.wandb_entity or not args.wandb_project:
        parser.error(
            "--wandb-entity and --wandb-project are required (or set WANDB_ENTITY/WANDB_PROJECT env vars). "
            "If you want no logging, still pass placeholders and use --wandb-mode disabled."
        )

    model_args = {
        "vocab_size": args.vocab_size,
        "batch_size": args.batch_size,
        "context_length": args.context_length,
        "num_layers": args.num_layers,
        "d_model": args.d_model,
        "num_heads": args.num_heads,
        "rope_theta": args.rope_theta,
    }
    if args.d_ff is not None:
        model_args["d_ff"] = args.d_ff  # otherwise train() will use 4*d_model

    optimizer_args = {
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "betas": args.betas,
    }

    if args.no_checkpoints:
        checkpoint_args = None
    else:
        os.makedirs(args.checkpoint_path, exist_ok=True)
        checkpoint_args = {"path": args.checkpoint_path}
        if args.checkpoint_rate is not None:
            checkpoint_args["rate"] = args.checkpoint_rate

    train(
        model_args=model_args,
        optimizer_args=optimizer_args,
        train_file=args.train_file,
        num_steps=args.num_steps,
        device=args.device,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        checkpoint_args=checkpoint_args,
    )
    return 0


def _parse_betas(s: str) -> tuple[float, float]:
    """
    Parse betas from either "0.9,0.999" or "0.9 0.999".
    """
    parts = [p for p in s.replace(",", " ").split() if p]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected two floats for betas, e.g. '0.9,0.999' or '0.9 0.999'.")
    try:
        b1 = float(parts[0])
        b2 = float(parts[1])
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid betas: {s!r}") from e
    return b1, b2
