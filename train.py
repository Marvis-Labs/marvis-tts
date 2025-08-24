import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from marvis_tts.data import make_collate_fn, IterableDatasetBatcher
from marvis_tts.models import Model, ModelArgs
from marvis_tts.trainer import Trainer
from marvis_tts.utils import load_llama3_tokenizer, load_smollm2_tokenizer

from datasets import load_dataset


@dataclass
class TrainingConfig:
    backbone_flavor: str
    decoder_flavor: str
    tokenizer: str
    dataset_repo_id: str
    audio_num_codebooks: int = 32
    learning_rate: float = 1e-4
    max_tokens: int = 10_000
    max_batch_size: int = 64
    device: str = "cuda"
    precision: str = "bf16"
    pad_multiple: int = 64
    decoder_fraction: float = 1 / 16
    freeze_backbone: bool = False
    resume_from_checkpoint: str | None = None
    finetune: bool = False


def load_checkpoint(checkpoint_path, model, optimizer=None, finetune=False):
    """Load model, optimizer, scheduler state and return current step."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")

    model.load_state_dict(checkpoint["model_state"])
    print("Model state loaded.")

    if not finetune and optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        print("Optimizer state loaded.")

    scheduler_state = None
    if not finetune and "scheduler_state" in checkpoint and checkpoint["scheduler_state"] is not None:
        scheduler_state = checkpoint["scheduler_state"]

    start_step = checkpoint.get("global_step", 0) if not finetune else 0
    print(f"Resuming from step: {start_step}")

    return model, optimizer, scheduler_state, start_step


def train(config: TrainingConfig):
    if config.tokenizer == "smollm2":
        text_tokenizer = load_smollm2_tokenizer()
        vocab_size = text_tokenizer.vocab_size
    elif config.tokenizer == "llama3":
        text_tokenizer = load_llama3_tokenizer()
        vocab_size = 128_256
    else:
        raise ValueError(f"Unknown tokenizer: {config.tokenizer}")

    def create_model(device: str = "cuda", dtype: torch.dtype = torch.bfloat16) -> Model:
        print(f"Using vocab size: {vocab_size}")

        model_args = ModelArgs(
            backbone_flavor=config.backbone_flavor,
            decoder_flavor=config.decoder_flavor,
            text_vocab_size=vocab_size,
            audio_vocab_size=2051,
            audio_num_codebooks=config.audio_num_codebooks,
            pad_multiple=config.pad_multiple,
        )
        return Model(model_args).to(device=device, dtype=dtype)

    device = config.device
    dtype = torch.bfloat16 if config.precision == "bf16" else torch.float32
    model = create_model(device=device, dtype=dtype)

    if config.freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad_(False)

    # use disk cache or not
    use_local_shards = True

    if use_local_shards:
        # load wds shards directly from disk cache
        hf_cache_dir = os.getenv("HF_DATASETS_CACHE", "~/.cache/huggingface/hub")
        repo_path = "datasets--" + config.dataset_repo_id.replace("/", "--")
        dataset_path = Path(f"{hf_cache_dir}/{repo_path}/").expanduser()
        shards = [p.as_posix() for p in sorted(list(dataset_path.rglob("**/*.tar")))]
        print(f"Using dataset path: {dataset_path} with {len(shards)} shards")

        dataset = load_dataset("webdataset", streaming=True, split="train", data_files={"train": shards})
        dataset = dataset.shuffle(seed=42)
        print(f"Using dataset: {dataset}")
    else:
        # otherwise stream
        dataset = load_dataset(config.dataset_repo_id, streaming=True, split="train")
        dataset = dataset.shuffle(buffer_size=1_000, seed=42)
        print(f"Using dataset: {dataset}")

    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [{"params": decay_params, "weight_decay": 1e-2}, {"params": nodecay_params, "weight_decay": 0.0}]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"Total decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"Total non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

    optimizer = AdamW(optim_groups, lr=config.learning_rate, betas=(0.9, 0.95))

    start_step = None
    scheduler_state = None
    if config.resume_from_checkpoint:
        checkpoint_path = Path(config.resume_from_checkpoint).expanduser()
        if checkpoint_path.exists():
            model, optimizer, scheduler_state, start_step = load_checkpoint(checkpoint_path, model, optimizer, finetune=config.finetune)
        else:
            print(f"Checkpoint not found: {checkpoint_path}")

    batched_dataset = IterableDatasetBatcher(
        dataset,
        max_tokens=config.max_tokens,
        max_batch_size=config.max_batch_size,
        drop_oversized_samples=True,
    )
    collate_fn = make_collate_fn(text_tokenizer=text_tokenizer, audio_num_codebooks=config.audio_num_codebooks)
    train_dl = DataLoader(
        batched_dataset,
        collate_fn=collate_fn,
        num_workers=16 if device == "cuda" else 0,
        pin_memory=True if device == "cuda" else False,
        prefetch_factor=4 if device == "cuda" else None,
        persistent_workers=True if device == "cuda" else False,
    )

    use_wandb = False
    try:
        import wandb

        use_wandb = True
    except ImportError:
        pass

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_dl,
        num_warmup_steps=5000,
        decoder_fraction=config.decoder_fraction,
        accelerate_kwargs={
            "log_with": "wandb" if use_wandb else None,
            "mixed_precision": "bf16" if config.precision == "bf16" else "no",
        },
    )
    trainer.train(total_steps=2_000_000, start_step=start_step, scheduler_state=scheduler_state)


# Usage: `accelerate launch train.py <config>`

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "config",
        type=str,
        help="Path to the training configuration file.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config_data = json.load(f)
    config = TrainingConfig(**config_data)

    train(config)
