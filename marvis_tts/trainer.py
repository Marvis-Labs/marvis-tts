from pathlib import Path
import time

import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration, DistributedDataParallelKwargs

from tqdm import tqdm

from .models import Model


class Trainer:
    def __init__(
        self,
        model: Model,
        optimizer: optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        num_warmup_steps=1_000,
        max_grad_norm: float = 1.0,
        decoder_fraction: float = 1 / 16,
        checkpoint_dir: str = "./checkpoints",
        accelerate_kwargs: dict = dict(),
        ddp_kwargs: DistributedDataParallelKwargs = DistributedDataParallelKwargs(
            find_unused_parameters=False,
        ),
    ):
        self.accelerator = Accelerator(
            dataloader_config=DataLoaderConfiguration(dispatch_batches=False),
            kwargs_handlers=[ddp_kwargs],
            **accelerate_kwargs,
        )
        self.print = self.accelerator.print

        self.model = model
        self.model.setup_caches(1, training=True)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_warmup_steps = num_warmup_steps
        self.max_grad_norm = max_grad_norm
        self.decoder_fraction = decoder_fraction
        self.optimizer = optimizer

        if self.val_loader is not None:
            (
                self.model,
                self.optimizer,
                self.train_loader,
                self.val_loader,
            ) = self.accelerator.prepare(
                self.model,
                self.optimizer,
                self.train_loader,
                self.val_loader,
            )
        else:
            (
                self.model,
                self.optimizer,
                self.train_loader,
            ) = self.accelerator.prepare(
                self.model,
                self.optimizer,
                self.train_loader,
            )

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        def num_params(model: nn.Module) -> int:
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.print(f"Backbone parameter count: {num_params(model.backbone):,}")
        self.print(f"Decoder parameter count: {num_params(model.decoder):,}")
        self.print(f"Text embeddings parameter count: {num_params(model.text_embeddings):,}")
        self.print(f"Audio embeddings parameter count: {num_params(model.audio_embeddings):,}")
        self.print(f"Cookbook 0 head parameter count: {num_params(model.codebook0_head):,}")
        self.print(f"*Total parameter count*: {num_params(model):,}")

    def save_checkpoint(self, global_step, epoch, total_tokens, force_sync=True):
        try:
            if force_sync:
                self.accelerator.wait_for_everyone()

            unwrapped = self.accelerator.unwrap_model(self.model)
            ckpt = {
                "global_step": global_step,
                "model_state": unwrapped.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict() if hasattr(self, "scheduler") else None,
                "epoch": epoch,
                "total_tokens": total_tokens,
            }
            path = self.checkpoint_dir / f"model_{global_step}.pt"
            torch.save(ckpt, path)
            self.print(f"Saved checkpoint: {path}")

        except Exception as e:
            self.print(f"Warning: Failed to save checkpoint at step {global_step}: {e}")

    def safe_backward_and_step(self, loss, global_step):
        try:
            self.accelerator.backward(loss)

            if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.scheduler.step()
            self.optimizer.step()
            self.optimizer.zero_grad()

            return True

        except RuntimeError as e:
            if "timeout" in str(e).lower() or "nccl" in str(e).lower():
                self.print(f"NCCL timeout detected at step {global_step}: {e}")
                # try to recover by skipping this step
                self.optimizer.zero_grad()
                return False
            else:
                raise e

    def train(
        self,
        total_steps: int = 1_000_000,
        validate_every: int = 1000,
        save_every: int = 10_000,
        use_cosine_annealing: bool = True,
        start_step=None,
        epoch=None,
        scheduler_state=None,
    ):
        decay_steps = total_steps - self.num_warmup_steps
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1e-2,
            end_factor=1.0,
            total_iters=self.num_warmup_steps,
        )
        if not use_cosine_annealing:
            decay_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_steps)
        else:
            decay_scheduler = CosineAnnealingLR(self.optimizer, T_max=decay_steps, eta_min=1e-8)

        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, decay_scheduler],
            milestones=[self.num_warmup_steps],
            last_epoch=-1,
        )
        if scheduler_state:
            self.scheduler.load_state_dict(scheduler_state)
            print("Scheduler state loaded")
        self.scheduler = self.accelerator.prepare(self.scheduler)

        hps = {
            "total_steps": total_steps,
            "num_warmup_steps": self.num_warmup_steps,
            "save_every": save_every,
            "validate_every": validate_every,
        }
        self.accelerator.init_trackers("marvis-tts", config=hps)

        global_step = start_step if start_step else 0
        total_tokens = 0
        consecutive_timeouts = 0
        epoch = epoch if epoch else 1
        max_consecutive_timeouts = 3

        while global_step < total_steps:
            self.optimizer.train()
            self.model.train()

            # set epoch for distributed sampler if needed
            if hasattr(self.train_loader.batch_sampler, "set_epoch"):
                self.train_loader.batch_sampler.set_epoch(epoch)

            running_loss = 0.0
            num_batches = 0

            progress_bar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch}",
                unit="",
                disable=not self.accelerator.is_local_main_process,
            )

            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

            for batch_idx, batch in enumerate(progress_bar):
                if global_step >= total_steps:
                    break

                tokens = batch["tokens"]
                tokens_mask = batch["tokens_mask"]
                total_tokens += tokens.shape[0] * tokens.shape[1]

                with torch.autocast(device_type=device_type, dtype=dtype, enabled=True):
                    losses = self.model(tokens, tokens_mask, decoder_fraction=self.decoder_fraction)

                loss = losses["loss"]

                success = self.safe_backward_and_step(loss, global_step)

                if not success:
                    consecutive_timeouts += 1
                    if consecutive_timeouts >= max_consecutive_timeouts:
                        self.print("Max consecutive timeouts reached. Stopping training.")
                        break
                    continue
                else:
                    consecutive_timeouts = 0

                running_loss += loss.item()
                num_batches += 1
                global_step += 1

                if self.accelerator.is_local_main_process and global_step % 10 == 0:
                    backbone_loss = losses["backbone_loss"]
                    decoder_loss = losses["decoder_loss"]
                    c0_accuracy = losses["c0_accuracy"]

                    self.accelerator.log(
                        {
                            "loss": loss.item(),
                            "backbone_loss": backbone_loss.item(),
                            "decoder_loss": decoder_loss.item(),
                            "c0_accuracy": c0_accuracy.item(),
                            "per_level_losses": {f"codebook_{i + 1}": loss.item() for i, loss in enumerate(losses["per_level_losses"])},
                            "total_tokens": total_tokens,
                            "lr": self.scheduler.get_last_lr()[0],
                            "epoch": epoch,
                        },
                        step=global_step,
                    )

                    progress_bar.set_postfix(
                        loss=f"{loss.item():.3f}",
                        backbone_loss=f"{backbone_loss.item():.3f}",
                        c0_acc=f"{c0_accuracy.item():.3f}",
                        decoder_loss=f"{decoder_loss.item():.3f}",
                        tokens=f"{total_tokens / 1_000_000:.2f}M",
                    )

                    if global_step % save_every == 0:
                        self.save_checkpoint(global_step, epoch, total_tokens)

            avg_loss = running_loss / num_batches
            self.print(f"Epoch {epoch} average loss: {avg_loss:.4f}")
            self.accelerator.log({"avg_epoch_loss": avg_loss}, step=global_step)

            if self.val_loader and global_step % validate_every == 0:
                val_loss = self.validate_epoch(global_step)
                if val_loss is not None:
                    self.accelerator.log({"val_loss": val_loss}, step=global_step)

            epoch += 1
            if consecutive_timeouts >= max_consecutive_timeouts:
                break

        self.save_checkpoint("final", epoch, total_tokens)
        self.accelerator.end_training()
        self.print("Training complete.")

    @torch.no_grad()
    def validate_epoch(self, global_step: int):
        self.optimizer.eval()
        if self.val_loader is None:
            return None

        self.model.eval()
        running_loss = 0.0

        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        try:
            for batch in self.val_loader:
                tokens = batch["tokens"]
                tokens_mask = batch["tokens_mask"]

                with torch.autocast(device_type=device_type, dtype=dtype, enabled=True):
                    loss = self.model(tokens, tokens_mask)["loss"]

                running_loss += loss.item()

            avg_loss = running_loss / len(self.val_loader)
            self.print(f"Step {global_step} validation loss: {avg_loss:.4f}")
            return avg_loss

        except Exception as e:
            self.print(f"Validation failed at step {global_step}: {e}")
            return None
        finally:
            self.model.train()
