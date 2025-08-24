from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .flavors import FLAVORS


def _prepare_transformer(model):
    embed_dim = model.tok_embeddings.embedding_dim
    model.tok_embeddings = nn.Identity()
    model.output = nn.Identity()
    return model, embed_dim


def _create_causal_mask(seq_len: int, device: torch.device):
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))


def _index_causal_mask(mask: torch.Tensor, input_pos: torch.Tensor):
    """
    Args:
        mask: (max_seq_len, max_seq_len)
        input_pos: (batch_size, seq_len)

    Returns:
        (batch_size, seq_len, max_seq_len)
    """
    r = mask[input_pos, :]
    return r


def _multinomial_sample_one_no_sync(probs):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)


def sample_topk(logits: torch.Tensor, topk: int, temperature: float):
    logits = logits / temperature

    filter_value: float = -float("Inf")
    indices_to_remove = logits < torch.topk(logits, topk)[0][..., -1, None]
    scores_processed = logits.masked_fill(indices_to_remove, filter_value)
    scores_processed = torch.nn.functional.log_softmax(scores_processed, dim=-1)
    probs = torch.nn.functional.softmax(scores_processed, dim=-1)

    sample_token = _multinomial_sample_one_no_sync(probs)
    return sample_token


def _pad_seq_to(x, tgt_len, value):
    B, S, *rest = x.shape
    if S == tgt_len:
        valid = x.new_ones(B, S, dtype=torch.bool)
        return x, valid

    pad_len = tgt_len - S
    pad_shape = (B, pad_len, *rest)
    pad = x.new_full(pad_shape, value)
    x_padded = torch.cat([x, pad], dim=1)

    valid = torch.cat([x.new_ones(B, S, dtype=torch.bool), x.new_zeros(B, pad_len, dtype=torch.bool)], dim=1)
    return x_padded, valid


@dataclass
class ModelArgs:
    backbone_flavor: str
    decoder_flavor: str
    text_vocab_size: int
    audio_vocab_size: int
    audio_num_codebooks: int
    pad_multiple: int = 32


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.backbone, backbone_dim = _prepare_transformer(FLAVORS[args.backbone_flavor]())
        self.decoder, decoder_dim = _prepare_transformer(FLAVORS[args.decoder_flavor]())

        self.text_embeddings = nn.Embedding(args.text_vocab_size, backbone_dim)
        nn.init.normal_(self.text_embeddings.weight, mean=0.0, std=1.0 / math.sqrt(self.text_embeddings.embedding_dim))

        self.audio_embeddings = nn.Embedding(args.audio_vocab_size * args.audio_num_codebooks, backbone_dim)
        nn.init.normal_(self.audio_embeddings.weight, mean=0.0, std=1.0 / math.sqrt(self.audio_embeddings.embedding_dim))

        self.projection = nn.Linear(backbone_dim, decoder_dim, bias=False)
        nn.init.normal_(self.projection.weight, mean=0.0, std=1.0 / math.sqrt(self.projection.in_features))

        self.codebook0_head = nn.Linear(backbone_dim, args.audio_vocab_size, bias=False)
        nn.init.zeros_(self.codebook0_head.weight)

        self.audio_head = nn.Parameter(torch.empty(args.audio_num_codebooks - 1, decoder_dim, args.audio_vocab_size))
        nn.init.zeros_(self.audio_head)

    def setup_caches(self, max_batch_size: int, training: bool = False):
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device

        if not training:
            with device:
                self.backbone.setup_caches(max_batch_size, dtype)
                self.decoder.setup_caches(max_batch_size, dtype, decoder_max_seq_len=self.args.audio_num_codebooks)

        self.register_buffer("backbone_causal_mask", _create_causal_mask(self.backbone.max_seq_len, device), persistent=False)
        self.register_buffer("decoder_causal_mask", _create_causal_mask(self.args.audio_num_codebooks, device), persistent=False)

    def _embed_audio(self, codebook: int, tokens: torch.Tensor) -> torch.Tensor:
        return self.audio_embeddings(tokens + codebook * self.args.audio_vocab_size)

    def _embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        text_embeds = self.text_embeddings(tokens[:, :, -1]).unsqueeze(-2)

        audio_tokens = tokens[:, :, :-1] + (self.args.audio_vocab_size * torch.arange(self.args.audio_num_codebooks, device=tokens.device))
        audio_embeds = self.audio_embeddings(audio_tokens.view(-1)).reshape(
            tokens.size(0), tokens.size(1), self.args.audio_num_codebooks, -1
        )

        return torch.cat([audio_embeds, text_embeds], dim=-2)

    def _forward_inner(
        self,
        tokens: torch.Tensor,  # (b, s, c) where c = audio_num_codebooks + 1
        tokens_mask: torch.Tensor,  # (b, s, c)
        decoder_fraction: float,  # compute-amortisation ratio
    ):
        dtype, device = next(self.parameters()).dtype, tokens.device
        b, s, c = tokens.shape
        n = self.args.audio_num_codebooks
        if s < 2:
            raise ValueError("sequence must contain at least two frames for next‑frame training")

        # backbone
        tokens_in = tokens[:, :-1, :]  # (b, s-1, c)
        tokens_mask_in = tokens_mask[:, :-1, :]  # (b, s-1, c)
        s_in = s - 1

        input_pos = torch.arange(s_in, device=device).unsqueeze(0).expand(b, s_in)  # (b, s-1)
        curr_backbone_mask = _index_causal_mask(self.backbone_causal_mask, input_pos)[:, :, :s_in]

        embeds_in = self._embed_tokens(tokens_in)  # (b, s-1, c, dim)
        masked_embeds = (embeds_in * tokens_mask_in.unsqueeze(-1)).sum(dim=2)  # (b, s-1, dim)

        h = self.backbone(masked_embeds, input_pos=input_pos, mask=curr_backbone_mask).to(dtype=dtype)  # (b, s-1, dim)

        # codebook 0 (semantic) loss
        tokens_tgt = tokens[:, 1:, :]  # (b, s-1, c)
        tokens_mask_tgt = tokens_mask[:, 1:, :]  # (b, s-1, c)

        c0_mask = tokens_mask_tgt[:, :, 0]  # (b, s-1)
        c0_h = h[c0_mask]  # (m, dim)
        c0_tgt = tokens_tgt[:, :, 0][c0_mask]  # (m,)
        c0_logits = self.codebook0_head(c0_h)  # (m, v_a)
        c0_loss = F.cross_entropy(c0_logits, c0_tgt, reduction="mean")
        c0_accuracy = (c0_logits.argmax(dim=-1) == c0_tgt).float().mean()

        # decoder
        frame_idx = torch.nonzero(c0_mask, as_tuple=False)  # (m, 2)
        keep = torch.rand(len(frame_idx), device=device) < decoder_fraction
        sel_idx = frame_idx[keep]

        if sel_idx.numel() == 0:
            decoder_loss = torch.zeros((), device=device, dtype=dtype)
            per_level_losses = []
        else:
            sb, ss = sel_idx[:, 0], sel_idx[:, 1]
            last_h = h[sb, ss]  # (msel, dim)
            tgt_row = tokens_tgt[sb, ss]  # (msel, c)
            c0_gt = tgt_row[:, 0]  # (msel,)
            audio_gt = tgt_row[:, 1:-1]  # (msel, n-1)

            # [h_t, c0, c1 … c_{n-2}]
            seq_parts = [
                last_h.unsqueeze(1),
                self._embed_audio(0, c0_gt).unsqueeze(1),
            ]
            for k in range(1, n - 1):
                seq_parts.append(self._embed_audio(k, audio_gt[:, k - 1]).unsqueeze(1))
            dec_in = torch.cat(seq_parts, dim=1)  # (msel, n, dim_backbone)
            proj_h = self.projection(dec_in)  # (msel, n, dim_decoder)

            pos = torch.arange(n, device=device).unsqueeze(0).expand(proj_h.size(0), -1)
            dec_mask = _index_causal_mask(self.decoder_causal_mask, pos)[:, :, :n]

            decoder_h = self.decoder(proj_h, input_pos=pos, mask=dec_mask)  # (msel, n-1, dim_dec)

            dec_h_no_c0 = decoder_h[:, 1:, :]
            logits = torch.einsum("bkd,kdv->bkv", dec_h_no_c0, self.audio_head)
            token_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                audio_gt.reshape(-1),
                reduction="none",
            ).view(logits.size(0), logits.size(1))
            per_level_losses = token_loss.mean(dim=0)
            decoder_loss = per_level_losses.mean()

        total_loss = c0_loss + decoder_loss
        return {
            "loss": total_loss,
            "backbone_loss": c0_loss.detach(),
            "decoder_loss": decoder_loss.detach(),
            "c0_accuracy": c0_accuracy.detach(),
            "per_level_losses": [p.detach() for p in per_level_losses],
        }

    def forward(
        self,
        tokens,  # (B,S,C)
        tokens_mask,  # (B,S,C)
        decoder_fraction=1 / 16,
    ):
        if self.args.pad_multiple > 1:
            tgt_len = math.ceil(tokens.shape[1] / self.args.pad_multiple) * self.args.pad_multiple
            tokens, valid = _pad_seq_to(tokens, tgt_len, value=0)
            tokens_mask, _ = _pad_seq_to(tokens_mask, tgt_len, value=False)
            tokens_mask = tokens_mask & valid.unsqueeze(-1)
        return self._forward_inner(tokens, tokens_mask, decoder_fraction)

    def generate_frame(
        self,
        tokens: torch.Tensor,
        tokens_mask: torch.Tensor,
        input_pos: torch.Tensor,
        temperature: float,
        topk: int,
    ) -> torch.Tensor:
        """
        Args:
            tokens: (batch_size, seq_len, audio_num_codebooks+1)
            tokens_mask: (batch_size, seq_len, audio_num_codebooks+1)
            input_pos: (batch_size, seq_len) positions for each token
            mask: (batch_size, seq_len, max_seq_len

        Returns:
            (batch_size, audio_num_codebooks) sampled tokens
        """
        dtype = next(self.parameters()).dtype
        b, s, _ = tokens.size()

        assert self.backbone.caches_are_enabled(), "backbone caches are not enabled"
        curr_backbone_mask = _index_causal_mask(self.backbone_causal_mask, input_pos)
        embeds = self._embed_tokens(tokens)
        masked_embeds = embeds * tokens_mask.unsqueeze(-1)
        h = masked_embeds.sum(dim=2)
        h = self.backbone(h, input_pos=input_pos, mask=curr_backbone_mask).to(dtype=dtype)

        last_h = h[:, -1, :]
        c0_logits = self.codebook0_head(last_h)
        c0_sample = sample_topk(c0_logits, topk, temperature)
        c0_embed = self._embed_audio(0, c0_sample)

        curr_h = torch.cat([last_h.unsqueeze(1), c0_embed], dim=1)
        curr_sample = c0_sample.clone()
        curr_pos = torch.arange(0, curr_h.size(1), device=curr_h.device).unsqueeze(0).repeat(curr_h.size(0), 1)

        # Decoder caches must be reset every frame.
        self.decoder.reset_caches()
        for i in range(1, self.args.audio_num_codebooks):
            curr_decoder_mask = _index_causal_mask(self.decoder_causal_mask, curr_pos)
            decoder_proj_h = self.projection(curr_h)
            decoder_h = self.decoder(decoder_proj_h, input_pos=curr_pos, mask=curr_decoder_mask).to(dtype=dtype)
            ci_logits = torch.mm(decoder_h[:, -1, :], self.audio_head[i - 1])
            ci_sample = sample_topk(ci_logits, topk, temperature)
            ci_embed = self._embed_audio(i, ci_sample)

            curr_h = ci_embed
            curr_sample = torch.cat([curr_sample, ci_sample], dim=1)
            curr_pos = curr_pos[:, -1:] + 1

        return curr_sample

    def reset_caches(self):
        self.backbone.reset_caches()
        self.decoder.reset_caches()
