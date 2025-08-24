from dataclasses import dataclass
from typing import Tuple

import torch

from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer


@dataclass
class Segment:
    speaker: int
    text: str | torch.Tensor
    audio: torch.Tensor


def load_llama3_tokenizer():
    """
    https://github.com/huggingface/transformers/issues/22794#issuecomment-2092623992
    """
    tokenizer_name = "unsloth/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    bos = tokenizer.bos_token
    eos = tokenizer.eos_token
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f"{bos}:0 $A:0 {eos}:0",
        pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
        special_tokens=[(f"{bos}", tokenizer.bos_token_id), (f"{eos}", tokenizer.eos_token_id)],
    )
    return tokenizer


def load_smollm2_tokenizer():
    tokenizer_name = "HuggingFaceTB/SmolLM2-135M"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    bos = "<|im_start|>"
    eos = "<|im_end|>"
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f"{bos}:0 $A:0 {eos}:0",
        pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
        special_tokens=[(f"{bos}", tokenizer.bos_token_id), (f"{eos}", tokenizer.eos_token_id)],
    )
    return tokenizer

def _tokenize_text_segment(
    text_tokenizer, text: str | torch.Tensor, speaker: int, audio_num_codebooks: int = 32, device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(text, str):
        text_tokens = torch.tensor(text_tokenizer.encode(f"[{speaker}]{text}"))
    else:
        text_tokens = text
    text_frame = torch.zeros(len(text_tokens), audio_num_codebooks + 1).long()
    text_frame_mask = torch.zeros(len(text_tokens), audio_num_codebooks + 1).bool()
    text_frame[:, -1] = text_tokens
    text_frame_mask[:, -1] = True
    return text_frame.to(device), text_frame_mask.to(device)


def _tokenize_audio(audio_tokenizer, audio: torch.Tensor, audio_num_codebooks: int = 32, add_eos: bool = True, device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    if audio.ndim == 1:
        audio_tokens = audio_tokenizer.encode(audio.unsqueeze(0).unsqueeze(0))[0].to(device)
    else:
        audio_tokens = audio.to(device)

    # add EOS frame
    if add_eos:
        eos_frame = torch.zeros(audio_tokens.size(0), 1).to(device)
        audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)

    audio_frame = torch.zeros(audio_tokens.size(1), audio_num_codebooks + 1).long().to(device)
    audio_frame_mask = torch.zeros(audio_tokens.size(1), audio_num_codebooks + 1).bool().to(device)
    audio_frame[:, :-1] = audio_tokens.transpose(0, 1)
    audio_frame_mask[:, :-1] = True

    return audio_frame, audio_frame_mask


def _tokenize_segment(text_tokenizer, audio_tokenizer, segment: Segment, audio_num_codebooks: int = 32, add_eos: bool = True, device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        (seq_len, audio_num_codebooks + 1), (seq_len, audio_num_codebooks + 1)
    """
    text_tokens, text_masks = _tokenize_text_segment(text_tokenizer, segment.text, segment.speaker, audio_num_codebooks, device)
    audio_tokens, audio_masks = _tokenize_audio(audio_tokenizer, segment.audio, audio_num_codebooks, add_eos, device)
    return torch.cat([text_tokens, audio_tokens], dim=0), torch.cat([text_masks, audio_masks], dim=0)
