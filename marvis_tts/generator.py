from typing import List

from tqdm import tqdm, trange

import torch

from huggingface_hub import hf_hub_download
from moshi.models import loaders

from .models import Model
from .utils import Segment, _tokenize_segment, _tokenize_text_segment


class Generator:
    def __init__(
        self,
        model: Model,
        text_tokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self._model = model
        self._model.setup_caches(1)

        self._text_tokenizer = text_tokenizer

        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        mimi = loaders.get_mimi(mimi_weight, device=device)
        mimi.eval()

        mimi.set_num_codebooks(self._model.args.audio_num_codebooks)
        self._audio_tokenizer = mimi

        self.sample_rate = mimi.sample_rate
        self.device = device

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        speaker: int,
        context: List[Segment],
        max_audio_length_ms: float = 90_000,
        temperature: float = 0.9,
        topk: int = 50,
        voice_match: bool = False,
    ) -> torch.Tensor:
        self._model.reset_caches()

        max_audio_frames = int(max_audio_length_ms / 80)
        tokens, tokens_mask = [], []
        for segment in context:
            segment_tokens, segment_tokens_mask = _tokenize_segment(
                self._text_tokenizer,
                self._audio_tokenizer,
                segment,
                self._model.args.audio_num_codebooks,
                add_eos=not voice_match,
                device=self.device,
            )
            tokens.append(segment_tokens)
            tokens_mask.append(segment_tokens_mask)

        if not voice_match:
            gen_segment_tokens, gen_segment_tokens_mask = _tokenize_text_segment(
                self._text_tokenizer, text, speaker, self._model.args.audio_num_codebooks, self.device
            )
            tokens.append(gen_segment_tokens)
            tokens_mask.append(gen_segment_tokens_mask)

        prompt_tokens = torch.cat(tokens, dim=0).long().to(self.device)
        prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(self.device)

        samples = []
        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device)

        max_seq_len = 2048 - max_audio_frames
        if curr_tokens.size(1) >= max_seq_len:
            raise ValueError(f"Inputs too long, must be below max_seq_len - max_audio_frames: {max_seq_len}")

        for _ in trange(max_audio_frames):
            sample = self._model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
            if torch.all(sample == 0):
                break  # eos

            samples.append(sample)

            curr_tokens = torch.cat([sample, torch.zeros(1, 1).long().to(self.device)], dim=1).unsqueeze(1)
            curr_tokens_mask = torch.cat([torch.ones_like(sample).bool(), torch.zeros(1, 1).bool().to(self.device)], dim=1).unsqueeze(1)
            curr_pos = curr_pos[:, -1:] + 1

        print(f"semantic codebook 0: {torch.stack(samples)[:, 0, 0].tolist()}")

        audio = self._audio_tokenizer.decode(torch.stack(samples).permute(1, 2, 0))[-1, -1, :]
        return audio
