from dataclasses import dataclass
import math
from typing import Any, Iterable, Callable, Dict, List, Iterator

import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset
from torch.nn.utils.rnn import pad_sequence

from .utils import Segment, _tokenize_segment


@dataclass
class _Bin:
    batch: List[int]
    used: int


class IterableDatasetBatcher(IterableDataset):
    def __init__(
        self,
        dataset: Iterable[Any],
        max_tokens: int = 10_000,
        max_batch_size: int = 64,
        drop_oversized_samples: bool = True,
    ):
        super().__init__()

        if not hasattr(dataset, "__iter__"):
            raise ValueError("dataset_to_wrap must be an iterable).")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer.")

        self.dataset = dataset
        self.max_tokens = max_tokens
        self.max_batch_size = max_batch_size
        self.drop_oversized_samples = drop_oversized_samples

    def __iter__(self) -> Iterator[List[Any]]:
        current_batch: List[Any] = []
        current_batch_tokens: int = 0
        dataset_iterator = iter(self.dataset)

        while True:
            try:
                sample = next(dataset_iterator)
            except StopIteration:
                break

            try:
                # sample_len = sample["json"].get("total_tokens_length", 0)
                audio_sample_len = len(sample["json"].get("audio_tokens", [])[0])
                text_sample_len = sample["json"].get("text_tokens_length", 0)
                sample_len = audio_sample_len + text_sample_len
            except Exception as e:
                print(f"Warning: Could not determine length for a sample. Skipping. Error: {e}")
                continue

            if sample_len <= 0:
                print(f"Warning: Skipping sample with non-positive length: {sample_len}")
                continue
                break

            if sample_len > self.max_tokens:
                if current_batch:
                    yield current_batch
                    current_batch = []
                    current_batch_tokens = 0

                if not self.drop_oversized_samples:
                    yield [sample]  # Yield oversized sample as its own batch
                else:
                    print(f"Warning: Dropping oversized sample (length {sample_len} > max_tokens {self.max_tokens}).")
                    continue

            if current_batch and ((len(current_batch) > self.max_batch_size) or (current_batch_tokens + sample_len > self.max_tokens)):
                yield current_batch
                current_batch = [sample]
                current_batch_tokens = sample_len
            else:
                current_batch.append(sample)
                current_batch_tokens += sample_len

        if current_batch:
            yield current_batch

    def __len__(self):
        raise TypeError(f"{self.__class__.__name__} does not have a defined length.")


def pad_to_multiple(t: torch.Tensor, multiple: int, value=0):
    B, S, *rest = t.shape
    target_len = math.ceil(S / multiple) * multiple
    pad_len = target_len - S
    if pad_len == 0:
        return t, 0
    padded = F.pad(t, (0, 0, 0, pad_len), value=value)
    return padded, pad_len


def tokenize_batch(batch: List[Segment], text_tokenizer: Callable | None, audio_num_codebooks: int = 32):
    tokens, tokens_mask = [], []
    for segment in batch:
        segment_tokens, segment_tokens_mask = _tokenize_segment(text_tokenizer, None, segment, audio_num_codebooks)
        tokens.append(segment_tokens)
        tokens_mask.append(segment_tokens_mask)

    padded_batch = pad_sequence(tokens, batch_first=True, padding_value=0)
    padded_batch_mask = pad_sequence(tokens_mask, batch_first=True, padding_value=False)

    return padded_batch, padded_batch_mask


def make_collate_fn(
    text_tokenizer: Callable | None = None,
    audio_num_codebooks: int = 32,
) -> Callable[[List[Dict]], Dict[str, torch.Tensor]]:
    def collate_fn(batch: List[Dict]):
        segments_for_batch: List[Segment] = []

        for sample_dict in batch[0]:
            sample_json = sample_dict["json"]

            if "audio_tokens" not in sample_json:
                print(f"Warning: 'audio_tokens' not found in sample: {sample_json.get('__key__', 'N/A')}")
                continue
            if not ("text_tokens" in sample_json or "text" in sample_json):
                print(f"Warning: Neither 'text_tokens' nor 'text' found in sample: {sample_json.get('__key__', 'N/A')}")
                continue

            audio_data = torch.as_tensor(sample_json["audio_tokens"], dtype=torch.long)

            if text_tokenizer is None or "text_tokens" in sample_json:
                text_data = torch.as_tensor(sample_json["text_tokens"], dtype=torch.long)
            else:
                text_data = sample_json["text"]

            if audio_data.ndim > 1 and audio_data.shape[0] != audio_num_codebooks:
                audio_data = audio_data[:audio_num_codebooks, :]

            current_segment = Segment(
                text=text_data,
                audio=audio_data,
                speaker=sample_json.get("speaker", 0),
            )
            segments_for_batch.append(current_segment)

        if not segments_for_batch:
            return {"tokens": torch.empty(0), "tokens_mask": torch.empty(0, dtype=torch.bool)}

        tokens, tokens_mask = tokenize_batch(segments_for_batch, text_tokenizer=text_tokenizer, audio_num_codebooks=audio_num_codebooks)

        return {
            "tokens": tokens,
            "tokens_mask": tokens_mask,
        }

    return collate_fn
