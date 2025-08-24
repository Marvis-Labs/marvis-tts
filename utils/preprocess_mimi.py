import os
import json
from pathlib import Path

from tqdm import tqdm

import torch
import torchaudio

from moshi.models import loaders
from huggingface_hub import hf_hub_download
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
mimi = loaders.get_mimi(mimi_weight, device=device)
mimi.set_num_codebooks(32)
mimi.eval()


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


text_tokenizer = load_smollm2_tokenizer()


def process_file(filepath):
    output_path = Path(filepath).with_suffix(".json")
    if output_path.exists():
        return

    wave, sr = torchaudio.load(filepath)
    wave = wave[None, ...].to(device)
    audio_tokens = mimi.encode(wave)
    audio_tokens = audio_tokens.squeeze(0).cpu().tolist()

    text_path = output_path.with_suffix(".txt")
    with open(text_path, "r") as f:
        text = f.read().strip()

    text_tokens = text_tokenizer.encode(text)

    duration = wave.shape[1] / sr
    metadata = {
        "duration": duration,
        "speaker_id": filepath.split("/")[-2],
        "text": text,
        "audio_tokens": audio_tokens,
        "audio_tokens_length": len(audio_tokens[0]),
        "text_tokens": text_tokens,
        "text_tokens_length": len(text_tokens),
        "total_tokens_length": len(audio_tokens[0]) + len(text_tokens),
    }
    with open(output_path, "w") as f:
        json.dump(metadata, f)


def process_directory(directory):
    for root, dirs, files in tqdm(os.walk(directory)):
        for file in tqdm(files):
            if file.endswith(".wav"):
                process_file(os.path.join(root, file))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("directory", help="")

    args = parser.parse_args()
    process_directory(args.directory)
