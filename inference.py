import argparse
from pathlib import Path
import numpy as np

import torch
import sounddevice as sd
import soundfile as sf

from marvis_tts.generator import Generator
from marvis_tts.models import Model, ModelArgs
from marvis_tts.utils import Segment, load_smollm2_tokenizer


def create_model(tokenizer, device: str = "cuda", dtype: torch.dtype = torch.bfloat16) -> Model:
    model_args = ModelArgs(
        backbone_flavor="llama-250M",
        decoder_flavor="llama-60M",
        text_vocab_size=tokenizer.vocab_size,
        audio_vocab_size=2051,
        audio_num_codebooks=32,
    )
    return Model(model_args).to(device=device, dtype=dtype)


def load_checkpoint(model: Model, ckpt_path: Path):
    obj = torch.load(ckpt_path, weights_only=True, map_location="cpu")
    state_dict = obj["model_state"] if isinstance(obj, dict) and "model_state" in obj else obj
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def read_audio_as_mono_tensor(path: Path, device: str) -> torch.Tensor:
    wav, sr = sf.read(str(path))
    if wav.ndim == 2:  # (num_frames, channels) -> mono
        wav = wav.mean(axis=1)
    wav = torch.tensor(wav[None, None, :], dtype=torch.float32, device=device)  # [B=1, C=1, T]
    return wav


def main():
    parser = argparse.ArgumentParser(description="Voice-match TTS generation")
    parser.add_argument("--model_path", required=True, type=Path, help="Path to checkpoint .pt")
    parser.add_argument("--ref_audio", required=True, type=Path, help="Path to reference audio .wav")
    parser.add_argument("--ref_text", required=True, type=Path, help="Path to reference text file")
    parser.add_argument(
        "--text",
        required=True,
        action="append",
        help="Text to generate (can be passed multiple times for multiple generations)",
    )
    parser.add_argument("--play", action="store_true", help="Skip realtime playback")
    parser.add_argument("--output", type=Path, default=Path("output.wav"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_smollm2_tokenizer()

    model = create_model(tokenizer, device=device, dtype=torch.float32)
    load_checkpoint(model, args.model_path)
    generator = Generator(model, text_tokenizer=tokenizer, device=device)

    ref_audio = read_audio_as_mono_tensor(args.ref_audio, device=device)
    ref_audio_tokens = generator._audio_tokenizer.encode(ref_audio)[-1, :, :]

    with open(args.ref_text, "r") as f:
        ref_text = f.read().strip()

    all_audio = []
    for one_text in args.text:
        all_text = (ref_text + " " + one_text.strip()).strip()
        context = [
            Segment(
                text=torch.tensor(tokenizer.encode(f"[0]{all_text}"), dtype=torch.long, device=device),
                audio=torch.tensor(ref_audio_tokens, dtype=torch.long, device=device)[:32, :],
                speaker=0,
            )
        ]

        with torch.inference_mode():
            audio = generator.generate(
                text="",  # n.b. conditioned entirely via context
                speaker=0,
                context=context,
                max_audio_length_ms=args.max_ms,
                temperature=args.temperature,
                topk=args.topk,
                voice_match=True,
            )

        all_audio.append(audio.cpu().numpy())

    out_wav = np.concatenate(all_audio) if len(all_audio) > 1 else all_audio[0]
    sf.write(args.output, out_wav, samplerate=24_000, subtype="PCM_16")
    if args.play:
        sd.play(out_wav, samplerate=24_000, blocking=True)
