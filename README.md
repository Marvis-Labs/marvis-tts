
# Introduction

Marvis is a cutting-edge conversational speech model designed to enable real-time voice cloning and streaming text-to-speech synthesis. Built with efficiency and accessibility in mind, Marvis addresses the growing need for high-quality, real-time voice synthesis that can run on consumer devices such as Apple Silicon and others.

Traditional voice cloning models require either the whole text input, lengthy audio samples or lack real-time streaming capabilities. Marvis bridges this gap by enabling voice cloning with just 10 seconds of audio while maintaining natural-sounding speech through intelligent text processing and streaming audio generation.

## Key Features

- **Rapid Voice Cloning**: Clone any voice using just 10 seconds of reference audio
- **Real-time Streaming**: Stream audio chunks as text is processed, enabling natural conversational flow
- **Compact Size**: Only 500MB when quantized, enabling on-device inference
- **Edge deployment**: Optimized for real-time Speech-to-Speech (STS) on mobile devices (i.e., iPad, iPhone and etc)
- **Natural Audio Flow**: Process entire text context for coherent speech synthesis without chunking artifacts
- **Multimodal Architecture**: Seamlessly handles interleaved text and audio tokens

## Supported Languages

Currently optimized for English with support for expressive speech synthesis with additional languages such as German, Portuguese, French and Mandarin coming soon.

# Quick Start

## Using MLX

```bash
pip install -U mlx-audio
python -m mlx_audio.tts.generate --model Marvis-AI/marvis-tts-250m-v0.1  --stream \
 --text "Marvis TTS is a new text-to-speech model that provides fast streaming on edge devices."
```

## Using transformers

**Without Voice Cloning**
```python
import torch
from transformers import AutoTokenizer, AutoProcessor, CsmForConditionalGeneration
from tokenizers.processors import TemplateProcessing
import soundfile as sf

model_id = "Marvis-AI/marvis-tts-250m-v0.1"
device = "cuda"if torch.cuda.is_available() else "cpu"

# load the model and the processor
processor = AutoProcessor.from_pretrained(model_id)
model = CsmForConditionalGeneration.from_pretrained(model_id, device_map=device)

# prepare the inputs
text = "[0]Marvis TTS is a new text-to-speech model that provides fast streaming on edge devices." # `[0]` for speaker id 0
inputs = processor(text, add_special_tokens=True, return_tensors="pt").to(device).pop("token_type_ids")
# infer the model
audio = model.generate(**inputs, output_audio=True)
sf.write("example_without_context.wav", audio[0].cpu(), samplerate=24_000, subtype="PCM_16")

```


# Model Description

Marvis is built on the [Sesame CSM-1B](https://huggingface.co/sesame/csm-1b) (Conversational Speech Model) architecture, a multimodal transformer that operates directly on Residual Vector Quantization (RVQ) tokens and uses [Kyutai's mimi codec](https://huggingface.co/kyutai/mimi). The architecture enables end-to-end training while maintaining low-latency generation and employs a dual-transformer approach:

- **Multimodal Backbone (250M parameters)**: Processes interleaved text and audio sequences to model the zeroth codebook level, providing semantic understanding and context.

- **Audio Decoder (60M parameters)**: A smaller, specialized transformer that models the remaining 31 codebook levels to reconstruct high-quality speech from the backbone's representations.

Unlike models that require text chunking based on regex patterns, Marvis processes entire text sequences contextually, resulting in more natural speech flow and intonation.

**Key Architectural Innovation**: Unlike models that require text chunking based on regex patterns, Marvis processes entire text sequences contextually, resulting in more natural speech flow and intonation.

# Training Details

**Pretraining**: 
- Dataset: Emilia-YODAS 
- Training Steps: 2M steps
- Hardware: 1x NVIDIA GH200 96GB
- Precision: bfloat16
- Learning Rate: 3e-4
- Batch Size: 64

**Post-training**: 
- Dataset: Expressive Speech
- Training Steps: 200K steps
- Expressiveness Setting: 0.5
- Hardware: 1x NVIDIA GH200 96GB
- Precision: bfloat16
- Learning Rate: 1e-4
- Batch Size: 64

**Total Training Cost**: ~$2,000 
- Pretraining and fine-tuning: $246.69 (1x GH200)
- Post-training data generation: $167.94 (RTX6000 Ada)
- Additional experimentation: ~$1,500 across various GPU configurations
- Platforms: Prime-Intellect and Jarvis-Labs

## Use Cases

- **Real-time Voice Assistants**: Deploy natural-sounding voice interfaces with custom voices
- **Content Creation**: Generate voiceovers and narration with personalized voices
- **Accessibility Tools**: Create personalized speech synthesis for communication aids
- **Interactive Applications**: Build conversational AI with consistent voice identity
- **Podcast & Media**: Generate natural-sounding speech for automated content

### Local & Cloud Deployment

**Local Deployment:**
- Minimum Requirements: 1GB RAM, GPU recommended for real-time inference
- Quantized Model: 500MB download
- Platforms: iOS, Android, Windows, macOS, Linux

**Cloud Deployment:**
- API-ready architecture
- Scalable inference pipeline
- Low-latency streaming support

### Technical Limitations

- Language Support: Currently optimized primarily for English. Performance on other languages may be suboptimal
- Audio Quality Dependency: Voice cloning quality is dependent on the clarity and quality of the 10-second reference audio
- Background Noise: Performance degrades with noisy reference audio or inference environments
- Hallucinations: The model might hallucinate words specially for new words or short sentences.

### Legal and Ethical Considerations:

- Users are responsible for complying with local laws regarding voice synthesis and impersonation
- Consider intellectual property rights when cloning voices of public figures
- Respect privacy laws and regulations in your jurisdiction
- Obtain appropriate consent and permissions before deployment

## License & Agreement

* Apache 2.0

## Citation

If you use Marvis in your research or applications, please cite:

```bibtex
@misc{marvis-tts-2025,
  title={Marvis-TTS: Efficient Real-time Voice Cloning with Streaming Speech Synthesis},
  author={Prince Canuma and Lucas Newman},
  year={2025}
}
```

## Acknowledgments

Special thanks to Sesame and Kyutai for their groundbreaking open-source contributions that inspired our work, and to the broader open-source community for their unwavering support and collaboration.

---

**Version**: 0.1 

**Release Date**: 26/08/2025  

**Creators**: Prince Canuma & Lucas Newman
