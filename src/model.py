"""MedGemma loading + text-only inference via the transformers pipeline API.

Verified against Google's official quick_start_with_hugging_face.ipynb and the
google/medgemma-4b-it model card:
  - task ``"image-text-to-text"`` (Gemma-3 multimodal head; text-only here),
  - text-only chat messages (system + user, content = [{"type":"text",...}]),
  - greedy decoding (do_sample=False),
  - reply read from ``output[0]["generated_text"][-1]["content"]``.

torch/transformers are imported lazily inside the functions so the CPU-only unit
tests (parsing, alignment, scoring) never require a GPU or a model download.
"""

from .config import GEN_CONFIG, LOAD_IN_4BIT, MODEL_ID
from .prompt import build_messages


def load_medgemma(model_id=MODEL_ID, load_in_4bit=LOAD_IN_4BIT):
    """Build a text-generation pipeline for MedGemma.

    4-bit quantization (via BitsAndBytesConfig) keeps the 4B model within a free
    Colab T4's VRAM (~5-7GB used). The model is gated — call
    ``huggingface_hub.login()`` before this.
    """
    import torch
    from transformers import BitsAndBytesConfig, pipeline

    model_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    if load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    return pipeline(
        "image-text-to-text",
        model=model_id,
        model_kwargs=model_kwargs,
    )


def run_medgemma(pipe, sentence, gen_config=None):
    """Prompt MedGemma with one sentence and return the raw reply string."""
    gen = dict(GEN_CONFIG)
    if gen_config:
        gen.update(gen_config)
    output = pipe(text=build_messages(sentence), **gen)
    return output[0]["generated_text"][-1]["content"]
