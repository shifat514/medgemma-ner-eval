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


def run_messages(pipe, messages, gen_config=None, default_gen=None):
    """Run pre-built chat `messages` and return the raw reply string.

    Split out of ``run_medgemma`` so the MIMIC medication evaluation can supply
    its own messages and its own generation config while sharing this call path.

    GENERATION ARGUMENTS GO IN ``generate_kwargs``, NOT AS BARE KWARGS, AND THE
    DIFFERENCE IS SILENT. ``ImageTextToTextPipeline._sanitize_parameters`` names
    exactly four things it will route to ``model.generate``: ``max_new_tokens``,
    ``generate_kwargs``, and two return-shape flags. Everything else in
    ``**kwargs`` is swept into ``preprocess_params`` and handed to the
    *processor*, which does not know what to do with it and drops it.

    So ``pipe(text=messages, max_new_tokens=1024, do_sample=False,
    repetition_penalty=1.15)`` applied the token cap and discarded the other
    two. The pipeline says so, but only in a log line among several:

        Keyword argument `do_sample` is not a valid argument for this processor
        and will be ignored.

    This was found on 2026-08-27 when a repetition-penalty A/B on the billing
    evaluation returned two BYTE-IDENTICAL runs — same counts, same false
    positives, same order, including one code out of sequence in both. The
    penalty had never reached the sampler.

    ``do_sample=False`` was being dropped the same way and for the whole life of
    this module. In practice the two independent runs above were identical, so
    decoding was deterministic regardless — but that was luck, not the flag
    working, and nothing measured before today should be read as having had a
    generation config it chose.
    """
    gen = dict(default_gen if default_gen is not None else GEN_CONFIG)
    if gen_config:
        gen.update(gen_config)
    # Everything in one dict: passing max_new_tokens both here and as a direct
    # parameter is an explicit ValueError in the pipeline.
    output = pipe(text=messages, generate_kwargs=gen)
    return output[0]["generated_text"][-1]["content"]


def run_medgemma(pipe, sentence, gen_config=None):
    """Prompt MedGemma with one sentence and return the raw reply string."""
    return run_messages(pipe, build_messages(sentence), gen_config=gen_config)


def count_tokens(pipe, text):
    """Token count of `text` under the pipeline's tokenizer, or None.

    Used to detect a reply that ran into max_new_tokens (i.e. was truncated
    mid-JSON). Returns None if no tokenizer is reachable, so callers must treat
    "unknown" as "not truncated" rather than crashing.
    """
    tok = getattr(pipe, "tokenizer", None)
    if tok is None:
        processor = getattr(pipe, "processor", None)
        tok = getattr(processor, "tokenizer", None)
    if tok is None:
        return None
    try:
        return len(tok.encode(text, add_special_tokens=False))
    except Exception:  # noqa: BLE001 - diagnostics must never break a run
        return None
