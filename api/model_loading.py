"""Model/generation configuration shared by training, evaluation, and serving.

These helpers used to live in train.py. They were moved here so a process that
only needs to *generate* — the FastAPI service in api/ — can import them without
dragging in trl, datasets and the Liger kernels that train.py imports at module
scope.

They are deliberately not in prompting.py: that module stays importable without
torch or transformers so the data-only audit scripts remain cheap, while
everything here needs both.
"""

import torch
from transformers import AutoConfig, GenerationConfig

from prompting import resolve_stop_token_ids


def resolve_dtype(name):
    """Turn a config dtype name such as "bfloat16" into a torch.dtype."""
    dtype = getattr(torch, str(name), None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"model.dtype must name a torch dtype (for example bfloat16); got {name!r}")
    return dtype


def load_generation_safe_model_config(base_model_id, revision=None):
    """Load model config without TranslateGemma's invalid sampling defaults."""
    config = AutoConfig.from_pretrained(base_model_id, revision=revision)
    # Apply this to both multimodal wrappers and their decoder config. Model
    # construction creates GenerationConfig objects for nested models too.
    for candidate in (config, config.get_text_config()):
        candidate.temperature = 1.0
        candidate.top_p = 1.0
        candidate.top_k = 50
    return config


def make_deterministic_generation_config(model_config, processor, base_model_id=None, base_revision=None):
    """Return explicit, warning-free defaults for translation generation.

    The stop set is resolved through prompting.resolve_stop_token_ids rather
    than taken from the model config. GenerationConfig.from_model_config reads
    config.json only, which for google/translategemma-12b-it yields just
    <eos> (1) and omits the chat turn ender <end_of_turn> (106) that
    generation_config.json publishes and that SFT trains every target to emit.
    A decoder missing 106 does not stop a fine-tuned model at all; see
    docs/2026-08-10_adapter_degeneration_analysis.md.
    """
    tokenizer = processor.tokenizer
    base_model_id = base_model_id or getattr(model_config, "_name_or_path", None)
    generation_config = GenerationConfig.from_model_config(model_config)
    generation_config.do_sample = False
    generation_config.temperature = 1.0
    generation_config.top_p = 1.0
    generation_config.top_k = 50
    if generation_config.bos_token_id is None:
        generation_config.bos_token_id = tokenizer.bos_token_id
    # Unconditional: eos_token_id is present but incomplete, so an
    # "if ... is None" guard would never fire.
    generation_config.eos_token_id = resolve_stop_token_ids(
        tokenizer, generation_config, base_model_id=base_model_id, base_revision=base_revision
    )
    if generation_config.pad_token_id is None:
        generation_config.pad_token_id = tokenizer.pad_token_id
    if generation_config.pad_token_id is None:
        generation_config.pad_token_id = tokenizer.eos_token_id
    # from_pretrained reconstructs a supplied GenerationConfig via from_dict,
    # which does not preserve _original_object_hash. Leaving this marked as
    # model-derived makes generate() enter a legacy hash check and crash.
    generation_config._from_model_config = False
    return generation_config
