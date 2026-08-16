"""Prompt rendering and stop-token resolution shared by training and inference.

Two properties of the TranslateGemma chat template caused the 2026-08-10 adapter
to translate correctly and then never stop (see
docs/2026-08-10_adapter_degeneration_analysis.md). Both are easy to reintroduce
independently in each generation entry point, so both are resolved here once.

1. ``add_generation_prompt=True`` is NOT the prefix the assistant turn is
   trained to continue. The training rendering opens the assistant turn with a
   blank line and eight spaces of Jinja block indentation::

       training  : ... <start_of_turn> model "\\n\\n" "<eight spaces>"
       generation: ... <start_of_turn> model "\\n"

   Every row of the 2026-08-10 evaluation was therefore generated from a prefix
   the adapter had never seen. ``render_training_prompt`` reproduces the
   training rendering exactly, by rendering a completed assistant turn whose
   content is a unique marker and cutting at the marker.

2. The chat turn ender ``<end_of_turn>`` is not always in a model config's
   ``eos_token_id``. ``GenerationConfig.from_model_config()`` reads config.json
   only, and for google/translategemma-12b-it that yields ``[1]`` (``<eos>``)
   while the published generation_config.json lists ``[1, 106]``. Since SFT
   trains every target to end in ``<end_of_turn>``, a decoder that does not stop
   on it decodes straight through the turn boundary. ``resolve_stop_token_ids``
   unions every known source so no single missing file can reintroduce this.
"""

TARGET_BOUNDARY_MARKER = "<|translategemma-target-boundary|>"
CHAT_TURN_END_TOKEN = "<end_of_turn>"


def as_token_id_set(value):
    """Normalize an eos_token_id, which may be None, an int, or a sequence."""
    if value is None:
        return set()
    if isinstance(value, int):
        return {value}
    return {int(item) for item in value if item is not None}


def resolve_stop_token_ids(tokenizer, *generation_configs, base_model_id=None, base_revision=None):
    """Return every id that should end generation, as a sorted list.

    Unions the tokenizer's eos, each supplied generation config's eos, the
    published generation_config.json when ``base_model_id`` is given, and the
    chat turn ender. Union rather than "first non-empty" is deliberate: a stop
    set can only be too small, never too large, because a token that the model
    never emits costs nothing.

    ``base_revision`` must be threaded through wherever the caller pinned the
    base model to a revision. Reading generation_config.json from the default
    revision while the weights came from another one silently mixes two
    releases' stop contracts, which is the failure this module exists to stop.
    """
    stop_ids = as_token_id_set(getattr(tokenizer, "eos_token_id", None))
    for generation_config in generation_configs:
        if generation_config is not None:
            stop_ids |= as_token_id_set(getattr(generation_config, "eos_token_id", None))
    if base_model_id:
        # Imported lazily so this module stays importable without transformers,
        # which keeps the data-only audit scripts cheap.
        from transformers import GenerationConfig

        try:
            published = GenerationConfig.from_pretrained(base_model_id, revision=base_revision)
        except (OSError, ValueError):
            # No generation_config.json published, or the hub is unreachable in
            # an offline deployment. The explicit turn ender below still applies.
            published = None
        if published is not None:
            stop_ids |= as_token_id_set(published.eos_token_id)
    turn_end = tokenizer.convert_tokens_to_ids(CHAT_TURN_END_TOKEN)
    if isinstance(turn_end, int) and turn_end >= 0 and turn_end != tokenizer.unk_token_id:
        stop_ids.add(turn_end)
    if not stop_ids:
        raise ValueError(
            "Could not resolve any stop token id. Generation would run to "
            "max_new_tokens on every example."
        )
    return sorted(stop_ids)


def render_training_prompt(processor, user_message):
    """Render the exact prefix that SFT teaches the assistant turn to continue.

    Do not replace this with ``apply_chat_template(..., add_generation_prompt=True)``:
    that omits the assistant-turn indentation the training rendering emits, and
    the difference is silent — generation still produces text, just from a
    prefix the adapter was never conditioned on.
    """
    marker_messages = [user_message, {"role": "assistant", "content": TARGET_BOUNDARY_MARKER}]
    marker_text = processor.apply_chat_template(
        marker_messages, tokenize=False, add_generation_prompt=False
    )
    try:
        boundary = marker_text.rindex(TARGET_BOUNDARY_MARKER)
    except ValueError as error:
        raise ValueError(
            "TranslateGemma chat template did not preserve the assistant boundary marker."
        ) from error
    return marker_text[:boundary]


def render_training_prompts(processor, user_messages):
    """Batch form of render_training_prompt."""
    return [render_training_prompt(processor, message) for message in user_messages]


def render_inference_prompts(processor, user_messages, use_training_rendering):
    """Render prompts the way the model under test was actually conditioned.

    ``use_training_rendering`` should be true for a model carrying one of this
    repository's SFT adapters and false for an untouched upstream checkpoint.
    The two renderings differ (see the module docstring), so applying either one
    to both systems would query one of them off-distribution. Comparing a
    baseline and an adapter each at their own native prompt is the honest
    comparison; using one prompt for both is not.
    """
    if use_training_rendering:
        return render_training_prompts(processor, user_messages)
    return [
        processor.apply_chat_template([message], tokenize=False, add_generation_prompt=True)
        for message in user_messages
    ]


def tokenize_prompts_for_generation(processor, prompts, device=None):
    """Left-padded, batch-ready inputs for prompts rendered by this module.

    ``add_special_tokens=False`` matches train.py's tokenization of the full
    rendering: the chat template already emits whatever leading special tokens
    the model expects, and adding them twice shifts every position by one.
    """
    tokenizer = processor.tokenizer
    if tokenizer.padding_side != "left":
        raise ValueError(
            "Batched causal generation requires tokenizer.padding_side='left'; "
            f"got {tokenizer.padding_side!r}."
        )
    inputs = tokenizer(
        prompts,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )
    return inputs.to(device) if device is not None else inputs
