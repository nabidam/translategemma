"""Model loading and batched translation for the serving API.

The generation path is a deliberate line-for-line match with
evaluate_translations.generate_translations: same generation-safe model config,
same twice-resolved stop set, same per-system prompt rendering, same
generation kwargs, same decode. Those are the fixes from
docs/2026-08-10_adapter_degeneration_analysis.md, and every one of them fails
silently -- an adapter queried after the wrong prefix, or decoded without
<end_of_turn> in the stop set, still returns fluent Farsi. A served translation
must be the translation the harness scored.

The one intentional difference is multi-GPU: the harness shards a fixed test set
across ranks, which has no meaning for a request/response server. One process
serves one model on one device.
"""

import logging

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

from config import ModelMode, System
from model_loading import (
    load_generation_safe_model_config,
    make_deterministic_generation_config,
    resolve_dtype,
)
from prompting import (
    render_inference_prompts,
    resolve_stop_token_ids,
    tokenize_prompts_for_generation,
)

logger = logging.getLogger("translategemma.api")


class SentenceSplitter:
    """pysbd segmenters, created lazily and cached per language.

    Falls back to treating the text as one segment for languages pysbd does not
    support: a translation of the whole text is a far better outcome than a 400
    on an otherwise valid request.
    """

    def __init__(self):
        self._segmenters = {}

    def split(self, text: str, language: str) -> list[str]:
        segmenter = self._segmenter(language)
        if segmenter is None:
            return [text]
        segments = [segment.strip() for segment in segmenter.segment(text)]
        return [segment for segment in segments if segment] or [text]

    def _segmenter(self, language: str):
        if language not in self._segmenters:
            import pysbd

            try:
                self._segmenters[language] = pysbd.Segmenter(language=language, clean=False)
            except ValueError:
                logger.warning(
                    "pysbd has no model for language %r; translating the text unsplit.", language
                )
                self._segmenters[language] = None
        return self._segmenters[language]


class TranslationEngine:
    """Holds the loaded model(s) and turns texts into translations.

    Not thread-safe: `translate` runs one `generate()` at a time and the caller
    serializes access (main.py holds a lock). A single GPU worker is the
    intended deployment; scale by running more containers.
    """

    def __init__(self, settings):
        self.settings = settings
        self.device = torch.device(
            settings.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        self.splitter = SentenceSplitter()
        self.model = None
        self.processor = None
        self.stop_token_ids = []

    # ------------------------------------------------------------------ load

    def load(self):
        settings = self.settings
        logger.info(
            "Loading %s (mode=%s, adapter=%s) on %s",
            settings.base_model_id,
            settings.model_mode,
            settings.adapter_path or "none",
            self.device,
        )

        processor = AutoProcessor.from_pretrained(
            settings.base_model_id, use_fast=True, fix_mistral_regex=False
        )
        # Required for batched causal generation; tokenize_prompts_for_generation
        # refuses to run otherwise.
        processor.tokenizer.padding_side = "left"
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

        model_config = load_generation_safe_model_config(settings.base_model_id)
        load_kwargs = {
            "config": model_config,
            "generation_config": make_deterministic_generation_config(
                model_config, processor, settings.base_model_id
            ),
            "dtype": resolve_dtype(settings.dtype),
            "attn_implementation": settings.attn_implementation,
        }
        if settings.load_in_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=settings.bnb_4bit_use_double_quant,
                bnb_4bit_quant_type=settings.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=resolve_dtype(settings.dtype),
            )

        # Resolved a second time and passed on every generate() call: a cached
        # or adapter-supplied generation config must not be able to reintroduce
        # a stop set that omits <end_of_turn>.
        self.stop_token_ids = resolve_stop_token_ids(
            processor.tokenizer, base_model_id=settings.base_model_id
        )
        logger.info(
            "Stop tokens for generation: %s -> %s",
            processor.tokenizer.convert_ids_to_tokens(self.stop_token_ids),
            self.stop_token_ids,
        )

        model = AutoModelForCausalLM.from_pretrained(settings.base_model_id, **load_kwargs)
        if not settings.load_in_4bit:
            # bitsandbytes places quantized weights during from_pretrained;
            # moving them afterwards is unnecessary and unsupported.
            model = model.to(self.device)
        if settings.model_mode is not ModelMode.BASE:
            # One set of base weights either way. In "both" mode the adapter is
            # switched off per request through disable_adapter() rather than by
            # loading a second 12B model.
            model = PeftModel.from_pretrained(model, settings.adapter_path)
        model.eval()

        self.processor = processor
        self.model = model
        logger.info("Model ready. Systems: %s", ", ".join(self.settings.loaded_systems))

    def unload(self):
        self.model = None
        self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    # ------------------------------------------------------------- translate

    def translate(
        self,
        texts: list[str],
        system: System,
        source_lang: str,
        target_lang: str,
        max_new_tokens: int,
        split_sentences: bool,
    ) -> list[str]:
        """Translate texts, preserving order. One output per input.

        With split_sentences, each text is segmented, every segment of every
        text is batched together, and the segments are rejoined per text. That
        keeps batches full even when one request carries a single long document.
        """
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded.")
        if system not in self.settings.loaded_systems:
            raise ValueError(f"System {system!r} is not loaded.")

        if split_sentences:
            segments_per_text = [self.splitter.split(text, source_lang) for text in texts]
        else:
            segments_per_text = [[text] for text in texts]

        flat_segments = [segment for segments in segments_per_text for segment in segments]
        flat_translations = self._generate(
            flat_segments, system, source_lang, target_lang, max_new_tokens
        )

        translations = []
        cursor = 0
        for segments in segments_per_text:
            chunk = flat_translations[cursor : cursor + len(segments)]
            cursor += len(segments)
            translations.append(" ".join(part for part in chunk if part))
        return translations

    def _generate(
        self,
        segments: list[str],
        system: System,
        source_lang: str,
        target_lang: str,
        max_new_tokens: int,
    ) -> list[str]:
        settings = self.settings
        tokenizer = self.processor.tokenizer
        use_training_rendering = settings.use_training_rendering(system)
        outputs: list[str] = []

        with self._active_system(system):
            for start in range(0, len(segments), settings.batch_size):
                batch = segments[start : start + settings.batch_size]
                user_messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "source_lang_code": source_lang,
                                "target_lang_code": target_lang,
                                "text": segment,
                            }
                        ],
                    }
                    for segment in batch
                ]

                # The adapter is conditioned on the SFT rendering, which
                # add_generation_prompt=True does not reproduce; the untouched
                # base model is conditioned on the generation prompt. Each
                # system is queried the way it was trained.
                prompts = render_inference_prompts(
                    self.processor, user_messages, use_training_rendering
                )
                inputs = tokenize_prompts_for_generation(
                    self.processor, prompts, self.model.device
                )

                with torch.inference_mode():
                    generation_kwargs = {
                        "max_new_tokens": max_new_tokens,
                        "do_sample": settings.do_sample,
                        "num_beams": settings.num_beams,
                        "pad_token_id": tokenizer.pad_token_id,
                        "eos_token_id": self.stop_token_ids,
                    }
                    if settings.do_sample:
                        generation_kwargs.update(
                            temperature=settings.temperature, top_p=settings.top_p
                        )
                    else:
                        # Explicit neutral values: TranslateGemma's config.json
                        # ships sampling parameters that generate() warns about
                        # under greedy decoding.
                        generation_kwargs.update(temperature=1.0, top_p=1.0, top_k=50)
                    generated = self.model.generate(**inputs, **generation_kwargs)

                input_length = inputs["input_ids"].shape[-1]
                generated_tokens = generated[:, input_length:]
                # Decoded exactly as the harness decodes it, without a strip().
                # Trailing whitespace is the visible signature of an unstopped
                # decoder (70% of rows in the 2026-08-10 run); trimming it here
                # would hide a regression from whoever is reading the output.
                outputs.extend(
                    self.processor.batch_decode(generated_tokens, skip_special_tokens=True)
                )

        return outputs

    def _active_system(self, system: System):
        """Context manager selecting which system answers this batch.

        In "both" mode the base system is served by switching the LoRA layers
        off in place, so both systems share one copy of the 12B weights.
        """
        if system is System.BASE and self.settings.model_mode is ModelMode.BOTH:
            return self.model.disable_adapter()
        return _null_context()


class _null_context:
    """No-op context manager; the loaded model already is the requested system."""

    def __enter__(self):
        return None

    def __exit__(self, *exc_info):
        return False
