from __future__ import annotations

import platform
import time
from typing import Any, Protocol

import pandas as pd

from .config import BenchmarkConfig
from .io import write_candidate_output


class TranslationRunner(Protocol):
    def translate(self, texts: list[str]) -> tuple[list[str], list[float]]: ...
    def close(self) -> None: ...


def _torch_dtype(torch: Any, name: str):
    aliases = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    if name not in aliases:
        raise ValueError(f"Unsupported dtype {name!r}.")
    return aliases[name]


class TranslateGemmaRunner:
    def __init__(self, candidate: dict[str, Any], settings: dict[str, Any]):
        import torch
        from peft import PeftModel
        from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
        from train import load_generation_safe_model_config, make_deterministic_generation_config

        self.torch = torch
        self.settings = settings
        self.source_lang = candidate.get("source_lang", "en")
        self.target_lang = candidate.get("target_lang", "fa")
        model_id = candidate["model"]
        revision_kwargs = {"revision": candidate["revision"]} if candidate.get("revision") else {}
        self.processor = AutoProcessor.from_pretrained(
            candidate.get("processor", model_id), use_fast=True, fix_mistral_regex=False, **revision_kwargs
        )
        self.processor.tokenizer.padding_side = "left"
        if revision_kwargs:
            model_config = AutoConfig.from_pretrained(model_id, **revision_kwargs)
            for model_part in (model_config, model_config.get_text_config()):
                model_part.temperature = 1.0
                model_part.top_p = 1.0
                model_part.top_k = 50
        else:
            model_config = load_generation_safe_model_config(model_id)
        kwargs = {
            "config": model_config,
            "generation_config": make_deterministic_generation_config(model_config, self.processor),
            "dtype": _torch_dtype(torch, candidate.get("dtype", "bfloat16")),
        }
        kwargs.update(revision_kwargs)
        if attention := candidate.get("attn_implementation"):
            kwargs["attn_implementation"] = attention
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        device = candidate.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        if adapter := candidate.get("adapter") or candidate.get("adapter_repo"):
            adapter_kwargs = {"revision": candidate["adapter_revision"]} if candidate.get("adapter_revision") else {}
            self.model = PeftModel.from_pretrained(self.model, adapter, **adapter_kwargs)
        self.model.eval()

    def translate(self, texts: list[str]) -> tuple[list[str], list[float]]:
        conversations = [[{"role": "user", "content": [{
            "type": "text", "source_lang_code": self.source_lang,
            "target_lang_code": self.target_lang, "text": text,
        }]}] for text in texts]
        prompts = [self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        ) for conversation in conversations]
        inputs = self.processor(text=prompts, padding=True, return_tensors="pt").to(self.model.device)
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.processor.tokenizer.eos_token_id
        kwargs = {
            "max_new_tokens": int(self.settings.get("max_new_tokens", 512)),
            "do_sample": bool(self.settings.get("do_sample", False)),
            "num_beams": int(self.settings.get("num_beams", 1)),
            "pad_token_id": pad_id,
        }
        if kwargs["do_sample"]:
            kwargs.update(temperature=float(self.settings.get("temperature", 1.0)), top_p=float(self.settings.get("top_p", 1.0)))
        started = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, **kwargs)
        elapsed = time.perf_counter() - started
        input_width = inputs["input_ids"].shape[1]
        translations = self.processor.batch_decode(output[:, input_width:], skip_special_tokens=True)
        return [text.strip() for text in translations], [elapsed / len(texts)] * len(texts)

    def close(self) -> None:
        del self.model, self.processor
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


class NllbRunner:
    def __init__(self, candidate: dict[str, Any], settings: dict[str, Any]):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.torch = torch
        self.settings = settings
        model_id = candidate["model"]
        revision_kwargs = {"revision": candidate["revision"]} if candidate.get("revision") else {}
        self.tokenizer = AutoTokenizer.from_pretrained(
            candidate.get("tokenizer", model_id), src_lang=candidate["source_lang"], **revision_kwargs
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id, dtype=_torch_dtype(torch, candidate.get("dtype", "bfloat16")), **revision_kwargs
        )
        device = candidate.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        if adapter := candidate.get("adapter") or candidate.get("adapter_repo"):
            adapter_kwargs = {"revision": candidate["adapter_revision"]} if candidate.get("adapter_revision") else {}
            self.model = PeftModel.from_pretrained(self.model, adapter, **adapter_kwargs)
        self.model.eval()
        self.target_lang = candidate["target_lang"]

    def translate(self, texts: list[str]) -> tuple[list[str], list[float]]:
        tokenize_kwargs = {"padding": True, "truncation": True, "return_tensors": "pt"}
        if self.settings.get("max_source_tokens"):
            tokenize_kwargs["max_length"] = int(self.settings["max_source_tokens"])
        inputs = self.tokenizer(texts, **tokenize_kwargs).to(self.model.device)
        target_id = self.tokenizer.convert_tokens_to_ids(self.target_lang)
        if target_id == self.tokenizer.unk_token_id:
            raise ValueError(f"Unknown NLLB target language token {self.target_lang!r}.")
        kwargs = {
            "forced_bos_token_id": target_id,
            "max_new_tokens": int(self.settings.get("max_new_tokens", 512)),
            "num_beams": int(self.settings.get("num_beams", 1)),
            "do_sample": bool(self.settings.get("do_sample", False)),
        }
        if kwargs["do_sample"]:
            kwargs.update(temperature=float(self.settings.get("temperature", 1.0)), top_p=float(self.settings.get("top_p", 1.0)))
        started = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, **kwargs)
        elapsed = time.perf_counter() - started
        translations = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        return [text.strip() for text in translations], [elapsed / len(texts)] * len(texts)

    def close(self) -> None:
        del self.model, self.tokenizer
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


RUNNERS = {"translategemma": TranslateGemmaRunner, "nllb": NllbRunner}


def generate_candidate(config: BenchmarkConfig, candidate: dict[str, Any], dataset: pd.DataFrame, dataset_manifest: dict[str, Any]):
    profiles = config.raw.get("generation_profiles", {})
    settings = {**profiles.get(candidate.get("generation_profile"), {}), **candidate.get("generation", {})}
    batch_size = int(settings.get("batch_size", 1))
    if batch_size <= 0:
        raise ValueError("generation batch_size must be positive.")
    runner = RUNNERS[candidate["runner"]](candidate, settings)
    environment = {
        "python": platform.python_version(),
        "torch": runner.torch.__version__,
        "cuda": runner.torch.version.cuda,
        "device": str(next(runner.model.parameters()).device),
    }
    translations: list[str] = []
    latencies: list[float] = []
    try:
        sources = dataset["source"].tolist()
        for offset in range(0, len(sources), batch_size):
            batch_translations, batch_latencies = runner.translate(sources[offset:offset + batch_size])
            translations.extend(batch_translations)
            latencies.extend(batch_latencies)
    finally:
        runner.close()
    if len(translations) != len(dataset) or len(latencies) != len(dataset):
        raise RuntimeError(
            f"Runner returned {len(translations)} translations and {len(latencies)} latencies "
            f"for {len(dataset)} examples."
        )
    output = pd.DataFrame({
        "example_id": dataset["example_id"],
        "translation": translations,
        "status": ["ok" if value else "empty" for value in translations],
        "latency_seconds": latencies,
        "source_chars": dataset["source"].str.len(),
        "output_chars": [len(value) for value in translations],
    })
    return write_candidate_output(config, candidate, output, dataset_manifest, {
        "generation_settings": settings,
        "environment": environment,
    })
