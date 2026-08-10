from __future__ import annotations

import platform
import time
import json
from typing import Any, Protocol

import pandas as pd
from accelerate import PartialState

from .config import BenchmarkConfig
from .io import candidate_dir, candidate_output_path, write_candidate_output


class TranslationRunner(Protocol):
    def translate(self, texts: list[str]) -> tuple[list[str], list[float], list[int]]: ...
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
        from prompting import resolve_stop_token_ids
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
            "generation_config": make_deterministic_generation_config(
                model_config, self.processor, model_id
            ),
            "dtype": _torch_dtype(torch, candidate.get("dtype", "bfloat16")),
        }
        # Passed on every generate() call as well, so an adapter's own
        # generation config cannot drop <end_of_turn> from the stop set.
        self.stop_token_ids = resolve_stop_token_ids(
            self.processor.tokenizer, base_model_id=model_id
        )
        kwargs.update(revision_kwargs)
        if attention := candidate.get("attn_implementation"):
            kwargs["attn_implementation"] = attention
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        device = candidate.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        adapter = candidate.get("adapter") or candidate.get("adapter_repo")
        if adapter:
            adapter_kwargs = {"revision": candidate["adapter_revision"]} if candidate.get("adapter_revision") else {}
            self.model = PeftModel.from_pretrained(self.model, adapter, **adapter_kwargs)
        # An SFT adapter from this repository is conditioned on the training
        # rendering; an untouched upstream checkpoint is conditioned on the
        # generation prompt. The two are not the same string.
        self.use_training_rendering = bool(adapter)
        self.model.eval()

    def translate(self, texts: list[str]) -> tuple[list[str], list[float], list[int]]:
        from prompting import render_inference_prompts, tokenize_prompts_for_generation

        user_messages = [{"role": "user", "content": [{
            "type": "text", "source_lang_code": self.source_lang,
            "target_lang_code": self.target_lang, "text": text,
        }]} for text in texts]
        prompts = render_inference_prompts(self.processor, user_messages, self.use_training_rendering)
        inputs = tokenize_prompts_for_generation(self.processor, prompts, self.model.device)
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.processor.tokenizer.eos_token_id
        kwargs = {
            "max_new_tokens": int(self.settings.get("max_new_tokens", 512)),
            "do_sample": bool(self.settings.get("do_sample", False)),
            "num_beams": int(self.settings.get("num_beams", 1)),
            "pad_token_id": pad_id,
            "eos_token_id": self.stop_token_ids,
        }
        if kwargs["do_sample"]:
            kwargs.update(temperature=float(self.settings.get("temperature", 1.0)), top_p=float(self.settings.get("top_p", 1.0)))
        started = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, **kwargs)
        elapsed = time.perf_counter() - started
        input_width = inputs["input_ids"].shape[1]
        generated_tokens = output[:, input_width:]
        translations = self.processor.batch_decode(generated_tokens, skip_special_tokens=True)
        token_counts = (generated_tokens != pad_id).sum(dim=1).tolist()
        return [text.strip() for text in translations], [elapsed / len(texts)] * len(texts), token_counts

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

    def translate(self, texts: list[str]) -> tuple[list[str], list[float], list[int]]:
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
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            token_counts = [int(output.shape[1])] * len(texts)
        else:
            # Seq2seq outputs include one decoder-start token. Exclude it from
            # the count so this stays comparable to causal generated tokens.
            token_counts = [max(0, int(value) - 1) for value in (output != pad_id).sum(dim=1).tolist()]
        return [text.strip() for text in translations], [elapsed / len(texts)] * len(texts), token_counts

    def close(self) -> None:
        del self.model, self.tokenizer
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


RUNNERS = {"translategemma": TranslateGemmaRunner, "nllb": NllbRunner}


def _generate_local_shard(
    candidate: dict[str, Any],
    settings: dict[str, Any],
    dataset: pd.DataFrame,
    device: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    runtime_candidate = {**candidate, "device": str(device)}
    runner = RUNNERS[candidate["runner"]](runtime_candidate, settings)
    environment = {
        "python": platform.python_version(),
        "torch": runner.torch.__version__,
        "cuda": runner.torch.version.cuda,
        "device": str(next(runner.model.parameters()).device),
    }
    translations: list[str] = []
    latencies: list[float] = []
    output_token_counts: list[int] = []
    batch_size = int(settings.get("batch_size", 1))
    try:
        sources = dataset["source"].tolist()
        for offset in range(0, len(sources), batch_size):
            batch_translations, batch_latencies, batch_token_counts = runner.translate(
                sources[offset:offset + batch_size]
            )
            translations.extend(batch_translations)
            latencies.extend(batch_latencies)
            output_token_counts.extend(batch_token_counts)
    finally:
        runner.close()
    lengths = {len(translations), len(latencies), len(output_token_counts), len(dataset)}
    if len(lengths) != 1:
        raise RuntimeError(
            f"Runner returned translations={len(translations)}, latencies={len(latencies)}, "
            f"token_counts={len(output_token_counts)} for {len(dataset)} examples."
        )
    max_new_tokens = int(settings.get("max_new_tokens", 512))
    output = pd.DataFrame({
        "example_id": dataset["example_id"].tolist(),
        "translation": translations,
        "status": ["ok" if value else "empty" for value in translations],
        "latency_seconds": latencies,
        "source_chars": dataset["source"].str.len().tolist(),
        "output_chars": [len(value) for value in translations],
        "output_tokens": output_token_counts,
        "hit_max_new_tokens": [count >= max_new_tokens for count in output_token_counts],
    })
    return output, environment


def generate_candidate(config: BenchmarkConfig, candidate: dict[str, Any], dataset: pd.DataFrame, dataset_manifest: dict[str, Any]):
    profiles = config.raw.get("generation_profiles", {})
    settings = {**profiles.get(candidate.get("generation_profile"), {}), **candidate.get("generation", {})}
    batch_size = int(settings.get("batch_size", 1))
    if batch_size <= 0:
        raise ValueError("generation batch_size must be positive.")
    state = PartialState()
    local_dataset = dataset.iloc[state.process_index::state.num_processes].copy()
    if local_dataset.empty:
        local_output = pd.DataFrame(columns=[
            "example_id", "translation", "status", "latency_seconds", "source_chars",
            "output_chars", "output_tokens", "hit_max_new_tokens",
        ])
        environment = {"python": platform.python_version(), "device": str(state.device)}
    else:
        device = state.device if state.num_processes > 1 else candidate.get("device", state.device)
        local_output, environment = _generate_local_shard(candidate, settings, local_dataset, device)

    shard_dir = candidate_dir(config, candidate["id"]) / "distributed_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / f"rank-{state.process_index:05d}.csv"
    local_output.to_csv(shard_path, index=False)
    (shard_dir / f"rank-{state.process_index:05d}.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    state.wait_for_everyone()

    if state.is_main_process:
        shard_frames = [
            pd.read_csv(shard_dir / f"rank-{rank:05d}.csv", dtype={"example_id": str})
            for rank in range(state.num_processes)
        ]
        combined = pd.concat(shard_frames, ignore_index=True)
        if combined["example_id"].duplicated().any():
            raise RuntimeError(f"Distributed generation produced duplicate IDs for {candidate['id']}.")
        ordered = dataset[["example_id"]].merge(combined, on="example_id", how="left", validate="one_to_one")
        if ordered["translation"].isna().any():
            missing = ordered.loc[ordered["translation"].isna(), "example_id"].tolist()
            raise RuntimeError(f"Distributed generation missed {len(missing)} IDs for {candidate['id']}.")
        environments = [
            json.loads((shard_dir / f"rank-{rank:05d}.json").read_text(encoding="utf-8"))
            for rank in range(state.num_processes)
        ]
        write_candidate_output(config, candidate, ordered, dataset_manifest, {
            "generation_settings": settings,
            "distributed": {
                "strategy": "data_parallel",
                "num_processes": state.num_processes,
                "environments": environments,
            },
        })
    state.wait_for_everyone()
    return candidate_output_path(config, candidate["id"])
