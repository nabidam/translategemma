"""Evaluate a base TranslateGemma model or a LoRA adapter on the configured test split."""

import argparse
import json
import math
import os
import sys
from contextlib import contextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from accelerate import PartialState
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from degeneration import audit_outputs
from language_pairs import resolve_language_pair
from translation_benchmark.metrics import METRIC_DIRECTIONS, score_transparent_metrics
from logging_utils import console, logger, setup_logging, log_config_summary, load_config
from prompting import (
    render_inference_prompts,
    resolve_stop_token_ids,
    tokenize_prompts_for_generation,
)
from train import (
    load_generation_safe_model_config,
    make_deterministic_generation_config,
    resolve_dtype,
)


def make_progress(style, enabled):
    """Build the shared evaluation progress bar.

    bar_width is left flexible on purpose: a fixed-width bar plus the counter
    and timing columns overflows the 80-column console Rich assumes when stdout
    is not a terminal, and Rich then silently drops the trailing (timing)
    columns. Elapsed and remaining are both shown so a long shard still reports
    progress before the ETA estimate stabilises.
    """
    return Progress(
        SpinnerColumn(),
        TextColumn(f"[bold {style}]{{task.description}}[/bold {style}]"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TextColumn("[dim]elapsed[/dim]"),
        TimeElapsedColumn(),
        TextColumn("[dim]eta[/dim]"),
        TimeRemainingColumn(compact=True, elapsed_when_finished=True),
        console=console,
        disable=not enabled,
        # Generation steps are seconds apart, so the default 10 Hz redraw only
        # costs terminal bandwidth on a bar that cannot change that fast.
        refresh_per_second=2,
        transient=False,
    )


def resolve_adapter_path(adapter_path):
    """Return an absolute local adapter directory, failing early if it is unusable.

    PeftModel.from_pretrained treats any path without adapter_config.json as a
    Hub repo id, so a wrong local directory surfaces as an opaque
    HFValidationError on every rank instead of a path problem.
    """
    path = Path(adapter_path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(
            f"Adapter path does not exist: {path} (resolved from {adapter_path!r}, cwd={Path.cwd()})"
        )
    if not (path / "adapter_config.json").is_file():
        available = sorted(
            child.parent.relative_to(path).as_posix()
            for child in path.rglob("adapter_config.json")
        )
        hint = f" Adapters found below it: {available[:10]}" if available else ""
        raise FileNotFoundError(f"No adapter_config.json in {path}.{hint}")
    return str(path)


def _gather_cached_hypotheses(output_dir, prefix, total_rows):
    """Gather cached hypotheses from single or multi-rank cache files."""
    combined = {}
    cache_files = list(output_dir.glob(f".cache_{prefix}_hypotheses*.jsonl")) + list(output_dir.glob(f".cache_{prefix}_rank*.jsonl"))
    for cache_path in cache_files:
        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            entry = json.loads(line)
                            combined[entry["index"]] = entry["hypothesis"]
                        except Exception:
                            continue
    return combined


def generate_translations(test_df, config, adapter_path=None, prefix="", force=False):
    model_cfg, eval_cfg, data_cfg = config["model"], config["evaluation"], config["data"]
    output_dir = Path(eval_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    state = PartialState()
    num_processes = state.num_processes
    process_index = state.process_index
    is_main = state.is_main_process

    batch_size = eval_cfg.get("eval_batch_size", 8)
    total_rows = len(test_df)

    rank_cache_path = output_dir / (f".cache_{prefix}_rank{process_index}.jsonl" if num_processes > 1 else f".cache_{prefix}_hypotheses.jsonl")

    if force and is_main:
        for p in output_dir.glob(f".cache_{prefix}_*"):
            p.unlink()

    state.wait_for_everyone()

    cached_hypotheses = _gather_cached_hypotheses(output_dir, prefix, total_rows)

    if len(cached_hypotheses) == total_rows:
        if is_main:
            logger.info(
                "Found all %d cached translations for [bold green]%s[/bold green]. Skipping model loading and generation.",
                total_rows,
                prefix,
            )
        return [cached_hypotheses[i] for i in range(total_rows)]

    # Round-robin assignment across processes for optimal workload distribution
    rank_indices = [i for i in range(process_index, total_rows, num_processes)]
    uncached_rank_indices = [i for i in rank_indices if i not in cached_hypotheses]

    if is_main:
        logger.info(
            "Generating translations for %s using %d GPU process(es) with eval_batch_size=%d (total: %d, remaining: %d)...",
            prefix,
            num_processes,
            batch_size,
            total_rows,
            total_rows - len(cached_hypotheses),
        )

    model_name = adapter_path or model_cfg["base_model_id"]
    if is_main:
        logger.info("Loading model for translation generation: [bold cyan]%s[/bold cyan]", model_name)

    processor = AutoProcessor.from_pretrained(
        model_cfg["base_model_id"], use_fast=True, fix_mistral_regex=False
    )
    # Require left-padding for batched Causal LM generation
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    model_config = load_generation_safe_model_config(model_cfg["base_model_id"])
    load_kwargs = {
        "config": model_config,
        "generation_config": make_deterministic_generation_config(
            model_config, processor, model_cfg["base_model_id"]
        ),
        "dtype": resolve_dtype(model_cfg["dtype"]),
        "attn_implementation": model_cfg["attn_implementation"],
    }
    # Resolved again here and passed on every generate() call: a cached or
    # adapter-supplied generation config must not be able to reintroduce a stop
    # set that omits <end_of_turn>.
    stop_token_ids = resolve_stop_token_ids(
        processor.tokenizer, base_model_id=model_cfg["base_model_id"]
    )
    if is_main:
        logger.info(
            "Stop tokens for generation: %s -> %s",
            processor.tokenizer.convert_ids_to_tokens(stop_token_ids),
            stop_token_ids,
        )

    base_model = AutoModelForCausalLM.from_pretrained(model_cfg["base_model_id"], **load_kwargs)
    base_model = base_model.to(state.device)
    model = PeftModel.from_pretrained(base_model, adapter_path) if adapter_path else base_model
    model.eval()

    progress = make_progress("cyan", is_main)

    with open(rank_cache_path, "a", encoding="utf-8") as f_cache, progress:
        task = progress.add_task(f"Generating ({prefix})", total=total_rows, completed=len(cached_hypotheses))
        
        for batch_start in range(0, len(uncached_rank_indices), batch_size):
            batch_indices = uncached_rank_indices[batch_start : batch_start + batch_size]
            user_messages = []
            for i in batch_indices:
                row = test_df.iloc[i]
                source = row[data_cfg["source_column"]]
                source_lang, target_lang = resolve_language_pair(row, data_cfg)
                user_messages.append({"role": "user", "content": [{"type": "text", "source_lang_code": source_lang, "target_lang_code": target_lang, "text": source}]})

            # The adapter is conditioned on the SFT rendering, which
            # add_generation_prompt=True does not reproduce; the untouched base
            # model is conditioned on the generation prompt. Each system is
            # queried the way it was trained.
            prompts = render_inference_prompts(processor, user_messages, adapter_path is not None)
            inputs = tokenize_prompts_for_generation(processor, prompts, model.device)

            with torch.inference_mode():
                pad_token_id = processor.tokenizer.pad_token_id
                generation_kwargs = {
                    "max_new_tokens": eval_cfg["max_new_tokens"],
                    "do_sample": eval_cfg["do_sample"],
                    "num_beams": eval_cfg["num_beams"],
                    "pad_token_id": pad_token_id,
                    "eos_token_id": stop_token_ids,
                }
                if eval_cfg["do_sample"]:
                    generation_kwargs.update(temperature=eval_cfg["temperature"], top_p=eval_cfg["top_p"])
                else:
                    generation_kwargs.update(temperature=1.0, top_p=1.0, top_k=50)
                outputs = model.generate(**inputs, **generation_kwargs)

            input_length = inputs["input_ids"].shape[-1]
            generated_tokens = outputs[:, input_length:]
            batch_hypotheses = processor.batch_decode(generated_tokens, skip_special_tokens=True)

            for i, hyp in zip(batch_indices, batch_hypotheses):
                f_cache.write(json.dumps({"index": i, "hypothesis": hyp}, ensure_ascii=False) + "\n")
            f_cache.flush()

            if is_main:
                progress.update(task, advance=len(batch_indices) * num_processes)

    del model, base_model, processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    state.wait_for_everyone()

    all_cached = _gather_cached_hypotheses(output_dir, prefix, total_rows)
    # An absent row means a rank failed, not that the model produced nothing.
    # Filling it with "" would send it into the degeneration audit as a genuine
    # empty translation and quietly move the failure rate.
    if missing := [i for i in range(total_rows) if i not in all_cached]:
        raise RuntimeError(
            f"Generation produced {len(missing)} of {total_rows} rows short for '{prefix}' "
            f"(first missing indices: {missing[:10]}). A rank likely failed; inspect "
            f"{output_dir}/.cache_{prefix}_*.jsonl and re-run, or pass --force to regenerate."
        )
    if is_main:
        logger.info("Completed translation generation for [bold green]%s[/bold green].", prefix)
    return [all_cached[i] for i in range(total_rows)]


# The auditable, model-free metrics shared with translation_benchmark. Listing
# them here (instead of deriving from METRIC_DIRECTIONS) keeps the config's
# evaluation.transparent_metrics.metrics list validatable without pulling in
# the model-based entries, which this script computes through their own stages.
TRANSPARENT_METRICS = (
    "sentence_bleu",
    "sentence_chrf",
    "number_preservation",
    "acronym_preservation",
    "formula_preservation",
    "empty_output",
    "source_copy",
)


def evaluate_transparent_metrics(sources, hypotheses, references, config, prefix=""):
    """Score sacrebleu and preservation metrics with translation_benchmark's implementation.

    These are CPU-only and cheap relative to generation, so they are neither
    cached nor sharded across ranks: the caller runs them on the main process.
    Sharing the benchmark's implementation keeps a number reported here
    comparable with the same number in a cross-model benchmark run.
    """
    settings = config["evaluation"].get("transparent_metrics") or {}
    requested = list(settings.get("metrics") or TRANSPARENT_METRICS)
    if unknown := sorted(set(requested) - set(TRANSPARENT_METRICS)):
        raise ValueError(
            f"Unknown evaluation.transparent_metrics.metrics entries: {unknown}. "
            f"Available: {sorted(TRANSPARENT_METRICS)}"
        )
    logger.info("Scoring transparent metrics for [bold green]%s[/bold green]: %s", prefix, requested)
    frame = pd.DataFrame({"source": sources, "translation": hypotheses, "reference": references})
    scored = score_transparent_metrics(frame, settings)
    columns = {name: scored[name] for name in requested}
    for name, values in columns.items():
        # Preservation metrics are NaN for rows whose source contains no
        # number/acronym/formula, so the mean is over the scored subset only.
        # scored_rows makes that denominator visible instead of implied.
        logger.info(
            "  %s (%s is better): [bold yellow]%.4f[/bold yellow] over %d scored rows",
            name, METRIC_DIRECTIONS[name], float(values.mean()), int(values.notna().sum()),
        )
    return columns


def _gather_cached_scores(output_dir, prefix, stage):
    """Read every rank's cache for one scoring stage into {index: score}."""
    combined = {}
    for cache_path in output_dir.glob(f".cache_{prefix}_{stage}*.jsonl"):
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        entry = json.loads(line)
                        combined[entry["index"]] = entry["score"]
                    except Exception:
                        continue
    return combined


def _score_sharded(stage, total, config, prefix, force, score_fn, style):
    """Run a cache-backed, rank-sharded scoring stage.

    Every rank scores its own round-robin slice into its own cache file, so a
    stage costs wall time proportional to total/num_processes instead of pinning
    one rank while the others idle at a barrier. This matters most for COMET:
    an XCOMET-sized model scoring a full test set from rank zero alone can
    outlast the process group's timeout and have the watchdog abort a run whose
    generation had already finished.

    score_fn(indices) must be a generator yielding (index, score). It is not
    called at all when this rank has nothing left to score, so the scoring model
    is never loaded for an empty slice; it must therefore contain no collective
    operations.
    """
    eval_cfg = config["evaluation"]
    output_dir = Path(eval_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    state = PartialState()
    is_main = state.is_main_process
    rank_cache_path = output_dir / (
        f".cache_{prefix}_{stage}_rank{state.process_index}.jsonl"
        if state.num_processes > 1
        else f".cache_{prefix}_{stage}.jsonl"
    )

    if force and is_main:
        for path in output_dir.glob(f".cache_{prefix}_{stage}*"):
            path.unlink()
    state.wait_for_everyone()

    cached = _gather_cached_scores(output_dir, prefix, stage)
    if len(cached) == total:
        if is_main:
            logger.info(
                "Found all %d cached %s scores for [bold green]%s[/bold green]. Skipping model loading.",
                total, stage, prefix,
            )
        return [cached[i] for i in range(total)]

    pending = [i for i in range(state.process_index, total, state.num_processes) if i not in cached]
    if is_main:
        logger.info(
            "Scoring %s for [bold green]%s[/bold green] on %d process(es) (total: %d, remaining: %d)...",
            stage, prefix, state.num_processes, total, total - len(cached),
        )

    progress = make_progress(style, is_main)
    with open(rank_cache_path, "a", encoding="utf-8") as f_cache, progress:
        task = progress.add_task(f"{stage} scoring ({prefix})", total=total, completed=len(cached))
        if pending:
            for index, score in score_fn(pending):
                f_cache.write(json.dumps({"index": index, "score": float(score)}, ensure_ascii=False) + "\n")
                f_cache.flush()
                if is_main:
                    progress.update(task, advance=state.num_processes)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    state.wait_for_everyone()

    gathered = _gather_cached_scores(output_dir, prefix, stage)
    # Never default a missing row to a neutral value. A rank that died leaves
    # gaps, and for a lower-is-better metric like MetricX a 0.0 fill reads as a
    # flawless translation, so the corpus mean would improve as a run degraded.
    if missing := [i for i in range(total) if i not in gathered]:
        raise RuntimeError(
            f"{stage} scoring produced {len(missing)} of {total} rows short for '{prefix}' "
            f"(first missing indices: {missing[:10]}). A rank likely failed; inspect "
            f"{output_dir}/.cache_{prefix}_{stage}*.jsonl and re-run, or pass --force to rescore."
        )
    return [gathered[i] for i in range(total)]


def evaluate_metricx(sources, hypotheses, references, config, prefix="", force=False):
    """Score with a reference-based MetricX-24 hybrid model (lower is better)."""
    try:
        from metricx24.models import MT5ForRegression
    except ImportError as error:
        raise ImportError("Install MetricX or set evaluation.metricx_enabled: false.") from error

    eval_cfg = config["evaluation"]
    state = PartialState()

    def score_fn(indices):
        if state.is_main_process:
            logger.info("Loading MetricX-24 model ([cyan]%s[/cyan])...", eval_cfg["metricx_model_id"])
        tokenizer = AutoTokenizer.from_pretrained(eval_cfg["metricx_tokenizer_id"])
        model = MT5ForRegression.from_pretrained(
            eval_cfg["metricx_model_id"], torch_dtype="auto"
        ).to(state.device).eval()
        try:
            with torch.inference_mode():
                for i in indices:
                    text = f"source: {sources[i]} candidate: {hypotheses[i]} reference: {references[i]}"
                    inputs = tokenizer(
                        text, return_tensors="pt", truncation=True,
                        max_length=eval_cfg["metricx_max_length"], padding=False,
                    )
                    # MetricX-24 is trained without the trailing EOS token.
                    inputs = {key: value[:, :-1].to(state.device) for key, value in inputs.items()}
                    # use_cache=False is required, not an optimisation. MetricX
                    # builds its decoder with is_encoder_decoder=False, so with
                    # caching on, MT5Stack allocates a plain DynamicCache rather
                    # than an EncoderDecoderCache; T5Attention then appends the
                    # cross-attention keys to the same cache as the decoder's
                    # self-attention key, making key_length one longer than the
                    # encoder mask and crashing on position_bias + causal_mask.
                    # The decoder runs a single dummy step here, so there is
                    # nothing for a cache to accelerate.
                    yield i, model(**inputs, use_cache=False).predictions.item()
        finally:
            del model, tokenizer

    scores = _score_sharded("metricx", len(sources), config, prefix, force, score_fn, "magenta")
    if state.is_main_process:
        logger.info(
            "MetricX finished for [bold green]%s[/bold green]. Mean (lower is better): [bold yellow]%.4f[/bold yellow]",
            prefix, float(pd.Series(scores).mean()),
        )
    return scores


# Lightning, which COMET drives internally, reads these to auto-detect a
# distributed launch. Under `accelerate launch` they describe *this* script's
# process group, which each rank's single-device COMET Trainer must not join.
DISTRIBUTED_ENV_VARS = (
    "WORLD_SIZE", "RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK",
    "NODE_RANK", "MASTER_ADDR", "MASTER_PORT", "TORCHELASTIC_RUN_ID",
)


@contextmanager
def _distributed_env_hidden():
    """Hide the launcher's distributed environment for the duration of a block.

    torch.distributed is already bootstrapped by the time this runs, so the
    existing process group is unaffected; only libraries that re-read the
    environment (Lightning) see a plain single-process launch.
    """
    saved = {name: os.environ.pop(name) for name in DISTRIBUTED_ENV_VARS if name in os.environ}
    try:
        yield
    finally:
        os.environ.update(saved)


def _download_comet_checkpoint(model_id):
    """Resolve the checkpoint, downloading once on the main process.

    Concurrent download_model calls from every rank race on the same cache
    directory. Called by all ranks so the barrier stays symmetric.
    """
    from comet import download_model

    state = PartialState()
    if state.is_main_process:
        logger.info("Resolving COMET checkpoint ([cyan]%s[/cyan])...", model_id)
        download_model(model_id)
    state.wait_for_everyone()
    return download_model(model_id)


def evaluate_comet(sources, hypotheses, references, config, prefix="", force=False):
    """Score with a COMET checkpoint (higher is better), sharded across ranks."""
    eval_cfg = config["evaluation"]
    state = PartialState()
    checkpoint = _download_comet_checkpoint(eval_cfg["comet_model_id"])

    def score_fn(indices):
        from comet import load_from_checkpoint

        model = load_from_checkpoint(checkpoint)
        try:
            # Quality-estimation checkpoints (CometKiwi, XCOMET-*-QE) reject a
            # "ref" key; XCOMET and wmt22-comet-da require it. The checkpoint
            # decides, so it is configured rather than sniffed from the model id.
            reference_free = eval_cfg.get("comet_reference_free", False)
            data = [
                {"src": sources[i], "mt": hypotheses[i]}
                if reference_free
                else {"src": sources[i], "mt": hypotheses[i], "ref": references[i]}
                for i in indices
            ]
            # devices pins this rank's Trainer to the GPU accelerate already gave
            # it. Without it every rank's Trainer would default to cuda:0 and
            # stack every copy of the model onto one device.
            predict_kwargs = (
                {"gpus": eval_cfg["comet_gpus"], "devices": [state.local_process_index]}
                if torch.cuda.is_available()
                else {"gpus": 0}
            )
            with _distributed_env_hidden():
                output = model.predict(
                    data,
                    batch_size=eval_cfg["comet_batch_size"],
                    progress_bar=False,
                    **predict_kwargs,
                )
            yield from zip(indices, output.scores)
        finally:
            del model

    scores = _score_sharded("comet", len(sources), config, prefix, force, score_fn, "green")
    # Recomputed rather than taken from output.system_score: each rank only sees
    # its own shard, and for these checkpoints the system score is the mean of
    # the segment scores.
    system_score = float(pd.Series(scores).mean())
    if state.is_main_process:
        logger.info(
            "COMET finished for [bold green]%s[/bold green]. System score (higher is better): [bold yellow]%.4f[/bold yellow]",
            prefix, system_score,
        )
    return scores, system_score


def _write_human_review_sample(results, config, output_dir, prefix):
    data_cfg, eval_cfg = config["data"], config["evaluation"]
    domain_column = data_cfg["domain_column"]
    sample = (results.groupby(domain_column, group_keys=False)
              .apply(lambda group: group.sample(n=min(len(group), eval_cfg["human_review_samples_per_domain"]), random_state=eval_cfg["human_review_seed"]))
              .reset_index(drop=True))
    candidates = [
        data_cfg["id_column"], domain_column, data_cfg["source_column"],
        data_cfg["target_column"], "generated_farsi", "metricx_score", "comet_score",
        *TRANSPARENT_METRICS,
    ]
    columns = [column for column in candidates if column in sample]
    sample[columns].to_csv(output_dir / f"{prefix}_{eval_cfg['human_review_filename']}", index=False)


def _run_one(config, adapter_path, prefix, force=False):
    data_cfg, eval_cfg = config["data"], config["evaluation"]
    is_main = PartialState().is_main_process
    if is_main:
        console.print(Panel(f"[bold green]Starting Evaluation Stage: {prefix.upper()}[/bold green]\nAdapter: {adapter_path or 'Base Model'}", border_style="cyan"))
    test_df = pd.read_json(data_cfg["test_dataset_path"], lines=True)
    required = {data_cfg["source_column"], data_cfg["target_column"], data_cfg["domain_column"]}
    if missing := required - set(test_df.columns):
        raise ValueError(f"Test dataset is missing columns: {sorted(missing)}")
    if max_examples := eval_cfg.get("smoke_test_max_examples"):
        test_df = test_df.head(max_examples).copy()
        if is_main:
            logger.info("Limiting evaluation to %s examples for smoke test.", len(test_df))
    sources, references = test_df[data_cfg["source_column"]].tolist(), test_df[data_cfg["target_column"]].tolist()
    results = test_df.copy()
    results["generated_farsi"] = generate_translations(test_df, config, adapter_path, prefix=prefix, force=force)
    summary = {"label": prefix, "examples": len(results), "adapter_path": adapter_path}
    if eval_cfg.get("degeneration_audit_enabled", True):
        # Runs before the scoring models load. MetricX and COMET are corpus
        # averages and cannot express "this system stopped translating and
        # filled the token budget", so this is the only check in the pipeline
        # that can fail a structurally broken run.
        audit = audit_outputs(results["generated_farsi"], references)
        summary["degeneration"] = {
            key: audit[key]
            for key in ("rows", "clean_rows", "clean_rate", "failure_rate", "failures",
                        "mean_chars", "mean_chars_trimmed", "mean_trailing_chars",
                        "max_trailing_chars")
        }
        if is_main:
            logger.info(
                "Decoding audit for [bold green]%s[/bold green]: clean [bold]%.1f%%[/bold] "
                "(%d/%d), mean chars %.0f (trailing %.0f)",
                prefix, 100 * audit["clean_rate"], audit["clean_rows"], audit["rows"],
                audit["mean_chars"], audit["mean_trailing_chars"],
            )
            for failure, values in audit["failures"].items():
                logger.warning("  %s: %d rows (%.2f%%)", failure, values["rows"], 100 * values["rate"])
    if eval_cfg.get("transparent_metrics_enabled", True) and is_main:
        # Runs on the main process only: CPU-bound and cheap next to generation,
        # and only the main process writes the detailed CSV these land in.
        columns = evaluate_transparent_metrics(
            sources, results["generated_farsi"].tolist(), references, config, prefix=prefix
        )
        summary["transparent"] = {}
        for name, values in columns.items():
            # Positional assignment: the scored frame is freshly built, while
            # results carries the test split's own index.
            results[name] = values.to_numpy()
            summary["transparent"][name] = {
                "mean": float(values.mean()),
                "direction": METRIC_DIRECTIONS[name],
                "scored_rows": int(values.notna().sum()),
            }
    if eval_cfg["metricx_enabled"]:
        results["metricx_score"] = evaluate_metricx(sources, results["generated_farsi"].tolist(), references, config, prefix=prefix, force=force)
        summary["metricx_mean_lower_is_better"] = float(results["metricx_score"].mean())
    if eval_cfg["comet_enabled"]:
        # Every rank participates: evaluate_comet shards like MetricX and
        # contains a barrier, so a main-process-only call would deadlock.
        scores, system_score = evaluate_comet(
            sources, results["generated_farsi"].tolist(), references, config, prefix=prefix, force=force
        )
        results["comet_score"] = scores
        summary["comet_system_score_higher_is_better"] = float(system_score)
    output_dir = Path(eval_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if is_main:
        detailed_path = output_dir / f"{prefix}_{eval_cfg['detailed_filename']}"
        sample_path = output_dir / f"{prefix}_{eval_cfg['human_review_filename']}"
        results.to_csv(detailed_path, index=False)
        logger.info("Detailed results saved to [bold]%s[/bold]", detailed_path)
        _write_human_review_sample(results, config, output_dir, prefix)
        logger.info("Human review sample saved to [bold]%s[/bold]", sample_path)
    return summary


def run_evaluation(config, adapter_path=None, force=False):
    eval_cfg = config["evaluation"]
    adapter_path = adapter_path or eval_cfg["adapter_path"]
    if not adapter_path:
        raise ValueError("Provide an adapter path or set evaluation.adapter_path before adapter evaluation.")
    adapter_path = resolve_adapter_path(adapter_path)
    summaries = []
    if eval_cfg["run_baseline"]:
        summaries.append(_run_one(config, None, eval_cfg["baseline_prefix"], force=force))
    summaries.append(_run_one(config, adapter_path, eval_cfg["adapter_prefix"], force=force))

    is_main = PartialState().is_main_process
    if is_main:
        output_path = Path(eval_cfg["output_dir"]) / eval_cfg["summary_filename"]
        output_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
        _render_summary(summaries, eval_cfg, output_path)
        logger.info("Evaluation results saved to [bold]%s[/bold]", output_path)

    _enforce_degeneration_gate(summaries, eval_cfg)
    return summaries


# Display labels and units for every metric this script can report. Metric keys
# are the summary keys, not the DataFrame column names, so a metric is added to
# the report by adding it here and to _collect_metrics.
METRIC_LABELS = {
    "clean_rate": "Clean decoding",
    "metricx": "MetricX",
    "comet": "COMET",
    "sentence_bleu": "BLEU",
    "sentence_chrf": "chrF++",
    "number_preservation": "Numbers kept",
    "acronym_preservation": "Acronyms kept",
    "formula_preservation": "Formulas kept",
    "empty_output": "Empty output",
    "source_copy": "Source copied",
}
# Rates, rendered as percentages; their deltas are therefore percentage points.
PERCENT_METRICS = frozenset({
    "clean_rate", "number_preservation", "acronym_preservation",
    "formula_preservation", "empty_output", "source_copy",
})
# Shown in the narrow per-system table. Everything else lives in the delta
# table, which grows downward and so never widens past a 80-column console.
HEADLINE_METRICS = ("clean_rate", "metricx", "comet")


def _collect_metrics(summary):
    """Flatten one system's summary into {name: (value, direction, scored_rows)}.

    scored_rows is the denominator behind the mean and is None where it equals
    the example count. Preservation metrics skip rows whose source contains no
    number/acronym/formula, so their mean is over a subset.
    """
    metrics = {}
    if audit := summary.get("degeneration"):
        metrics["clean_rate"] = (audit["clean_rate"], "higher", None)
    if "metricx_mean_lower_is_better" in summary:
        metrics["metricx"] = (summary["metricx_mean_lower_is_better"], METRIC_DIRECTIONS["metricx"], None)
    if "comet_system_score_higher_is_better" in summary:
        metrics["comet"] = (summary["comet_system_score_higher_is_better"], METRIC_DIRECTIONS["comet"], None)
    for name, entry in (summary.get("transparent") or {}).items():
        metrics[name] = (entry["mean"], entry["direction"], entry["scored_rows"])
    return metrics


def _is_missing(value):
    return value is None or (isinstance(value, float) and math.isnan(value))


def _format_metric(name, value):
    if _is_missing(value):
        return "—"
    if name in PERCENT_METRICS:
        return f"{100 * value:.1f}%"
    if name in {"sentence_bleu", "sentence_chrf"}:
        return f"{value:.2f}"
    return f"{value:.4f}"


def _format_delta(name, delta):
    if _is_missing(delta):
        return "—"
    if name in PERCENT_METRICS:
        return f"{100 * delta:+.1f} pp"
    if name in {"sentence_bleu", "sentence_chrf"}:
        return f"{delta:+.2f}"
    return f"{delta:+.4f}"


def _short_system(summary):
    """Name a system by its adapter directory rather than its absolute path.

    The full path is already in summary.json; repeating it here is what pushed
    the old single table past the console width.
    """
    if not summary["adapter_path"]:
        return "base model"
    path = Path(summary["adapter_path"])
    return "/".join(path.parts[-2:]) if len(path.parts) > 1 else path.name


def _render_systems_table(summaries):
    table = Table(title="Systems", title_style="bold yellow", header_style="bold yellow", border_style="green")
    table.add_column("Label", style="cyan")
    table.add_column("System", style="dim")
    table.add_column("Examples", style="white", justify="right")
    collected = [_collect_metrics(summary) for summary in summaries]
    shown = [name for name in HEADLINE_METRICS if any(name in metrics for metrics in collected)]
    for name in shown:
        arrow = "↑" if METRIC_DIRECTIONS.get(name, "higher") == "higher" else "↓"
        table.add_column(f"{METRIC_LABELS[name]} {arrow}", style="white", justify="right")
    for summary, metrics in zip(summaries, collected):
        row = [summary["label"], _short_system(summary), str(summary["examples"])]
        row.extend(_format_metric(name, metrics.get(name, (None,))[0]) for name in shown)
        table.add_row(*row)
    return table


def _render_delta_table(baseline, candidate):
    """Compare two systems with one row per metric.

    Metrics are rows, not columns, so the table stays readable at any console
    width no matter how many metrics are enabled -- the failure of the previous
    single-table layout, which grew a column per metric.
    """
    base_metrics, candidate_metrics = _collect_metrics(baseline), _collect_metrics(candidate)
    names = list(dict.fromkeys([*base_metrics, *candidate_metrics]))
    if not names:
        return None
    table = Table(
        title=f"{candidate['label']} vs {baseline['label']}",
        title_style="bold yellow", header_style="bold yellow", border_style="green",
    )
    table.add_column("Metric", style="cyan")
    table.add_column("Better", style="dim")
    table.add_column(baseline["label"], justify="right")
    table.add_column(candidate["label"], justify="right")
    table.add_column("Δ", justify="right")
    table.add_column("Scored rows", style="dim", justify="right")
    for name in names:
        base_value, direction, _ = base_metrics.get(name, (None, METRIC_DIRECTIONS.get(name, "higher"), None))
        candidate_value, _, scored_rows = candidate_metrics.get(name, (None, direction, None))
        if _is_missing(base_value) or _is_missing(candidate_value):
            delta_cell = "[dim]—[/dim]"
        else:
            delta = candidate_value - base_value
            # A metric is improved when it moved in its own favourable
            # direction, which is not the same as the sign of the delta:
            # MetricX and the empty_output/source_copy rates are lower-better.
            improved = delta > 0 if direction == "higher" else delta < 0
            style = "dim" if delta == 0 else ("green" if improved else "red")
            delta_cell = f"[{style}]{_format_delta(name, delta)}[/{style}]"
        table.add_row(
            METRIC_LABELS.get(name, name),
            "higher" if direction == "higher" else "lower",
            _format_metric(name, base_value),
            _format_metric(name, candidate_value),
            delta_cell,
            "—" if scored_rows is None else str(scored_rows),
        )
    return table


def _render_failures_table(summaries):
    """Break down decoding failures by class, which was previously log-only."""
    rows = [
        (summary["label"], failure, values["rows"], values["rate"])
        for summary in summaries
        for failure, values in (summary.get("degeneration", {}).get("failures") or {}).items()
        if values["rows"]
    ]
    if not rows:
        return None
    table = Table(title="Decoding failures", title_style="bold yellow", header_style="bold yellow", border_style="red")
    table.add_column("Label", style="cyan")
    table.add_column("Failure", style="white")
    table.add_column("Rows", justify="right")
    table.add_column("Rate", justify="right")
    for label, failure, count, rate in sorted(rows, key=lambda row: (row[0], -row[3])):
        table.add_row(label, failure, str(count), f"[red]{100 * rate:.2f}%[/red]")
    return table


def _render_summary(summaries, eval_cfg, output_path):
    console.print(_render_systems_table(summaries))
    # Deltas compare the first system evaluated (the baseline, when
    # run_baseline is set) against the last one (the adapter under test).
    if len(summaries) >= 2:
        if delta_table := _render_delta_table(summaries[0], summaries[-1]):
            console.print(delta_table)
    if failures_table := _render_failures_table(summaries):
        console.print(failures_table)

    threshold = eval_cfg.get("max_degeneration_rate")
    breached = _degeneration_breaches(summaries, eval_cfg)
    if threshold is None:
        verdict = "[dim]Degeneration gate disabled (max_degeneration_rate is null).[/dim]"
    elif breached:
        detail = ", ".join(f"{label} {rate:.1%}" for label, rate in sorted(breached.items()))
        verdict = f"[bold red]FAILED[/bold red] decoding gate at {threshold:.1%}: {detail}"
    else:
        verdict = f"[bold green]PASSED[/bold green] decoding gate at {threshold:.1%}"
    output_dir = Path(eval_cfg["output_dir"])
    console.print(Panel(
        f"{verdict}\n\n"
        f"[bold]Summary[/bold]      {output_path}\n"
        f"[bold]Per-example[/bold]  {output_dir}/<label>_{eval_cfg['detailed_filename']}\n"
        f"[bold]Human review[/bold] {output_dir}/<label>_{eval_cfg['human_review_filename']}",
        title="Evaluation complete", border_style="red" if breached else "green",
    ))


def _degeneration_breaches(summaries, eval_cfg):
    """Systems whose decoding failure rate exceeds the configured threshold.

    Shared by the printed verdict and the gate so the report and the exit status
    can never disagree. Returns empty when the gate is disabled.
    """
    threshold = eval_cfg.get("max_degeneration_rate")
    if threshold is None:
        return {}
    return {
        summary["label"]: summary["degeneration"]["failure_rate"]
        for summary in summaries
        if "degeneration" in summary and summary["degeneration"]["failure_rate"] > threshold
    }


def _enforce_degeneration_gate(summaries, eval_cfg):
    """Fail the run when a system's decoding failure rate exceeds the threshold.

    Raising rather than warning is deliberate. The 2026-08-10 adapter posted the
    best eval_loss of its run while 87% of its output was unusable, and nothing
    in the pipeline objected; a human noticed by reading the HTML report. Set
    evaluation.max_degeneration_rate to null to report without failing.
    """
    threshold = eval_cfg.get("max_degeneration_rate")
    if breached := _degeneration_breaches(summaries, eval_cfg):
        detail = ", ".join(f"{label}={rate:.1%}" for label, rate in sorted(breached.items()))
        raise RuntimeError(
            f"Decoding failure rate above evaluation.max_degeneration_rate={threshold:.1%}: {detail}. "
            "Inspect <prefix>_detailed_scores.csv, or run scripts/audit_degeneration.py for the "
            "per-class breakdown."
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--force", action="store_true", help="Ignore existing cache files and re-generate from scratch.")
    args = parser.parse_args()
    config = load_config(args.config)
    setup_logging(config, run_name="evaluation")
    if PartialState().is_main_process:
        log_config_summary(config)
    run_evaluation(config, args.adapter_path, force=args.force)


if __name__ == "__main__":
    main()




