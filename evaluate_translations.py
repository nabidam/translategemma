"""Evaluate a base TranslateGemma model or a LoRA adapter on the configured test split."""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from logging_utils import console, logger, setup_logging, log_config_summary, load_config
from train import resolve_dtype


def generate_translations(test_df, config, adapter_path=None):
    model_cfg, eval_cfg, data_cfg = config["model"], config["evaluation"], config["data"]
    processor = AutoProcessor.from_pretrained(model_cfg["base_model_id"])
    # Same model.dtype / model.attn_implementation the adapter was trained
    # under, so evaluation never silently measures a different numeric setup.
    base_model = AutoModelForCausalLM.from_pretrained(
        model_cfg["base_model_id"], device_map=model_cfg["device_map"],
        dtype=resolve_dtype(model_cfg["dtype"]),
        attn_implementation=model_cfg["attn_implementation"],
    )
    model = PeftModel.from_pretrained(base_model, adapter_path) if adapter_path else base_model
    model.eval()
    hypotheses = []
    for _, row in test_df.iterrows():
        source = row[data_cfg["source_column"]]
        messages = [{"role": "user", "content": [{"type": "text", "source_lang_code": data_cfg["source_lang"], "target_lang_code": data_cfg["target_lang"], "text": source}]}]
        inputs = processor.apply_chat_template(messages, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generation_kwargs = {
                "max_new_tokens": eval_cfg["max_new_tokens"], "do_sample": eval_cfg["do_sample"],
                "num_beams": eval_cfg["num_beams"],
            }
            if eval_cfg["do_sample"]:
                generation_kwargs.update(temperature=eval_cfg["temperature"], top_p=eval_cfg["top_p"])
            outputs = model.generate(**inputs, **generation_kwargs)
        hypotheses.append(processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
    del model, base_model, processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return hypotheses


def evaluate_metricx(sources, hypotheses, references, config):
    """Score with a reference-based MetricX-24 hybrid model (lower is better).

    The input serialisation, the stripped EOS token and the tokenizer choice all
    follow metricx24/predict.py exactly. MetricX is sensitive to every one of
    them: a different prompt order or a trailing EOS silently shifts the scores
    instead of raising, so this must not be "simplified".
    """
    try:
        from metricx24.models import MT5ForRegression
    except ImportError as error:
        raise ImportError("Install MetricX or set evaluation.metricx_enabled: false.") from error
    eval_cfg = config["evaluation"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # MetricX checkpoints ship weights only; the tokenizer comes from the mT5
    # model they were initialised from.
    tokenizer = AutoTokenizer.from_pretrained(eval_cfg["metricx_tokenizer_id"])
    model = MT5ForRegression.from_pretrained(eval_cfg["metricx_model_id"], torch_dtype="auto").to(device).eval()
    scores = []
    with torch.inference_mode():
        for source, hypothesis, reference in zip(sources, hypotheses, references):
            text = f"source: {source} candidate: {hypothesis} reference: {reference}"
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=eval_cfg["metricx_max_length"], padding=False)
            # The models were trained on inputs without a trailing EOS token.
            inputs = {key: value[:, :-1].to(device) for key, value in inputs.items()}
            # forward() already clamps its regression output to [0, 25].
            scores.append(model(**inputs).predictions.item())
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return scores


def evaluate_comet(sources, hypotheses, references, config):
    from comet import download_model, load_from_checkpoint
    eval_cfg = config["evaluation"]
    model = load_from_checkpoint(download_model(eval_cfg["comet_model_id"]))
    data = [{"src": source, "mt": hypothesis, "ref": reference} for source, hypothesis, reference in zip(sources, hypotheses, references)]
    output = model.predict(data, batch_size=eval_cfg["comet_batch_size"], gpus=eval_cfg["comet_gpus"] if torch.cuda.is_available() else 0)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output.scores, output.system_score


def _write_human_review_sample(results, config, output_dir, prefix):
    data_cfg, eval_cfg = config["data"], config["evaluation"]
    domain_column = data_cfg["domain_column"]
    sample = (results.groupby(domain_column, group_keys=False)
              .apply(lambda group: group.sample(n=min(len(group), eval_cfg["human_review_samples_per_domain"]), random_state=eval_cfg["human_review_seed"]))
              .reset_index(drop=True))
    columns = [column for column in [data_cfg["id_column"], domain_column, data_cfg["source_column"], data_cfg["target_column"], "generated_farsi", "metricx_score", "comet_score"] if column in sample]
    sample[columns].to_csv(output_dir / f"{prefix}_{eval_cfg['human_review_filename']}", index=False)


def _run_one(config, adapter_path, prefix):
    data_cfg, eval_cfg = config["data"], config["evaluation"]
    test_df = pd.read_json(data_cfg["test_dataset_path"], lines=True)
    required = {data_cfg["source_column"], data_cfg["target_column"], data_cfg["domain_column"]}
    if missing := required - set(test_df.columns):
        raise ValueError(f"Test dataset is missing columns: {sorted(missing)}")
    if max_examples := eval_cfg.get("smoke_test_max_examples"):
        test_df = test_df.head(max_examples).copy()
        logger.info("Limiting evaluation to %s examples for smoke test.", len(test_df))
    sources, references = test_df[data_cfg["source_column"]].tolist(), test_df[data_cfg["target_column"]].tolist()
    results = test_df.copy()
    results["generated_farsi"] = generate_translations(test_df, config, adapter_path)
    summary = {"label": prefix, "examples": len(results), "adapter_path": adapter_path}
    if eval_cfg["metricx_enabled"]:
        results["metricx_score"] = evaluate_metricx(sources, results["generated_farsi"].tolist(), references, config)
        summary["metricx_mean_lower_is_better"] = float(results["metricx_score"].mean())
    if eval_cfg["comet_enabled"]:
        scores, system_score = evaluate_comet(sources, results["generated_farsi"].tolist(), references, config)
        results["comet_score"] = scores
        summary["comet_system_score_higher_is_better"] = float(system_score)
    output_dir = Path(eval_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / f"{prefix}_{eval_cfg['detailed_filename']}", index=False)
    _write_human_review_sample(results, config, output_dir, prefix)
    return summary


def run_evaluation(config, adapter_path=None):
    eval_cfg = config["evaluation"]
    adapter_path = adapter_path or eval_cfg["adapter_path"]
    if not adapter_path:
        raise ValueError("Provide an adapter path or set evaluation.adapter_path before adapter evaluation.")
    summaries = []
    if eval_cfg["run_baseline"]:
        summaries.append(_run_one(config, None, eval_cfg["baseline_prefix"]))
    summaries.append(_run_one(config, adapter_path, eval_cfg["adapter_prefix"]))
    output_path = Path(eval_cfg["output_dir"]) / eval_cfg["summary_filename"]
    output_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print_json(json.dumps(summaries, ensure_ascii=False))
    logger.info("Evaluation results saved to %s", output_path)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--adapter-path", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    setup_logging(config, run_name="evaluation")
    log_config_summary(config)
    run_evaluation(config, args.adapter_path)


if __name__ == "__main__":
    main()
