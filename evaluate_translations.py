import yaml
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from peft import PeftModel
from comet import download_model, load_from_checkpoint

# NOTE: MetricX uses a custom T5 regression architecture.
# You need the official MetricX codebase to load it properly:
# pip install git+https://github.com/google-research/metricx.git
try:
    from metricx23.models import MT5ForRegression
except ImportError:
    raise ImportError(
        "Please install metricx: pip install git+https://github.com/google-research/metricx.git"
    )


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def generate_translations(test_df, config):
    print(f"Loading Base Model: {config['model']['base_model_id']}...")
    processor = AutoProcessor.from_pretrained(config["model"]["base_model_id"])

    base_model = AutoModelForCausalLM.from_pretrained(
        config["model"]["base_model_id"],
        device_map=config["model"]["device_map"],
        torch_dtype=torch.bfloat16 if config["training"]["bf16"] else torch.float16,
    )

    print(f"Applying LoRA Adapter from: {config['evaluation']['adapter_path']}...")
    model = PeftModel.from_pretrained(base_model, config["evaluation"]["adapter_path"])

    hypotheses = []
    source_lang = config["data"]["source_lang"]
    target_lang = config["data"]["target_lang"]

    print("Generating Farsi translations...")
    for idx, row in test_df.iterrows():
        # Strict TranslateGemma formatting
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": source_lang,
                        "target_lang_code": target_lang,
                        "text": row["english"],
                    }
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(model.device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,  # Always greedy decoding for eval
            )

        input_len = inputs["input_ids"].shape[-1]
        translation = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
        hypotheses.append(translation)

        if idx % 10 == 0 and idx > 0:
            print(f"Translated {idx}/{len(test_df)}...")

    # Clean up VRAM to make room for evaluation models
    del model, base_model, processor
    torch.cuda.empty_cache()

    return hypotheses


def evaluate_metricx(sources, hypotheses, references, config):
    print(f"\nLoading MetricX: {config['evaluation']['metricx_model_id']}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(config["evaluation"]["metricx_model_id"])
    model = MT5ForRegression.from_pretrained(
        config["evaluation"]["metricx_model_id"]
    ).to(device)
    model.eval()

    scores = []
    print("Scoring with MetricX-24 (Lower is better, [0-25])...")
    with torch.no_grad():
        for src, hyp, ref in zip(sources, hypotheses, references):
            # MetricX-24 Hybrid expects inputs formatted as: "ref: {ref} hyp: {hyp} src: {src}"
            input_text = f"ref: {ref} hyp: {hyp} src: {src}"
            inputs = tokenizer(
                input_text, return_tensors="pt", truncation=True, max_length=1024
            ).to(device)

            output = model(**inputs)
            # MetricX-24 scores are strictly clamped between 0 and 25
            score = torch.clamp(output.logits, min=0.0, max=25.0).item()
            scores.append(score)

    del model, tokenizer
    torch.cuda.empty_cache()
    return scores


def evaluate_comet(sources, hypotheses, references, config):
    print(f"\nLoading COMET: {config['evaluation']['comet_model_id']}...")
    model_path = download_model(config["evaluation"]["comet_model_id"])
    model = load_from_checkpoint(model_path)

    data = [
        {"src": src, "mt": hyp, "ref": ref}
        for src, hyp, ref in zip(sources, hypotheses, references)
    ]

    print("Scoring with COMET (Higher is better, [0-1])...")
    device = 1 if torch.cuda.is_available() else 0
    model_output = model.predict(
        data, batch_size=config["evaluation"]["comet_batch_size"], gpus=device
    )

    del model
    torch.cuda.empty_cache()

    return model_output.scores, model_output.system_score


if __name__ == "__main__":
    config = load_config()

    test_path = config["data"]["test_dataset_path"]
    print(f"Loading test data from {test_path}...")
    test_df = pd.read_json(test_path, lines=True)

    sources = test_df["english"].tolist()
    references = test_df["farsi"].tolist()

    # 1. Generate translations
    hypotheses = generate_translations(test_df, config)
    test_df["generated_farsi"] = hypotheses

    # 2. Score with MetricX
    metricx_scores = evaluate_metricx(sources, hypotheses, references, config)
    test_df["metricx_score"] = metricx_scores

    # 3. Score with COMET
    comet_scores, comet_system_score = evaluate_comet(
        sources, hypotheses, references, config
    )
    test_df["comet_score"] = comet_scores

    # 4. Save and summarize
    output_file = config["evaluation"]["output_file"]
    test_df.to_csv(output_file, index=False)

    print("\n" + "=" * 40)
    print("EVALUATION COMPLETE")
    print("=" * 40)
    print(
        f"MetricX System Score (0-25, LOWER is better): {sum(metricx_scores)/len(metricx_scores):.4f}"
    )
    print(f"COMET System Score (0-1, HIGHER is better): {comet_system_score:.4f}")
    print(f"\nDetailed segment scores saved to {output_file}")
