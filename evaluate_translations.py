import torch
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from peft import PeftModel
from comet import download_model, load_from_checkpoint

# ==========================================
# 1. Configuration
# ==========================================
BASE_MODEL_ID = "google/translategemma-12b-it"
ADAPTER_PATH = "./translategemma-farsi-science/dpo_final"  # Path to your trained LoRA
TEST_DATA_PATH = "data/test_farsi_science.jsonl"
OUTPUT_FILE = "evaluation_results.csv"

# Using MetricX-23 Large.
# MetricX scores from 0 (perfect) to 25 (terrible)
METRICX_MODEL_ID = "google/metricx-23-large-v2p0"

# Using COMET-22.
# COMET scores from 0 to 1 (higher is better)
COMET_MODEL_ID = "Unbabel/wmt22-comet-da"

device = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================
# 2. Translation Generation Function
# ==========================================
def generate_translations(test_df):
    print("Loading Translation Model...")
    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16
    )
    # Apply your fine-tuned LoRA weights
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

    hypotheses = []

    print("Generating Farsi translations...")
    for idx, row in test_df.iterrows():
        # Strict TranslateGemma formatting
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": "en",
                        "target_lang_code": "fa",
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
                do_sample=False,  # Always greedy decoding for translation eval
            )

        input_len = inputs["input_ids"].shape[-1]
        translation = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
        hypotheses.append(translation)

        if idx % 10 == 0:
            print(f"Translated {idx}/{len(test_df)}...")

    # Clean up VRAM before loading evaluation models
    del model, base_model, processor
    torch.cuda.empty_cache()

    return hypotheses


# ==========================================
# 3. Evaluation Setup
# ==========================================
def evaluate_metricx(sources, hypotheses, references):
    print("\nLoading MetricX...")
    tokenizer = AutoTokenizer.from_pretrained(METRICX_MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(METRICX_MODEL_ID).to(
        device
    )
    model.eval()

    scores = []
    print("Scoring with MetricX (Lower is better, [0-25])...")
    with torch.no_grad():
        for src, hyp, ref in zip(sources, hypotheses, references):
            # MetricX expects inputs formatted as: "ref: {ref} hyp: {hyp} src: {src}"
            input_text = f"ref: {ref} hyp: {hyp} src: {src}"
            inputs = tokenizer(
                input_text, return_tensors="pt", truncation=True, max_length=1024
            ).to(device)

            output = model(**inputs)
            # Clip between 0 and 25 as per MetricX design
            score = torch.clamp(output.logits, min=0.0, max=25.0).item()
            scores.append(score)

    del model, tokenizer
    torch.cuda.empty_cache()
    return scores


def evaluate_comet(sources, hypotheses, references):
    print("\nLoading COMET...")
    model_path = download_model(COMET_MODEL_ID)
    model = load_from_checkpoint(model_path)

    data = []
    for src, hyp, ref in zip(sources, hypotheses, references):
        data.append({"src": src, "mt": hyp, "ref": ref})

    print("Scoring with COMET (Higher is better, [0-1])...")
    model_output = model.predict(data, batch_size=8, gpus=1 if device == "cuda" else 0)

    del model
    torch.cuda.empty_cache()

    # model_output.scores contains segment-level scores
    # model_output.system_score contains the system-level average
    return model_output.scores, model_output.system_score


# ==========================================
# 4. Main Execution
# ==========================================
if __name__ == "__main__":
    # Load your held-out test dataset
    print(f"Loading test data from {TEST_DATA_PATH}...")
    test_df = pd.read_json(TEST_DATA_PATH, lines=True)

    sources = test_df["english"].tolist()
    references = test_df["farsi"].tolist()

    # 1. Generate translations
    hypotheses = generate_translations(test_df)
    test_df["generated_farsi"] = hypotheses

    # 2. Score with MetricX (Reference-based)
    metricx_scores = evaluate_metricx(sources, hypotheses, references)
    test_df["metricx_score"] = metricx_scores

    # 3. Score with COMET (Reference-based)
    comet_scores, comet_system_score = evaluate_comet(sources, hypotheses, references)
    test_df["comet_score"] = comet_scores

    # 4. Save and summarize
    test_df.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 40)
    print("EVALUATION COMPLETE")
    print("=" * 40)
    print(
        f"MetricX System Score (0-25, LOWER is better): {sum(metricx_scores)/len(metricx_scores):.4f}"
    )
    print(f"COMET System Score (0-1, HIGHER is better): {comet_system_score:.4f}")
    print(f"\nDetailed segment scores saved to {OUTPUT_FILE}")
