import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from accelerate import Accelerator
from peft import PeftModel
from prompting import (
    render_training_prompts,
    resolve_stop_token_ids,
    tokenize_prompts_for_generation,
)
from train import load_generation_safe_model_config, make_deterministic_generation_config

# 1. Load Base Model and Processor
base_model_id = "google/translategemma-12b-it"
processor = AutoProcessor.from_pretrained(
    base_model_id, use_fast=True, fix_mistral_regex=False
)
# Required by tokenize_prompts_for_generation, and by batched decoding generally.
processor.tokenizer.padding_side = "left"
model_config = load_generation_safe_model_config(base_model_id)
stop_token_ids = resolve_stop_token_ids(processor.tokenizer, base_model_id=base_model_id)

accelerator = Accelerator()
load_kwargs = {
    "config": model_config,
    "generation_config": make_deterministic_generation_config(
        model_config, processor, base_model_id
    ),
    "torch_dtype": torch.bfloat16,
}

base_model = AutoModelForCausalLM.from_pretrained(base_model_id, **load_kwargs)
base_model = base_model.to(accelerator.device)

# 2. Merge your trained Farsi domain adapter
model = PeftModel.from_pretrained(base_model, "./final_farsi_adapter")

# 3. Format the input strictly following the TranslateGemma schema
english_text = "The model relies on multi-query attention and RoPE embeddings to process the genome sequence."

user_message = {
    "role": "user",
    "content": [
        {
            "type": "text",
            "source_lang_code": "en",
            "target_lang_code": "fa",
            "text": english_text,
        }
    ],
}

# 4. Render the prompt the adapter was trained to continue, and generate.
# Not add_generation_prompt=True: that omits the assistant-turn indentation the
# SFT rendering emits. See prompting.py.
prompts = render_training_prompts(processor, [user_message])
inputs = tokenize_prompts_for_generation(processor, prompts, model.device)

with torch.inference_mode():
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,  # Use deterministic greedy decoding for translation.
        temperature=1.0,
        top_p=1.0,
        top_k=50,
        pad_token_id=pad_token_id,
        eos_token_id=stop_token_ids,
    )

# 5. Decode only the newly generated tokens
input_len = inputs["input_ids"].shape[-1]
translation = processor.decode(outputs[0][input_len:], skip_special_tokens=True)

print(f"English: {english_text}")
print(f"Farsi: {translation}")
