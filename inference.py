import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import PeftModel

# 1. Load Base Model and Processor
base_model_id = "google/translategemma-12b-it"
processor = AutoProcessor.from_pretrained(base_model_id)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id, device_map="auto", torch_dtype=torch.bfloat16
)

# 2. Merge your trained Farsi domain adapter
model = PeftModel.from_pretrained(base_model, "./final_farsi_adapter")

# 3. Format the input strictly following the TranslateGemma schema
english_text = "The model relies on multi-query attention and RoPE embeddings to process the genome sequence."

messages = [
    {
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
]

# 4. Apply the template and generate
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,  # Tells the model to start the assistant's turn
    return_dict=True,
    return_tensors="pt",
).to(model.device)

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,  # Use greedy decoding for translations (temperature=0)
    )

# 5. Decode only the newly generated tokens
input_len = inputs["input_ids"].shape[-1]
translation = processor.decode(outputs[0][input_len:], skip_special_tokens=True)

print(f"English: {english_text}")
print(f"Farsi: {translation}")
