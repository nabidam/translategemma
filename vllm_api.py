import openai

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

# The alias defined in --lora-modules
adapter_name = "farsi-science"

response = client.chat.completions.create(
    model=adapter_name,
    messages=[
        {
            "role": "user",
            "content": "<<<source>>>en<<<target>>>fa<<<text>>>Cellular biology is the study of cell structure.",
        }
    ],
)

print(response.choices[0].message.content)
