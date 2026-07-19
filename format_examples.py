import json


def format_to_translategemma_schema(examples):
    # This function takes batches of English and Farsi pairs
    # and formats them into the required conversation structure.
    conversations = []

    for en_text, fa_text in zip(examples["english"], examples["farsi"]):
        message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": "en",
                        "target_lang_code": "fa",
                        "text": en_text,
                    }
                ],
            },
            {"role": "assistant", "content": fa_text},
        ]
        conversations.append(message)

    return {"messages": conversations}


# Apply this to your Hugging Face dataset
# formatted_dataset = raw_dataset.map(format_to_translategemma_schema, batched=True)


"""
Example:

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "source_lang_code": "en",
                "target_lang_code": "fa",
                "text": "The standard dose of aspirin is 81 mg daily for cardiovascular protection."
            }
        ]
    },
    {
        "role": "assistant",
        "content": "دوز استاندارد آسپرین برای محافظت قلبی عروقی ۸۱ میلی‌گرم در روز است."
    }
]
"""
