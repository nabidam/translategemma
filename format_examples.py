import json

from language_pairs import resolve_language_pair


def format_to_translategemma_schema(examples):
    conversations = []
    row_count = len(examples["english"])
    source_langs = examples.get("src_lang", [None] * row_count)
    target_langs = examples.get("tgt_lang", [None] * row_count)
    for source_text, target_text, row_source_lang, row_target_lang in zip(
        examples["english"], examples["farsi"], source_langs, target_langs
    ):
        source_lang, target_lang = resolve_language_pair(
            {
                "src_lang": row_source_lang,
                "tgt_lang": row_target_lang,
            },
            {"source_lang": "en", "target_lang": "fa"},
        )
        message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": source_lang,
                        "target_lang_code": target_lang,
                        "text": source_text,
                    }
                ],
            },
            {
                "role": "assistant",
                "content": str(target_text),  # Ensure this is a raw string
            },
        ]
        conversations.append(
            {"messages": message}
        )  # The Trainer expects a 'messages' key
    return {"messages": [c["messages"] for c in conversations]}


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
