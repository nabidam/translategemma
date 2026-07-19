import asyncio
import pandas as pd
import json
import os
from google.genai import Client
from google.genai import types

# Initialize the standard Google GenAI Client
# It automatically picks up the GEMINI_API_KEY environment variable
client = Client()


async def generate_bad_translation(
    en_text: str, sem: asyncio.Semaphore, index: int
) -> str:
    """
    Calls the Gemini API asynchronously to generate a lazy, transliterated
    Persian translation for the DPO 'rejected' column.
    """
    prompt = f"""You are creating negative examples for an RLHF dataset.
I will give you an English scientific sentence. Instead of providing the correct scientific Persian terminology, you must provide a lazy translation.
Heavily transliterate the English technical words directly into Persian script rather than translating them, and use poor, amateur phrasing. 

English: {en_text}

Provide ONLY the bad Farsi translation. Do not include quotes, explanations, or introductory text."""

    async with sem:
        try:
            # We use the async client interface (.aio) with gemini-2.5-flash for speed/cost
            response = await client.aio.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.8,  # Higher temperature for more creative "bad" translations
                ),
            )

            if index % 50 == 0:
                print(f"Processed {index} records...")

            return response.text.strip()

        except Exception as e:
            print(f"⚠️ Error generating for record {index}: {e}")
            return "[API_ERROR]"


async def main(
    input_csv="raw_scientific_translations.csv",
    output_jsonl="data/dpo_farsi_science.jsonl",
):
    print(f"Loading CSV from {input_csv}...")
    df = pd.read_csv(input_csv)

    # Filter out missing data
    df = df.dropna(subset=["en", "fa"]).copy()
    print(f"Found {len(df)} valid rows. Beginning generation...")

    # Semaphore to prevent hitting rate limits (adjust based on your API tier limits)
    # 15 concurrent requests is usually safe for the paid tier. Drop to 2-3 for free tier.
    concurrency_limit = 15
    sem = asyncio.Semaphore(concurrency_limit)

    tasks = []
    for idx, row in enumerate(df.itertuples()):
        tasks.append(generate_bad_translation(row.en, sem, idx))

    # Run all API calls concurrently
    rejected_translations = await asyncio.gather(*tasks)

    # Construct the final DPO DataFrame
    dpo_df = pd.DataFrame(
        {
            "id": df["id"],
            "domain": df["domain"],
            "english": df["en"],
            "farsi_chosen": df["fa"],
            "farsi_rejected": rejected_translations,
        }
    )

    # Filter out any API errors before saving
    failed_count = len(dpo_df[dpo_df["farsi_rejected"] == "[API_ERROR]"])
    if failed_count > 0:
        print(f"Dropping {failed_count} rows that failed due to API errors.")
        dpo_df = dpo_df[dpo_df["farsi_rejected"] != "[API_ERROR]"]

    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    dpo_df.to_json(output_jsonl, orient="records", lines=True, force_ascii=False)

    print(
        f"✅ Successfully generated and saved {len(dpo_df)} DPO pairs to {output_jsonl}"
    )


if __name__ == "__main__":
    # Execute the async main loop
    asyncio.run(main())
