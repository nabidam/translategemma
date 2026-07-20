import pandas as pd
import json


def convert_csv_to_jsonl(
    input_csv: str,
    sft_output: str = "data/sft_farsi_science.jsonl",
    dpo_output: str | None = "data/dpo_farsi_science.jsonl",
):
    """
    Converts a raw CSV with columns ['id', 'en', 'fa', 'domain']
    into Hugging Face-ready JSONL files for the TranslateGemma pipeline.
    """
    print(f"Loading CSV from {input_csv}...")

    # 1. Load data and validate columns
    df = pd.read_csv(input_csv)
    required_cols = {"id", "en", "fa", "domain"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"CSV missing required columns. Found: {df.columns}. Expected: {required_cols}"
        )

    # Drop any rows with missing translations
    df = df.dropna(subset=["en", "fa"]).copy()
    print(f"Found {len(df)} valid translation pairs.")

    if dpo_output is None:
        return

    # ==========================================
    # 2. Prepare SFT Dataset
    # ==========================================
    # train.py expects keys: "english" and "farsi"
    sft_df = pd.DataFrame(
        {"id": df["id"], "domain": df["domain"], "english": df["en"], "farsi": df["fa"]}
    )

    # Save SFT to JSONL
    # orient="records" creates a list of dictionaries, lines=True writes them line-by-line
    sft_df.to_json(sft_output, orient="records", lines=True, force_ascii=False)
    print(f"✅ SFT dataset saved to {sft_output}")

    # ==========================================
    # 3. Prepare DPO Dataset (Scaffolding)
    # ==========================================
    # train.py expects keys: "english", "farsi_chosen", and "farsi_rejected"
    # Note: True DPO requires negative examples. Here, we create the structure.
    # You MUST replace the placeholder with actual bad/rejected translations.

    dpo_df = pd.DataFrame(
        {
            "id": df["id"],
            "domain": df["domain"],
            "english": df["en"],
            "farsi_chosen": df["fa"],
            # TODO: Replace this placeholder with generated negative examples
            # (e.g., transliterated Farsi instead of proper scientific terminology)
            "farsi_rejected": df["en"].apply(
                lambda x: "[REQUIRES_REJECTED_TRANSLATION_GENERATION]"
            ),
        }
    )

    dpo_df.to_json(dpo_output, orient="records", lines=True, force_ascii=False)
    print(f"✅ DPO dataset scaffold saved to {dpo_output}")
    print(
        "\n⚠️  NOTE: You must populate 'farsi_rejected' in the DPO dataset before running the RL phase."
    )


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Convert raw CSV into SFT and DPO JSONL datasets."
    )
    parser.add_argument(
        "--input_csv",
        default="raw_scientific_translations.csv",
        help="Path to input CSV with columns [id, en, fa, domain].",
    )
    parser.add_argument(
        "--sft_output",
        default="data/sft_farsi_science.jsonl",
        help="Path to write the SFT JSONL file.",
    )
    parser.add_argument(
        "--dpo_output",
        default="data/dpo_farsi_science.jsonl",
        help="Path to write the DPO JSONL scaffold file.",
    )
    parser.add_argument(
        "--skip_dpo",
        action="store_true",
        help="Write only the SFT-compatible JSONL. Use this for a held-out test CSV.",
    )
    args = parser.parse_args()

    # Ensure the parent directories for the output files exist
    output_paths = [args.sft_output]
    if not args.skip_dpo:
        output_paths.append(args.dpo_output)
    for out_path in output_paths:
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    # Run conversion
    convert_csv_to_jsonl(
        input_csv=args.input_csv,
        sft_output=args.sft_output,
        dpo_output=None if args.skip_dpo else args.dpo_output,
    )
