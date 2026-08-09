"""Create deterministic, document-level SFT train/validation/test JSONL splits."""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from logging_utils import load_config


def _group_id(value, delimiter):
    value = str(value)
    return value.split(delimiter, 1)[0] if delimiter else value


def _load_records(path, required_columns):
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            missing = required_columns - set(record)
            if missing:
                raise ValueError(f"{path}:{line_number} is missing columns {sorted(missing)}")
            if any(not str(record[column]).strip() for column in required_columns):
                raise ValueError(f"{path}:{line_number} contains a blank required value")
            records.append(record)
    if not records:
        raise ValueError(f"{path} contains no valid JSONL records")
    return records


def _deduplicate(records, source_column, language_columns=()):
    unique, seen = [], set()
    for record in records:
        # The same text in two translation directions is not a duplicate.
        key = tuple(record.get(column) for column in language_columns) + (
            record[source_column],
        )
        if key not in seen:
            unique.append(record)
            seen.add(key)
    return unique


def _assign_groups(records, config):
    split_cfg, data_cfg = config["splitting"], config["data"]
    group_records = defaultdict(list)
    for record in records:
        group_records[_group_id(record[data_cfg["id_column"]], split_cfg["group_id_delimiter"])].append(record)

    buckets = defaultdict(list)
    for group, items in group_records.items():
        domain = Counter(item[data_cfg["domain_column"]] for item in items).most_common(1)[0][0]
        buckets[domain if split_cfg["stratify_by_domain"] else "all"].append((group, items))

    ratios = {"train": split_cfg["train_ratio"], "validation": split_cfg["validation_ratio"], "test": split_cfg["test_ratio"]}
    if ratios["train"] <= 0 or any(value < 0 for value in ratios.values()) or abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError("train_ratio must be positive; validation_ratio and test_ratio may be zero; all ratios must sum to 1.")
    ratios = {name: ratio for name, ratio in ratios.items() if ratio > 0}
    rng = random.Random(split_cfg["seed"])
    assignments = {name: [] for name in ratios}
    for items in buckets.values():
        rng.shuffle(items)
        total_rows = sum(len(group_rows) for _, group_rows in items)
        targets = {name: total_rows * ratio for name, ratio in ratios.items()}
        current = {name: 0 for name in ratios}
        for group, group_rows in items:
            split = max(ratios, key=lambda name: (targets[name] - current[name], -current[name], name))
            assignments[split].extend(group_rows)
            current[split] += len(group_rows)
    return assignments


def create_splits(config):
    data_cfg, split_cfg = config["data"], config["splitting"]
    required = {data_cfg["id_column"], data_cfg["domain_column"], data_cfg["source_column"], data_cfg["target_column"]}
    original = _load_records(split_cfg["input_dataset_path"], required)
    copy_train_only = split_cfg["validation_ratio"] == 0 and split_cfg["test_ratio"] == 0
    # A zero/zero split request deliberately means "copy input to train". Do not
    # de-duplicate or repartition it, so the output is byte-for-record equivalent.
    records = original if copy_train_only else (
        _deduplicate(
            original,
            data_cfg["source_column"],
            (
                data_cfg.get("source_lang_column", "source_lang_code"),
                data_cfg.get("target_lang_column", "target_lang_code"),
            ),
        ) if split_cfg["deduplicate_by_source"] else original
    )
    assignments = _assign_groups(records, config)
    output_dir = Path(split_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    filenames = {"train": split_cfg["train_filename"], "validation": split_cfg["validation_filename"], "test": split_cfg["test_filename"]}
    manifest = {
        "input_dataset_path": split_cfg["input_dataset_path"], "seed": split_cfg["seed"],
        "mode": "copy_train_only" if copy_train_only else "split",
        "input_rows": len(original), "deduplicated_rows": len(records), "duplicates_removed": len(original) - len(records),
        "splits": {},
    }
    for name, rows in assignments.items():
        path = output_dir / filenames[name]
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest["splits"][name] = {
            "path": str(path), "rows": len(rows),
            "documents": len({_group_id(row[data_cfg["id_column"]], split_cfg["group_id_delimiter"]) for row in rows}),
            "domains": dict(sorted(Counter(row[data_cfg["domain_column"]] for row in rows).items())),
        }
    manifest_path = output_dir / split_cfg["manifest_filename"]
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    manifest = create_splits(load_config(args.config))
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
