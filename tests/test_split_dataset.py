import json

from split_dataset import create_splits


def test_create_splits_deduplicates_and_keeps_documents_isolated(tmp_path):
    source = tmp_path / "source.jsonl"
    rows = [
        {"id": "a:1", "domain": "physics", "english": "one", "farsi": "یک"},
        {"id": "a:2", "domain": "physics", "english": "two", "farsi": "دو"},
        {"id": "b:1", "domain": "physics", "english": "three", "farsi": "سه"},
        {"id": "c:1", "domain": "materials", "english": "four", "farsi": "چهار"},
        {"id": "d:1", "domain": "materials", "english": "five", "farsi": "پنج"},
        {"id": "e:1", "domain": "materials", "english": "one", "farsi": "یک تکراری"},
    ]
    source.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    config = {
        "data": {"id_column": "id", "domain_column": "domain", "source_column": "english", "target_column": "farsi"},
        "splitting": {
            "input_dataset_path": str(source), "output_dir": str(tmp_path / "splits"),
            "train_filename": "train.jsonl", "validation_filename": "validation.jsonl", "test_filename": "test.jsonl",
            "group_id_delimiter": ":", "train_ratio": 0.6, "validation_ratio": 0.2, "test_ratio": 0.2,
            "seed": 9, "deduplicate_by_source": True, "stratify_by_domain": True, "manifest_filename": "manifest.json",
        },
    }

    manifest = create_splits(config)

    assert manifest["duplicates_removed"] == 1
    document_sets = []
    for split in manifest["splits"].values():
        output_rows = [json.loads(line) for line in open(split["path"], encoding="utf-8")]
        document_sets.append({row["id"].split(":")[0] for row in output_rows})
    assert not (document_sets[0] & document_sets[1])
    assert not (document_sets[0] & document_sets[2])
    assert not (document_sets[1] & document_sets[2])


def test_create_splits_with_zero_validation_and_test_copies_input_to_train(tmp_path):
    source = tmp_path / "source.jsonl"
    rows = [
        {"id": "a:1", "domain": "physics", "english": "one", "farsi": "یک"},
        {"id": "a:2", "domain": "physics", "english": "one", "farsi": "یک تکراری"},
    ]
    source.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    config = {
        "data": {"id_column": "id", "domain_column": "domain", "source_column": "english", "target_column": "farsi"},
        "splitting": {
            "input_dataset_path": str(source), "output_dir": str(tmp_path / "splits"),
            "train_filename": "train.jsonl", "validation_filename": "validation.jsonl", "test_filename": "test.jsonl",
            "group_id_delimiter": ":", "train_ratio": 1.0, "validation_ratio": 0.0, "test_ratio": 0.0,
            "seed": 9, "deduplicate_by_source": True, "stratify_by_domain": True, "manifest_filename": "manifest.json",
        },
    }

    manifest = create_splits(config)

    assert manifest["mode"] == "copy_train_only"
    assert manifest["duplicates_removed"] == 0
    assert set(manifest["splits"]) == {"train"}
    assert (tmp_path / "splits" / "train.jsonl").read_text(encoding="utf-8") == source.read_text(encoding="utf-8")
    assert not (tmp_path / "splits" / "validation.jsonl").exists()
    assert not (tmp_path / "splits" / "test.jsonl").exists()


def test_create_splits_does_not_deduplicate_across_language_pairs(tmp_path):
    source = tmp_path / "source.jsonl"
    rows = [
        {"id": "a:1", "domain": "science", "source": "radio", "target": "رادیو", "src_lang": "en", "tgt_lang": "fa"},
        {"id": "b:1", "domain": "science", "source": "radio", "target": "رادیو", "src_lang": "ru", "tgt_lang": "fa"},
    ]
    source.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    config = {
        "data": {"id_column": "id", "domain_column": "domain", "source_column": "source", "target_column": "target"},
        "splitting": {
            "input_dataset_path": str(source), "output_dir": str(tmp_path / "splits"),
            "train_filename": "train.jsonl", "validation_filename": "validation.jsonl", "test_filename": "test.jsonl",
            "group_id_delimiter": ":", "train_ratio": 0.5, "validation_ratio": 0.5, "test_ratio": 0.0,
            "seed": 9, "deduplicate_by_source": True, "stratify_by_domain": False, "manifest_filename": "manifest.json",
        },
    }

    manifest = create_splits(config)

    assert manifest["duplicates_removed"] == 0
    assert sum(split["rows"] for split in manifest["splits"].values()) == 2
