import json

import pandas as pd
import pytest
import yaml

from translation_benchmark.config import load_benchmark_config
from translation_benchmark.io import load_candidate_output, load_dataset
from translation_benchmark.metrics import pairwise_comparisons, preservation_score, NUMBER_RE
from translation_benchmark.pipeline import collect, report, score


def _write_fixture(tmp_path):
    dataset_path = tmp_path / "test.jsonl"
    rows = [
        {"id": "a:1", "source_text": "Use 12 kg of ATP.", "target_text": "از ۱۲ کیلوگرم ATP استفاده کنید.", "domain": "chemistry"},
        {"id": "b:1", "source_text": "The result is 5.", "target_text": "نتیجه ۵ است.", "domain": "physics"},
    ]
    dataset_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    pd.DataFrame({"row": ["a:1", "b:1"], "text": ["از ۱۲ کیلوگرم ATP استفاده کنید.", "نتیجه ۵ است."]}).to_csv(first, index=False)
    pd.DataFrame({"row": ["a:1", "b:1"], "text": ["از 10 کیلوگرم استفاده کنید.", "نتیجه نامشخص است."]}).to_csv(second, index=False)
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(yaml.safe_dump({
        "benchmark": {
            "title": "Test report", "output_dir": "out",
            "dataset": {"path": "test.jsonl", "columns": {
                "id": "id", "source": "source_text", "reference": "target_text", "domain": "domain"
            }},
        },
        "candidates": [
            {"id": "first", "label": "First", "type": "imported", "path": "first.csv", "columns": {"id": "row", "translation": "text"}},
            {"id": "second", "label": "Second", "type": "imported", "path": "second.csv", "columns": {"id": "row", "translation": "text"}},
        ],
        "metrics": {"transparent": {"enabled": True}},
        "statistics": {"bootstrap_samples": 100, "seed": 7},
        "report": {"slices": ["domain"]},
    }, sort_keys=False), encoding="utf-8")
    return config_path


def test_import_score_and_report_produce_aligned_human_review_artifacts(tmp_path):
    config = load_benchmark_config(_write_fixture(tmp_path))

    collected = collect(config)
    paths = score(config)
    html_path, markdown_path = report(config)

    assert len(collected) == 2
    assert paths["all_outputs"].exists()
    all_outputs = pd.read_csv(paths["all_outputs"])
    assert list(all_outputs["example_id"]) == ["a:1", "b:1"]
    assert {"translation__first", "translation__second"}.issubset(all_outputs.columns)
    summary = pd.read_csv(paths["summary"])
    first_chrf = summary.loc[summary["candidate_id"] == "first", "sentence_chrf"].iloc[0]
    second_chrf = summary.loc[summary["candidate_id"] == "second", "sentence_chrf"].iloc[0]
    assert first_chrf > second_chrf
    html = html_path.read_text(encoding="utf-8")
    assert "Human review explorer" in html
    assert "Translation — First" in html
    assert "Translation — Second" in html
    assert markdown_path.exists()


def test_import_rejects_missing_dataset_ids(tmp_path):
    config_path = _write_fixture(tmp_path)
    second = pd.read_csv(tmp_path / "second.csv").head(1)
    second.to_csv(tmp_path / "second.csv", index=False)
    config = load_benchmark_config(config_path)

    with pytest.raises(ValueError, match="ID mismatch"):
        collect(config, ["second"])


def test_candidate_output_loader_rejects_dataset_change(tmp_path):
    config = load_benchmark_config(_write_fixture(tmp_path))
    collect(config, ["first"])
    dataset, _ = load_dataset(config)
    dataset.loc[len(dataset)] = {"example_id": "new", "source": "x", "reference": "y", "domain": "other"}

    with pytest.raises(ValueError, match="ID mismatch"):
        load_candidate_output(config, "first", dataset)


def test_number_preservation_normalizes_persian_digits():
    score_value = preservation_score("Values are 12 and 5.", "مقادیر ۱۲ و ۵ هستند.", NUMBER_RE, lambda value: value.translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")))

    assert score_value == 1.0


def test_pairwise_comparison_respects_lower_is_better_direction():
    frame = pd.DataFrame({
        "candidate_id": ["a", "a", "b", "b"],
        "example_id": ["1", "2", "1", "2"],
        "metricx": [4.0, 6.0, 2.0, 5.0],
    })

    comparison = pairwise_comparisons(frame, samples=100, seed=3).iloc[0]

    assert comparison["delta_b_minus_a"] < 0
    assert comparison["b_win_rate"] == 1.0
