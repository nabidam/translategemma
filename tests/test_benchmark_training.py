import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "benchmark_training.py"
SPEC = importlib.util.spec_from_file_location("benchmark_training", MODULE_PATH)
benchmark_training = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(benchmark_training)

benchmark_config = benchmark_training.benchmark_config
build_benchmark_cases = benchmark_training.build_benchmark_cases
resolve_benchmark = benchmark_training.resolve_benchmark


def cli_args(**overrides):
    values = {
        "benchmark_types": None,
        "model_sizes": None,
        "gpu_counts": None,
        "batch_sizes": None,
        "batch_gpu_count": None,
        "training_options_gpu_count": None,
        "max_examples": None,
        "max_steps": None,
        "per_device_batch_size": None,
        "effective_batch_size": None,
        "output_dir": None,
        "accelerate_config_pattern": None,
        "run_evaluation": None,
        "devices": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.fixture
def training_config():
    return {
        "model": {"base_model_id": "configured-model", "output_dir": "output"},
        "data": {"prepared_cache_dir": "cache"},
        "training": {
            "run_sft": True,
            "gradient_checkpointing": True,
            "packing": True,
        },
        "evaluation": {"run_after_training": True},
    }


def test_predefined_model_profiles_use_documented_h200f_batch_math(training_config):
    training_config["benchmark"] = {
        "types": ["gpu_count"],
        "model_sizes": ["4b", "12b", "27b"],
        "gpu_counts": [1, 2, 4, 8],
    }

    settings = resolve_benchmark(training_config, cli_args())
    cases = build_benchmark_cases(training_config, settings)

    expected = {
        "4b": (12, 96, [8, 4, 2, 1]),
        "12b": (6, 48, [8, 4, 2, 1]),
        "27b": (2, 16, [8, 4, 2, 1]),
    }
    for model_size, (micro_batch, global_batch, accumulations) in expected.items():
        model_cases = [case for case in cases if case["model_size"] == model_size]
        assert [case["per_device_batch_size"] for case in model_cases] == [
            micro_batch
        ] * 4
        assert [case["effective_batch_size"] for case in model_cases] == [
            global_batch
        ] * 4
        assert [
            case["gradient_accumulation_steps"] for case in model_cases
        ] == accumulations


def test_batch_size_benchmark_keeps_effective_global_batch_constant(training_config):
    training_config["benchmark"] = {
        "types": ["batch_size"],
        "model_sizes": ["12b"],
        "batch_gpu_count": 2,
        "model_profiles": {"12b": {"batch_sizes": [1, 2, 3, 6]}},
    }

    settings = resolve_benchmark(training_config, cli_args())
    cases = build_benchmark_cases(training_config, settings)

    assert [case["gradient_accumulation_steps"] for case in cases] == [24, 12, 8, 4]
    assert {
        case["per_device_batch_size"]
        * case["gpu_count"]
        * case["gradient_accumulation_steps"]
        for case in cases
    } == {48}


def test_training_options_build_full_checkpointing_packing_factorial(
    training_config, tmp_path
):
    training_config["benchmark"] = {
        "types": ["training_options"],
        "model_sizes": ["4b"],
    }

    settings = resolve_benchmark(training_config, cli_args())
    cases = build_benchmark_cases(training_config, settings)

    assert [(case["gradient_checkpointing"], case["packing"]) for case in cases] == [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ]
    prepared = benchmark_config(training_config, tmp_path, cases[-1])
    assert prepared["model"]["base_model_id"] == "google/translategemma-4b-it"
    assert prepared["training"]["gradient_checkpointing"] is False
    assert prepared["training"]["packing"] is False


def test_non_divisible_batch_matrix_fails_before_launch(training_config):
    training_config["benchmark"] = {
        "types": ["batch_size"],
        "model_sizes": ["12b"],
        "batch_gpu_count": 2,
        "model_profiles": {"12b": {"batch_sizes": [5]}},
    }

    settings = resolve_benchmark(training_config, cli_args())

    with pytest.raises(ValueError, match="must be exactly divisible"):
        build_benchmark_cases(training_config, settings)
