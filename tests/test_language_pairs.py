import pytest

from language_pairs import resolve_language_pair


def test_row_language_pair_overrides_legacy_config_defaults():
    data_cfg = {"source_lang": "en", "target_lang": "fa"}

    assert resolve_language_pair(
        {"source_lang_code": "ru", "target_lang_code": "fa"}, data_cfg
    ) == ("ru", "fa")


def test_old_rows_use_legacy_config_language_pair():
    assert resolve_language_pair(
        {}, {"source_lang": "en", "target_lang": "fa"}
    ) == ("en", "fa")


def test_custom_language_column_names_are_supported():
    data_cfg = {
        "source_lang": "en",
        "target_lang": "fa",
        "source_lang_column": "src",
        "target_lang_column": "tgt",
    }

    assert resolve_language_pair({"src": " ru ", "tgt": " de-DE "}, data_cfg) == (
        "ru",
        "de-DE",
    )


def test_missing_row_and_config_language_code_is_rejected():
    with pytest.raises(ValueError, match="source_lang_code"):
        resolve_language_pair({"target_lang_code": "fa"}, {})
