"""Language-pair resolution shared by training, analysis, and evaluation."""

DEFAULT_SOURCE_LANG_COLUMN = "source_lang_code"
DEFAULT_TARGET_LANG_COLUMN = "target_lang_code"


def _language_code(value):
    """Return a normalized non-blank code, or None for missing dataframe values."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def resolve_language_pair(example, data_cfg):
    """Resolve row language codes, falling back to the legacy config-level pair."""
    source_column = data_cfg.get("source_lang_column", DEFAULT_SOURCE_LANG_COLUMN)
    target_column = data_cfg.get("target_lang_column", DEFAULT_TARGET_LANG_COLUMN)
    source_lang = _language_code(example.get(source_column)) or _language_code(
        data_cfg.get("source_lang")
    )
    target_lang = _language_code(example.get(target_column)) or _language_code(
        data_cfg.get("target_lang")
    )
    if source_lang is None:
        raise ValueError(
            f"Each row needs a non-blank {source_column!r}, or data.source_lang "
            "must provide a fallback."
        )
    if target_lang is None:
        raise ValueError(
            f"Each row needs a non-blank {target_column!r}, or data.target_lang "
            "must provide a fallback."
        )
    return source_lang, target_lang
