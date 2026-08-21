"""Folding rules for matching. Offsets must survive, or rewriting slices wrong."""

from glossary.normalize import fold_source, fold_target


def test_target_folds_arabic_yeh_and_kaf_to_persian():
    folded, _ = fold_target("يك")  # Arabic yeh, Arabic kaf
    assert folded == "یک"  # Persian yeh, Persian keheh


def test_target_drops_zwnj_but_keeps_offsets_pointing_at_the_original():
    original = "می‌رود"  # mi-ravad, with ZWNJ
    folded, offsets = fold_target(original)
    assert "‌" not in folded
    assert len(folded) == len(offsets)
    # Every folded character still maps back to the character it came from.
    assert all(original[offset] != "‌" for offset in offsets)
    # The character after the dropped ZWNJ maps past it, not onto it.
    assert offsets[2] == 3


def test_target_folds_arabic_indic_digits():
    folded, _ = fold_target("۱٢")  # extended Arabic-Indic 1, Arabic-Indic 2
    assert folded == "12"


def test_source_lowercases_when_not_case_sensitive():
    folded, offsets = fold_source("Genome", case_sensitive=False)
    assert folded == "genome"
    assert offsets == list(range(6))


def test_source_preserves_case_when_case_sensitive():
    folded, _ = fold_source("Genome", case_sensitive=True)
    assert folded == "Genome"


def test_offsets_allow_slicing_the_original_by_a_folded_match():
    original = "The يGenome‌ sequence"
    folded, offsets = fold_source(original, case_sensitive=False)
    start = folded.index("genome")
    end = start + len("genome")
    assert original[offsets[start] : offsets[end - 1] + 1] == "Genome"


def test_offsets_align_when_lowercasing_expands():
    """Test that expanding characters (e.g., Turkish İ → i̇) preserve offset invariant.

    Turkish capital dotted I (U+0130 İ) lowercases to i (U+0069) + combining dot
    above (U+0307), expanding from 1 character to 2. The offsets list must have
    one entry per output character, not per source character, so that slicing
    the original string by folded match indices still works.
    """
    original = "İstanbul"
    folded, offsets = fold_source(original, case_sensitive=False)
    # "İstanbul".lower() → "i̇stanbul" (9 chars: i + combining-dot + s,t,a,n,b,u,l)
    assert len(folded) == 9
    assert len(folded) == len(offsets)
    # Verify slicing: search for substring in folded, recover original slice
    start = folded.index("stanbul")
    end = start + len("stanbul")
    assert original[offsets[start] : offsets[end - 1] + 1] == "stanbul"
