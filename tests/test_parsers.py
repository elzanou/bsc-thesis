import pytest

from music_evalkit.evaluation.parsers import (
    normalize_category,
    parse_open_ended_recover,
    parse_open_ended_strict,
)


@pytest.mark.parametrize("raw,expected", [
    # Already valid
    ("pitch", "pitch"),
    ("no_mistake", "no_mistake"),
    ("rhythm_and_timing", "rhythm_and_timing"),
    ("tempo", "tempo"),
    ("articulation", "articulation"),
    ("dynamics", "dynamics"),
    ("harmony", "harmony"),
    ("technique", "technique"),
    # Whitespace/case
    ("  Pitch ", "pitch"),
    ("NO_MISTAKE", "no_mistake"),
    ("RHYTHM_AND_TIMING", "rhythm_and_timing"),
    # Flamingo rhythm variants
    ("rhythm.and timing", "rhythm_and_timing"),
    ("rhythm.andtiming", "rhythm_and_timing"),
    ("rhythm.entiming", "rhythm_and_timing"),
    ("rhythm-and-timing", "rhythm_and_timing"),
    ("rhythm_andtiming", "rhythm_and_timing"),
    ("rhythm[outro]", "rhythm_and_timing"),
    ("rhythm", "rhythm_and_timing"),
    # no_mistake variants
    ("no_mmistake", "no_mistake"),
    ("no_mιστάκε", "no_mistake"),
    # Combined labels — NOT recovered, return None
    ("pitch, technique", None),
    ("pitch, technique, articulation", None),
    ("dynamics, articulation", None),
    ("pitch, rhythm_andtiming, tempo", None),
    # Unrecoverable
    ("something_random", None),
    ("", None),
    (None, None),
])
def test_normalize_category(raw, expected):
    assert normalize_category(raw) == expected


def test_recover_valid_json():
    resp = '{"reason": "test", "category": "pitch", "mistake": "wrong note", "feedback": "fix it"}'
    pred = parse_open_ended_recover(resp)
    assert pred.category == "pitch"
    assert pred.mistake == "wrong note"
    assert pred.parse_success is True


def test_recover_single_quotes():
    resp = "{ 'reason': 'test', 'category': 'pitch', 'mistake': 'wrong note', 'feedback': 'fix it'}"
    pred = parse_open_ended_recover(resp)
    assert pred.category == "pitch"
    assert pred.parse_success is True


def test_recover_bare_no_mistake():
    resp = "{no_mistake}"
    pred = parse_open_ended_recover(resp)
    assert pred.category == "no_mistake"
    assert pred.mistake is None
    assert pred.feedback is None
    assert pred.parse_success is True


def test_recover_bare_no_mistake_with_trailing_text():
    resp = "{no_mistake} The performance is correct, matching the reference perfectly."
    pred = parse_open_ended_recover(resp)
    assert pred.category == "no_mistake"
    assert pred.parse_success is True


def test_recover_wrong_key_dictionary():
    resp = '{ "reason": "test", "dictionary": "articulation", "mistake": "staccato", "feedback": "fix"}'
    pred = parse_open_ended_recover(resp)
    assert pred.category == "articulation"
    assert pred.parse_success is True


def test_recover_wrong_key_directory():
    resp = '{ "reason": "test", "directory": "no_mistake", "mistake": null, "feedback": null}'
    pred = parse_open_ended_recover(resp)
    assert pred.category == "no_mistake"
    assert pred.parse_success is True


def test_recover_malformed_category_normalised():
    resp = '{"reason": "test", "category": "rhythm.and timing", "mistake": "off beat", "feedback": "fix"}'
    pred = parse_open_ended_recover(resp)
    assert pred.category == "rhythm_and_timing"
    assert pred.parse_success is True


def test_recover_combined_category_not_recovered():
    resp = '{"reason": "test", "category": "pitch, technique", "mistake": "wrong", "feedback": "fix"}'
    pred = parse_open_ended_recover(resp)
    assert pred.category is None  # combined labels not recovered


def test_recover_no_mistake_key_without_quotes():
    resp = '{no_mistake: "The performance is correct."}'
    pred = parse_open_ended_recover(resp)
    assert pred.category == "no_mistake"
    assert pred.parse_success is True


def test_recover_unquoted_keys_single_quote_values():
    """Bare keys with single-quote values: {reason: 'text', category: 'tempo'}"""
    resp = "{reason: 'The tempo is inconsistent', category: 'tempo', mistake: 'speeding up', feedback: 'practice slowly'}"
    pred = parse_open_ended_recover(resp)
    assert pred.category == "tempo"
    assert pred.mistake == "speeding up"
    assert pred.parse_success is True


def test_recover_unquoted_keys_with_category_normalisation():
    """Bare keys + malformed category: {category: 'rhythm_and timing'}"""
    resp = "{reason: 'off beat', category: 'rhythm_and timing', mistake: 'off the beat', feedback: 'match the rhythm'}"
    pred = parse_open_ended_recover(resp)
    assert pred.category == "rhythm_and_timing"
    assert pred.parse_success is True


def test_recover_regex_fallback():
    """Malformed JSON that regex can still extract from."""
    resp = '{ "reason": "notes are wrong," "category": "pitch", "mistake": "wrong note", "feedback": "fix"}'
    pred = parse_open_ended_recover(resp)
    assert pred.category == "pitch"


def test_recover_total_garbage():
    resp = "{code}"
    pred = parse_open_ended_recover(resp)
    assert pred.parse_success is False


def test_strict_valid_json():
    resp = '{"reason": "test", "category": "pitch", "mistake": "wrong note", "feedback": "fix it"}'
    pred = parse_open_ended_strict(resp)
    assert pred.category == "pitch"
    assert pred.mistake == "wrong note"
    assert pred.parse_success is True


def test_strict_rejects_single_quotes():
    resp = "{ 'reason': 'test', 'category': 'pitch', 'mistake': 'wrong note', 'feedback': 'fix it'}"
    pred = parse_open_ended_strict(resp)
    assert pred.parse_success is False
    assert pred.category is None


def test_strict_rejects_malformed_json():
    resp = '{reason": "test", "category": "pitch", "mistake": "wrong", "feedback": "fix"}'
    pred = parse_open_ended_strict(resp)
    assert pred.parse_success is False
    assert pred.category is None


def test_strict_rejects_bare_no_mistake():
    resp = "{no_mistake}"
    pred = parse_open_ended_strict(resp)
    assert pred.parse_success is False


def test_strict_rejects_free_text():
    resp = "The category is pitch and the mistake is a wrong note"
    pred = parse_open_ended_strict(resp)
    assert pred.parse_success is False
    assert pred.category is None
    assert pred.mistake is None
    assert pred.feedback is None
