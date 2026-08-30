import json
import pytest
from unittest.mock import patch

from music_evalkit.evaluation.llm_judge import (
    JudgeMetrics,
    JudgeResult,
    JudgeSource,
    LLMJudge,
    compute_judge_metrics,
)
from music_evalkit.evaluation.llm_judge.helpers import (
    _no_mistake_result,
    _missed_result,
    _hallucinated_result,
)


class TestJudgeResult:
    def test_valid_construction(self):
        r = JudgeResult(
            mistake="partially_correct",
            mistake_reasoning="Close",
            feedback="helpful",
            feedback_reasoning="Useful",
            raw_response="{}",
            source="llm",
        )
        assert r.mistake == "partially_correct"
        assert r.feedback == "helpful"

    def test_none_categories(self):
        r = JudgeResult(
            mistake=None,
            mistake_reasoning="Parse failed",
            feedback=None,
            feedback_reasoning="Parse failed",
            raw_response="garbage",
            source="llm",
        )
        assert r.mistake is None
        assert r.feedback is None

    def test_frozen(self):
        r = _no_mistake_result()
        with pytest.raises(Exception):
            r.mistake = "correct"

    def test_model_dump_roundtrip(self):
        r = JudgeResult(
            mistake="incorrect",
            mistake_reasoning="Bad",
            feedback="unhelpful",
            feedback_reasoning="Useless",
            raw_response='{"test": true}',
            source="llm",
        )
        d = r.model_dump()
        r2 = JudgeResult(**d)
        assert r == r2

    def test_invalid_category_rejected(self):
        with pytest.raises(Exception):
            JudgeResult(
                mistake="excellent",  # not a valid category
                mistake_reasoning="",
                feedback="helpful",
                feedback_reasoning="",
                raw_response="",
                source="llm",
            )


class TestDetectionOutcomes:
    def test_no_mistake(self):
        r = _no_mistake_result()
        assert r.source == JudgeSource.NO_MISTAKE
        assert r.mistake is None
        assert r.feedback is None

    def test_missed(self):
        r = _missed_result()
        assert r.source == JudgeSource.MISSED
        assert "no mistake text" in r.mistake_reasoning

    def test_hallucinated(self):
        r = _hallucinated_result()
        assert r.source == JudgeSource.HALLUCINATED
        assert "No GT mistake" in r.mistake_reasoning


class TestParseResponse:
    @pytest.fixture
    def judge(self):
        with patch.object(LLMJudge, "__init__", lambda self, **kw: None):
            j = LLMJudge.__new__(LLMJudge)
            j._provider = "ollama"
            j._model = "test"
            j._base_url = "http://localhost"
            j._api_key = "test"
            j._client = None
            return j

    def test_valid_json(self, judge):
        raw = json.dumps({
            "mistake_reasoning": "Right mistake",
            "mistake": "correct",
            "feedback_reasoning": "Useful advice",
            "feedback": "helpful",
        })
        r = judge._parse_response(raw)
        assert r.mistake == "correct"
        assert r.feedback == "helpful"
        assert r.source == JudgeSource.LLM

    def test_malformed_json_returns_none(self, judge):
        r = judge._parse_response("this is not json")
        assert r.mistake is None
        assert r.feedback is None
        assert r.source == JudgeSource.LLM

    def test_markdown_wrapped_json(self, judge):
        raw = '```json\n{"mistake_reasoning": "ok", "mistake": "partially_correct", "feedback_reasoning": "ok", "feedback": "generic"}\n```'
        r = judge._parse_response(raw)
        assert r.mistake == "partially_correct"
        assert r.feedback == "generic"

    def test_invalid_category_returns_none(self, judge):
        raw = json.dumps({
            "mistake_reasoning": "ok",
            "mistake": "excellent",  # invalid
            "feedback_reasoning": "ok",
            "feedback": "helpful",  # valid
        })
        r = judge._parse_response(raw)
        assert r.mistake is None  # invalid → None
        assert r.feedback == "helpful"  # valid → kept

    def test_missing_field_does_not_poison_other(self, judge):
        raw = json.dumps({
            "feedback_reasoning": "ok",
            "feedback": "helpful",
            # mistake entirely missing
        })
        r = judge._parse_response(raw)
        assert r.mistake is None
        assert r.feedback == "helpful"

    def test_empty_response(self, judge):
        r = judge._parse_response("")
        assert r.mistake is None
        assert r.feedback is None

    def test_case_insensitive(self, judge):
        raw = json.dumps({
            "mistake_reasoning": "ok",
            "mistake": "PARTIALLY_CORRECT",
            "feedback_reasoning": "ok",
            "feedback": "Helpful",
        })
        r = judge._parse_response(raw)
        assert r.mistake == "partially_correct"
        assert r.feedback == "helpful"


class TestComputeJudgeMetrics:
    def test_empty_results(self):
        m = compute_judge_metrics([])
        assert m.total == 0
        assert m.judged == 0

    def test_all_detection_outcomes(self):
        results = [
            {"source": "no_mistake", "mistake": None, "feedback": None},
            {"source": "missed", "mistake": None, "feedback": None},
            {"source": "hallucinated", "mistake": None, "feedback": None},
        ]
        m = compute_judge_metrics(results)
        assert m.total == 3
        assert m.no_mistake == 1
        assert m.missed == 1
        assert m.hallucinated == 1
        assert m.llm_evaluated == 0
        assert m.judged == 0
        # Routing rates
        assert abs(m.no_mistake_rate - 1 / 3) < 1e-9
        assert abs(m.missed_rate - 1 / 3) < 1e-9
        assert abs(m.hallucinated_rate - 1 / 3) < 1e-9
        assert m.llm_evaluated_rate == 0.0

    def test_llm_judged_samples(self):
        results = [
            {"source": "llm", "mistake": "correct", "feedback": "helpful"},
            {"source": "llm", "mistake": "partially_correct", "feedback": "generic"},
            {"source": "llm", "mistake": "incorrect", "feedback": "unhelpful"},
        ]
        m = compute_judge_metrics(results)
        assert m.total == 3
        assert m.llm_evaluated == 3
        assert m.judged == 3
        assert m.parse_errors == 0
        assert m.mistake_distribution == {"correct": 1, "partially_correct": 1, "incorrect": 1}
        assert m.feedback_distribution == {"helpful": 1, "generic": 1, "unhelpful": 1}
        # Quality rates (each 1/3)
        for cat in ["correct", "partially_correct", "incorrect"]:
            assert abs(m.mistake_rates[cat] - 1 / 3) < 1e-9
        for cat in ["helpful", "generic", "unhelpful"]:
            assert abs(m.feedback_rates[cat] - 1 / 3) < 1e-9
        # Routing rate
        assert m.llm_evaluated_rate == 1.0

    def test_parse_errors(self):
        results = [
            {"source": "llm", "mistake": "correct", "feedback": "helpful"},
            {"source": "llm", "mistake": None, "feedback": None},
        ]
        m = compute_judge_metrics(results)
        assert m.llm_evaluated == 2
        assert m.judged == 1
        assert m.parse_errors == 1
        assert m.parse_error_rate == 0.5

    def test_mixed_detection_and_quality(self):
        results = [
            {"source": "no_mistake", "mistake": None, "feedback": None},
            {"source": "missed", "mistake": None, "feedback": None},
            {"source": "llm", "mistake": "correct", "feedback": "helpful"},
            {"source": "llm", "mistake": "incorrect", "feedback": "unhelpful"},
        ]
        m = compute_judge_metrics(results)
        assert m.total == 4
        assert m.no_mistake == 1
        assert m.missed == 1
        assert m.llm_evaluated == 2
        assert m.judged == 2
        assert m.mistake_distribution["correct"] == 1
        assert m.mistake_distribution["incorrect"] == 1
        # Routing rates
        assert m.no_mistake_rate == 0.25
        assert m.missed_rate == 0.25
        assert m.llm_evaluated_rate == 0.5
        # Quality rates
        assert m.mistake_rates["correct"] == 0.5
        assert m.mistake_rates["incorrect"] == 0.5
