"""
Tests for Node 13: Beginner Explanation (LLM)

Strategy: the Groq LLM call is mocked in every test so these run instantly
and deterministically without a real API key.

Covers:
- _score_to_label: correct plain-English translation for all 4 streams × 5 buckets
- _build_user_prompt: structure, conditional sections (pattern, articles, risk warnings)
- _build_fallback_explanation: None sc vs valid sc
- beginner_explanation_node (main):
    - happy path with mocked LLM → explanation written to state
    - signal_components=None → fallback, error logged
    - LLM raises → fallback, error appended, no crash
    - trading_safe=False → risk warning injected in prompt
    - pattern_prediction sufficient_data=False → pattern block omitted
    - pattern_prediction agreement=disagree → caution injected
    - pump_and_dump_score > 50 → pump warning injected
    - streams_missing non-empty → note injected
    - weights_are_fallback=True → note injected
    - sentiment_breakdown=None → top_articles section absent
    - node_execution_times populated
    - errors list not mutated on happy path
"""

import pytest
from copy import deepcopy
from unittest.mock import MagicMock, patch

from src.langgraph_nodes.node_13_beginner_explanation import (
    DISCLAIMER,
    SYSTEM_PROMPT,
    _build_fallback_explanation,
    _build_user_prompt,
    _score_to_label,
    beginner_explanation_node,
)


# ============================================================================
# FIXTURES
# ============================================================================

def _make_sc(
    signal: str = "BUY",
    confidence: float = 0.72,
    agreement: int = 3,
    streams_missing: list = None,
    trading_safe: bool = True,
    risk_level: str = "LOW",
    pump_score: int = 12,
    sufficient_data: bool = True,
    job2_agreement: str = "mild",
    weights_are_fallback: bool = False,
    tech_score: float = 0.65,
    sent_score: float = 0.48,
    mkt_score: float = 0.30,
    mc_score: float = 0.36,
) -> dict:
    """Build a realistic signal_components dict."""
    return {
        "final_signal": signal,
        "final_confidence": confidence,
        "final_score": 0.42,
        "signal_agreement": agreement,
        "streams_missing": streams_missing or [],
        "stream_scores": {
            "technical":   {"raw_score": tech_score, "weight": 0.25, "contribution": tech_score * 0.25},
            "sentiment":   {"raw_score": sent_score, "weight": 0.25, "contribution": sent_score * 0.25},
            "market":      {"raw_score": mkt_score,  "weight": 0.25, "contribution": mkt_score  * 0.25},
            "monte_carlo": {"raw_score": mc_score,   "weight": 0.25, "contribution": mc_score   * 0.25},
        },
        "price_targets": {
            "current_price": 185.50,
            "forecasted_price": 192.00,
            "price_range_lower": 175.00,
            "price_range_upper": 205.00,
            "expected_return_pct": 3.5,
        },
        "risk_summary": {
            "overall_risk_level": risk_level,
            "trading_safe": trading_safe,
            "pump_and_dump_score": pump_score,
            "behavioral_summary": "No unusual behavioral patterns detected.",
            "trading_recommendation": "NORMAL",
            "alerts": ["Volume above average"] if not trading_safe else [],
            "primary_risk_factors": [],
            "detection_breakdown": {},
        },
        "pattern_prediction": {
            "sufficient_data": sufficient_data,
            "similar_days_found": 14,
            "prob_up": 0.71,
            "prob_down": 0.29,
            "expected_return_7d": 1.8,
            "worst_case_7d": -0.6,
            "best_case_7d": 4.2,
            "median_return_7d": 1.5,
            "agreement_with_job1": job2_agreement,
            "confidence_multiplier": 1.05,
            "reason": "Only 3 snapshots available",
        },
        "backtest_context": {
            "hold_threshold_pct": 58.0,
            "streams_reliable": 3,
            "weights_are_fallback": weights_are_fallback,
        },
        "prediction_graph_data": {"data_source": "blended"},
    }


def _make_sb(include_articles: bool = True) -> dict:
    articles = [
        {"title": "AAPL beats earnings", "source": "Reuters", "sentiment_score": 0.72, "credibility": 0.85},
        {"title": "iPhone demand strong", "source": "Bloomberg", "sentiment_score": 0.55, "credibility": 0.90},
    ] if include_articles else []
    return {
        "top_articles": articles,
        "stream_breakdown": {
            "stock_news": {"sentiment_score": 0.45, "article_count": 8, "label": "positive"},
            "market_news": {"sentiment_score": 0.22, "article_count": 5, "label": "neutral"},
            "related_company_news": {"sentiment_score": 0.18, "article_count": 3, "label": "neutral"},
        },
        "overall_sentiment_label": "positive",
    }


def _make_state(sc=None, sb=None, extra=None) -> dict:
    state = {
        "ticker": "AAPL",
        "signal_components": sc if sc is not None else _make_sc(),
        "sentiment_breakdown": sb if sb is not None else _make_sb(),
        "errors": [],
        "node_execution_times": {},
    }
    if extra:
        state.update(extra)
    return state


def _mock_groq_response(text: str = "Mocked LLM explanation.") -> MagicMock:
    msg = MagicMock()
    msg.content = text
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


# ============================================================================
# _score_to_label
# ============================================================================

class TestScoreToLabel:
    @pytest.mark.parametrize("stream", ["technical", "sentiment", "market", "monte_carlo"])
    def test_strong_positive(self, stream):
        label = _score_to_label(0.6, stream)
        assert any(w in label.lower() for w in ("strongly", "strong", "clearly"))

    @pytest.mark.parametrize("stream", ["technical", "sentiment", "market", "monte_carlo"])
    def test_mild_positive(self, stream):
        label = _score_to_label(0.2, stream)
        assert "mild" in label.lower() or "slightly" in label.lower() or "lean" in label.lower()

    @pytest.mark.parametrize("stream", ["technical", "sentiment", "market", "monte_carlo"])
    def test_neutral(self, stream):
        label = _score_to_label(0.0, stream)
        assert "neutral" in label.lower() or "mixed" in label.lower() or "no clear" in label.lower()

    @pytest.mark.parametrize("stream", ["technical", "sentiment", "market", "monte_carlo"])
    def test_mild_negative(self, stream):
        label = _score_to_label(-0.2, stream)
        assert any(w in label.lower() for w in ["mild", "slight", "lean", "unfav"])

    @pytest.mark.parametrize("stream", ["technical", "sentiment", "market", "monte_carlo"])
    def test_strong_negative(self, stream):
        label = _score_to_label(-0.7, stream)
        assert any(w in label.lower() for w in ("strong", "clearly"))

    def test_returns_string(self):
        for stream in ("technical", "sentiment", "market", "monte_carlo"):
            assert isinstance(_score_to_label(0.0, stream), str)

    def test_unknown_stream_returns_string(self):
        result = _score_to_label(0.5, "unknown_stream")
        assert isinstance(result, str)


# ============================================================================
# _build_user_prompt
# ============================================================================

class TestBuildUserPrompt:
    def test_contains_ticker(self):
        prompt = _build_user_prompt(_make_sc(), _make_sb(), "NVDA")
        assert "NVDA" in prompt

    def test_contains_signal(self):
        prompt = _build_user_prompt(_make_sc(signal="SELL"), _make_sb(), "AAPL")
        assert "SELL" in prompt

    def test_contains_confidence(self):
        prompt = _build_user_prompt(_make_sc(confidence=0.65), _make_sb(), "AAPL")
        assert "65%" in prompt

    def test_contains_stream_count(self):
        prompt = _build_user_prompt(_make_sc(agreement=2), _make_sb(), "AAPL")
        assert "2 out of 4" in prompt

    def test_pattern_block_when_sufficient(self):
        prompt = _build_user_prompt(_make_sc(sufficient_data=True), _make_sb(), "AAPL")
        assert "HISTORICAL PATTERN DATA" in prompt
        assert "14" in prompt          # similar_days_found
        assert "71%" in prompt         # prob_up

    def test_pattern_block_insufficient(self):
        prompt = _build_user_prompt(_make_sc(sufficient_data=False), _make_sb(), "AAPL")
        assert "Insufficient data" in prompt
        assert "omit" in prompt.lower()

    def test_disagree_caution_injected(self):
        prompt = _build_user_prompt(_make_sc(job2_agreement="disagree"), _make_sb(), "AAPL")
        assert "CAUTION" in prompt

    def test_disagree_caution_absent_when_mild(self):
        prompt = _build_user_prompt(_make_sc(job2_agreement="mild"), _make_sb(), "AAPL")
        assert "CAUTION" not in prompt

    def test_top_articles_present(self):
        prompt = _build_user_prompt(_make_sc(), _make_sb(include_articles=True), "AAPL")
        assert "AAPL beats earnings" in prompt or "Reuters" in prompt

    def test_top_articles_absent_when_no_sb(self):
        prompt = _build_user_prompt(_make_sc(), {}, "AAPL")
        assert "TOP NEWS HEADLINES" not in prompt

    def test_trading_safe_false_warning(self):
        prompt = _build_user_prompt(_make_sc(trading_safe=False), _make_sb(), "AAPL")
        assert "WARNING" in prompt and "trading_safe=False" in prompt

    def test_pump_score_high_warning(self):
        prompt = _build_user_prompt(_make_sc(pump_score=75), _make_sb(), "AAPL")
        assert "pump_and_dump_score=75" in prompt
        assert "WARNING" in prompt

    def test_pump_score_low_no_warning(self):
        prompt = _build_user_prompt(_make_sc(pump_score=20), _make_sb(), "AAPL")
        assert "pump_and_dump_score=75" not in prompt

    def test_streams_missing_note(self):
        prompt = _build_user_prompt(_make_sc(streams_missing=["monte_carlo"]), _make_sb(), "AAPL")
        assert "monte_carlo" in prompt
        assert "NOTE" in prompt

    def test_weights_fallback_note(self):
        prompt = _build_user_prompt(_make_sc(weights_are_fallback=True), _make_sb(), "AAPL")
        assert "NOTE" in prompt
        assert "historical learning" in prompt.lower()

    def test_disclaimer_present(self):
        prompt = _build_user_prompt(_make_sc(), _make_sb(), "AAPL")
        assert DISCLAIMER in prompt

    def test_section_structure_listed(self):
        prompt = _build_user_prompt(_make_sc(), _make_sb(), "AAPL")
        for section in ("Headline", "Signal Strength", "Historical Pattern", "Risk", "Disclaimer"):
            assert section in prompt

    def test_price_targets_present(self):
        prompt = _build_user_prompt(_make_sc(), _make_sb(), "AAPL")
        assert "185.50" in prompt
        assert "192.00" in prompt


# ============================================================================
# _build_fallback_explanation
# ============================================================================

class TestBuildFallbackExplanation:
    def test_none_sc_returns_unavailable(self):
        result = _build_fallback_explanation(None, "AAPL")
        assert "Unavailable" in result or "unavailable" in result.lower()
        assert "AAPL" in result
        assert DISCLAIMER in result

    def test_valid_sc_returns_signal(self):
        result = _build_fallback_explanation(_make_sc(signal="SELL", confidence=0.60), "MSFT")
        assert "SELL" in result
        assert "MSFT" in result
        assert DISCLAIMER in result

    def test_returns_string(self):
        assert isinstance(_build_fallback_explanation(None, "X"), str)
        assert isinstance(_build_fallback_explanation(_make_sc(), "X"), str)


# ============================================================================
# beginner_explanation_node — main node
# ============================================================================

class TestBeginnerExplanationNode:

    @patch("src.langgraph_nodes.node_13_beginner_explanation.Groq")
    def test_happy_path_writes_explanation(self, mock_groq_cls):
        mock_groq_cls.return_value.chat.completions.create.return_value = (
            _mock_groq_response("Great explanation text.")
        )
        state = _make_state()
        result = beginner_explanation_node(state)
        assert result["beginner_explanation"] == "Great explanation text."
        assert result["errors"] == []

    @patch("src.langgraph_nodes.node_13_beginner_explanation.Groq")
    def test_execution_time_recorded(self, mock_groq_cls):
        mock_groq_cls.return_value.chat.completions.create.return_value = (
            _mock_groq_response()
        )
        state = _make_state()
        result = beginner_explanation_node(state)
        assert "node_13" in result["node_execution_times"]
        assert result["node_execution_times"]["node_13"] >= 0.0

    def test_signal_components_none_returns_fallback(self):
        state = _make_state()
        state["signal_components"] = None
        result = beginner_explanation_node(state)
        assert "beginner_explanation" in result
        assert len(result["beginner_explanation"]) > 0
        assert len(result["errors"]) == 1
        assert "signal_components" in result["errors"][0]

    @patch("src.langgraph_nodes.node_13_beginner_explanation.Groq")
    def test_llm_failure_returns_fallback_no_crash(self, mock_groq_cls):
        mock_groq_cls.return_value.chat.completions.create.side_effect = RuntimeError("API down")
        state = _make_state()
        result = beginner_explanation_node(state)
        assert "beginner_explanation" in result
        assert len(result["beginner_explanation"]) > 0
        assert any("LLM failed" in e for e in result["errors"])

    @patch("src.langgraph_nodes.node_13_beginner_explanation.Groq")
    def test_no_groq_key_returns_fallback(self, mock_groq_cls):
        """Missing API key should trigger fallback, not crash."""
        mock_groq_cls.return_value.chat.completions.create.side_effect = ValueError("GROQ_API_KEY is not set")
        state = _make_state()
        result = beginner_explanation_node(state)
        assert "beginner_explanation" in result
        assert len(result["errors"]) >= 1

    @patch("src.langgraph_nodes.node_13_beginner_explanation.Groq")
    def test_sentiment_breakdown_none_no_crash(self, mock_groq_cls):
        mock_groq_cls.return_value.chat.completions.create.return_value = (
            _mock_groq_response("OK")
        )
        state = _make_state(sb={})
        state["sentiment_breakdown"] = None
        result = beginner_explanation_node(state)
        assert result["beginner_explanation"] == "OK"
        assert result["errors"] == []

    @patch("src.langgraph_nodes.node_13_beginner_explanation.Groq")
    def test_returns_state_dict(self, mock_groq_cls):
        mock_groq_cls.return_value.chat.completions.create.return_value = (
            _mock_groq_response()
        )
        state = _make_state()
        result = beginner_explanation_node(state)
        assert isinstance(result, dict)
        assert "ticker" in result

    @patch("src.langgraph_nodes.node_13_beginner_explanation.Groq")
    def test_strips_whitespace_from_llm_output(self, mock_groq_cls):
        mock_groq_cls.return_value.chat.completions.create.return_value = (
            _mock_groq_response("  \n  Trimmed output.  \n  ")
        )
        state = _make_state()
        result = beginner_explanation_node(state)
        assert result["beginner_explanation"] == "Trimmed output."

    @patch("src.langgraph_nodes.node_13_beginner_explanation.Groq")
    def test_missing_streams_does_not_crash(self, mock_groq_cls):
        mock_groq_cls.return_value.chat.completions.create.return_value = (
            _mock_groq_response("OK")
        )
        sc = _make_sc(streams_missing=["technical", "market"])
        state = _make_state(sc=sc)
        result = beginner_explanation_node(state)
        assert result["beginner_explanation"] == "OK"

    @patch("src.langgraph_nodes.node_13_beginner_explanation.Groq")
    def test_all_streams_zero_score_no_crash(self, mock_groq_cls):
        mock_groq_cls.return_value.chat.completions.create.return_value = (
            _mock_groq_response("OK")
        )
        sc = _make_sc(tech_score=0.0, sent_score=0.0, mkt_score=0.0, mc_score=0.0)
        state = _make_state(sc=sc)
        result = beginner_explanation_node(state)
        assert result["beginner_explanation"] == "OK"

    @patch("src.langgraph_nodes.node_13_beginner_explanation.Groq")
    def test_critical_risk_no_crash(self, mock_groq_cls):
        mock_groq_cls.return_value.chat.completions.create.return_value = (
            _mock_groq_response("OK")
        )
        sc = _make_sc(trading_safe=False, risk_level="CRITICAL", pump_score=88)
        state = _make_state(sc=sc)
        result = beginner_explanation_node(state)
        assert result["beginner_explanation"] == "OK"

    @patch("src.langgraph_nodes.node_13_beginner_explanation.Groq")
    def test_system_prompt_is_passed(self, mock_groq_cls):
        """Verify the system prompt constant is sent as the first message."""
        mock_instance = mock_groq_cls.return_value
        mock_instance.chat.completions.create.return_value = _mock_groq_response()
        state = _make_state()
        beginner_explanation_node(state)
        call_kwargs = mock_instance.chat.completions.create.call_args
        messages = call_kwargs[1]["messages"] if call_kwargs[1] else call_kwargs[0][0]
        system_msgs = [m for m in messages if m["role"] == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0]["content"] == SYSTEM_PROMPT

    @patch("src.langgraph_nodes.node_13_beginner_explanation.Groq")
    def test_existing_errors_preserved(self, mock_groq_cls):
        """Pre-existing errors in state must not be cleared."""
        mock_groq_cls.return_value.chat.completions.create.return_value = (
            _mock_groq_response("OK")
        )
        state = _make_state()
        state["errors"] = ["pre-existing error"]
        result = beginner_explanation_node(state)
        assert "pre-existing error" in result["errors"]
