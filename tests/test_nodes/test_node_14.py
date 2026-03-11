"""
Tests for Node 14: Technical Explanation (LLM)

Strategy: the Groq LLM call is mocked in every test so these run instantly
and deterministically without a real API key.

Covers:
- _build_user_prompt:
    - all 11 sections listed in prompt
    - stream scores table present for all 4 streams
    - technical indicators block (present / UNAVAILABLE)
    - pattern block (sufficient / insufficient / detail rows)
    - blending block present
    - sentiment/Node 8/NIV blocks
    - market context / MC blocks
    - risk summary fields all present
    - anomaly blocks (present / UNAVAILABLE)
    - weights_are_fallback surfaced
    - agreement=disagree surfaced
- _build_fallback_report: None sc vs valid sc
- technical_explanation_node (main):
    - happy path → report written, no errors
    - signal_components=None → fallback, error logged
    - LLM raises → fallback, error appended, no crash
    - ti=None → UNAVAILABLE in prompt, no crash
    - aw=None → no crash
    - all optional dicts None → no crash
    - node_execution_times populated
    - system prompt passed correctly
    - existing errors preserved
    - report is stripped of leading/trailing whitespace
"""

import pytest
from copy import deepcopy
from unittest.mock import MagicMock, patch

from src.langgraph_nodes.node_14_technical_explanation import (
    SYSTEM_PROMPT,
    _build_fallback_report,
    _build_user_prompt,
    technical_explanation_node,
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
) -> dict:
    return {
        "final_signal": signal,
        "final_confidence": confidence,
        "final_score": 0.42,
        "signal_agreement": agreement,
        "streams_missing": streams_missing or [],
        "stream_scores": {
            "technical":   {"raw_score": 0.65, "weight": 0.25, "contribution": 0.163},
            "sentiment":   {"raw_score": 0.48, "weight": 0.25, "contribution": 0.120},
            "market":      {"raw_score": 0.30, "weight": 0.25, "contribution": 0.075},
            "monte_carlo": {"raw_score": 0.36, "weight": 0.25, "contribution": 0.090},
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
            "behavioral_summary": "No unusual patterns.",
            "trading_recommendation": "NORMAL",
            "primary_risk_factors": ["volume_spike"],
            "alerts": ["Volume above average"],
            "detection_breakdown": {"volume_score": 0.4, "price_velocity": 0.1},
        },
        "pattern_prediction": {
            "sufficient_data": sufficient_data,
            "similar_days_found": 14,
            "similarity_threshold_used": 0.65,
            "prob_up": 0.7143,
            "prob_down": 0.2857,
            "expected_return_7d": 1.8000,
            "worst_case_7d": -0.6000,
            "best_case_7d": 4.2000,
            "median_return_7d": 1.5000,
            "agreement_with_job1": job2_agreement,
            "confidence_multiplier": 1.0500,
            "similar_days_detail": [
                {"date": "2025-11-01", "similarity_score": 0.82,
                 "actual_change_7d": 2.3, "direction": "UP"},
                {"date": "2025-10-15", "similarity_score": 0.79,
                 "actual_change_7d": -0.5, "direction": "DOWN"},
            ],
            "reason": "Only 3 snapshots",
        },
        "backtest_context": {
            "hold_threshold_pct": 58.0,
            "streams_reliable": 3,
            "weights_are_fallback": weights_are_fallback,
        },
        "prediction_graph_data": {
            "data_source": "blended",
            "blended_expected_return": 1.62,
            "gbm_expected_return": 1.20,
            "empirical_expected_return": 1.80,
            "gbm_spread_lower": -3.50,
            "gbm_spread_upper": 5.10,
            "empirical_lower": -0.60,
            "empirical_upper": 4.20,
        },
    }


def _make_ti() -> dict:
    return {
        "rsi": 58.4,
        "macd": 1.23,
        "macd_signal": 0.95,
        "macd_histogram": 0.28,
        "bb_upper": 195.0,
        "bb_lower": 172.0,
        "bb_middle": 183.5,
        "bb_position": 0.62,
        "sma_20": 183.5,
        "sma_50": 178.2,
        "volume_ratio": 1.4,
        "atr": 3.2,
    }


def _make_aw() -> dict:
    return {
        "technical_weight": 0.231,
        "stock_news_weight": 0.273,
        "market_news_weight": 0.244,
        "related_news_weight": 0.252,
        "hold_threshold_pct": 58.0,
        "streams_reliable": 3,
        "fallback_equal_weights": False,
    }


def _make_sb() -> dict:
    return {
        "stock": {
            "weighted_sentiment": 0.45,
            "article_count": 8,
            "dominant_label": "positive",
            "top_articles": [
                {"title": "AAPL beats earnings", "source": "Reuters",
                 "sentiment_score": 0.72, "credibility_weight": 0.85},
            ],
        },
        "market": {
            "weighted_sentiment": 0.22,
            "article_count": 5,
            "dominant_label": "neutral",
            "top_articles": [],
        },
        "related": {
            "weighted_sentiment": 0.18,
            "article_count": 3,
            "dominant_label": "neutral",
            "top_articles": [],
        },
        "overall": {
            "combined_sentiment": 0.35,
            "confidence": 0.7,
            "credibility": {
                "avg_source_credibility": 0.85,
                "high_credibility_articles": 3,
                "medium_credibility_articles": 2,
                "low_credibility_articles": 0,
                "avg_composite_anomaly": 0.1,
            },
        },
    }


def _make_sa() -> dict:
    return {
        "aggregated_sentiment": 0.38,
        "stock_news_sentiment": 0.45,
        "market_news_sentiment": 0.22,
        "related_news_sentiment": 0.18,
    }


def _make_niv() -> dict:
    return {
        "historical_correlation": 0.72,
        "news_accuracy_score": 68.0,
        "learning_adjustment": 1.2,
        "source_reliability": {
            "Reuters": {"accuracy_rate": 0.78, "total_articles": 38, "confidence_multiplier": 1.0},
        },
    }


def _make_mc_ctx() -> dict:
    return {
        "context_signal": "BUY",
        "market_correlation": 0.82,
        "sector_performance": 1.8,
        "market_trend": "BULLISH",
        "spy_5day_change": 1.5,
    }


def _make_mc() -> dict:
    return {
        "current_price": 185.50,
        "probability_up": 0.68,
        "expected_return": 3.5,
        "confidence_95": {"lower": 175.0, "upper": 205.0},
        "simulation_count": 10000,
        "time_horizon_days": 30,
    }


def _make_ba() -> dict:
    return {
        "risk_level": "LOW",
        "pump_and_dump_score": 12,
        "behavioral_summary": "Normal trading activity.",
        "primary_risk_factors": [],
        "detection_breakdown": {"volume_score": 0.2},
        "alerts": [],
        "trading_recommendation": "NORMAL",
    }


def _make_ca() -> dict:
    return {
        "early_risk_level": "LOW",
        "anomaly_flags": [],
        "suspicious_patterns": [],
    }


def _make_state(**overrides) -> dict:
    state = {
        "ticker": "AAPL",
        "signal_components": _make_sc(),
        "technical_indicators": _make_ti(),
        "technical_signal": "BUY",
        "technical_confidence": 0.72,
        "adaptive_weights": _make_aw(),
        "sentiment_breakdown": _make_sb(),
        "sentiment_analysis": _make_sa(),
        "news_impact_verification": _make_niv(),
        "market_context": _make_mc_ctx(),
        "monte_carlo_results": _make_mc(),
        "behavioral_anomaly_detection": _make_ba(),
        "content_analysis_summary": _make_ca(),
        "errors": [],
        "node_execution_times": {},
    }
    state.update(overrides)
    return state


def _mock_anthropic_response(text: str = "## Report\n\nMocked technical report.") -> MagicMock:
    content_block = MagicMock()
    content_block.text = text
    response = MagicMock()
    response.content = [content_block]
    return response


# ============================================================================
# _build_user_prompt
# ============================================================================

class TestBuildUserPrompt:
    def _prompt(self, sc=None, ti=None, aw=None, sb=None, sa=None,
                niv=None, mc_ctx=None, mc=None, ba=None, ca=None,
                ticker="AAPL", tech_sig="BUY", tech_conf=0.72):
        return _build_user_prompt(
            ticker=ticker,
            sc=sc or _make_sc(),
            ti=ti if ti is not None else _make_ti(),
            aw=aw if aw is not None else _make_aw(),
            sb=sb if sb is not None else _make_sb(),
            sa=sa if sa is not None else _make_sa(),
            niv=niv if niv is not None else _make_niv(),
            mc_ctx=mc_ctx if mc_ctx is not None else _make_mc_ctx(),
            mc=mc if mc is not None else _make_mc(),
            ba=ba if ba is not None else _make_ba(),
            ca=ca if ca is not None else _make_ca(),
        )

    def test_all_11_sections_listed(self):
        prompt = self._prompt()
        for section in (
            "Executive Summary", "Signal Decomposition", "Technical Analysis",
            "Sentiment Analysis", "Market Context", "Monte Carlo",
            "Historical Pattern Matching", "Adaptive Weighting",
            "Anomaly Detection", "Risk Quantification", "Methodology Notes",
        ):
            assert section in prompt, f"Section '{section}' missing from prompt"

    def test_ticker_present(self):
        assert "NVDA" in self._prompt(ticker="NVDA")

    def test_stream_scores_all_four_streams(self):
        prompt = self._prompt()
        for stream in ("technical", "sentiment", "market", "monte_carlo"):
            assert stream in prompt

    def test_technical_indicators_present(self):
        prompt = self._prompt()
        assert "rsi" in prompt.lower()
        assert "58.4" in prompt

    def test_technical_indicators_unavailable_when_empty(self):
        prompt = self._prompt(ti={})
        assert "UNAVAILABLE" in prompt

    def test_pattern_sufficient_data(self):
        prompt = self._prompt(sc=_make_sc(sufficient_data=True))
        assert "sufficient_data: True" in prompt
        assert "0.7143" in prompt       # prob_up exact
        assert "2025-11-01" in prompt   # similar day date

    def test_pattern_insufficient_data(self):
        prompt = self._prompt(sc=_make_sc(sufficient_data=False))
        assert "sufficient_data: False" in prompt
        assert "Only 3 snapshots" in prompt   # reason field

    def test_blending_block_present(self):
        prompt = self._prompt()
        assert "PREDICTION BLENDING" in prompt
        assert "60% empirical + 40% GBM" in prompt
        assert "blended" in prompt

    def test_sentiment_block_present(self):
        prompt = self._prompt()
        assert "SENTIMENT BREAKDOWN" in prompt
        assert "AAPL beats earnings" in prompt

    def test_sentiment_unavailable_when_empty(self):
        prompt = self._prompt(sb={})
        assert "UNAVAILABLE" in prompt

    def test_niv_block_present(self):
        prompt = self._prompt()
        assert "NEWS IMPACT VERIFICATION" in prompt
        assert "0.72" in prompt         # historical_correlation

    def test_niv_unavailable_when_empty(self):
        prompt = self._prompt(niv={})
        assert "UNAVAILABLE" in prompt

    def test_market_context_present(self):
        prompt = self._prompt()
        assert "MARKET CONTEXT" in prompt
        assert "BULLISH" in prompt

    def test_market_context_unavailable_when_empty(self):
        prompt = self._prompt(mc_ctx={})
        assert "UNAVAILABLE" in prompt

    def test_monte_carlo_present(self):
        prompt = self._prompt()
        assert "MONTE CARLO" in prompt
        assert "10000" in prompt        # simulation_count

    def test_risk_summary_fields_present(self):
        prompt = self._prompt()
        assert "overall_risk_level" in prompt
        assert "pump_and_dump_score" in prompt
        assert "detection_breakdown" in prompt
        assert "behavioral_summary" in prompt

    def test_anomaly_blocks_present(self):
        prompt = self._prompt()
        assert "CONTENT ANOMALY" in prompt
        assert "BEHAVIORAL ANOMALY" in prompt

    def test_anomaly_unavailable_when_empty(self):
        prompt = self._prompt(ba={}, ca={})
        assert "UNAVAILABLE" in prompt

    def test_weights_fallback_surfaced(self):
        prompt = self._prompt(sc=_make_sc(weights_are_fallback=True))
        assert "weights_are_fallback: True" in prompt

    def test_disagree_agreement_present(self):
        prompt = self._prompt(sc=_make_sc(job2_agreement="disagree"))
        assert "disagree" in prompt


# ============================================================================
# _build_fallback_report
# ============================================================================

class TestBuildFallbackReport:
    def test_none_sc_returns_unavailable(self):
        result = _build_fallback_report(None, "AAPL")
        assert "Unavailable" in result or "unavailable" in result.lower()
        assert "AAPL" in result

    def test_valid_sc_returns_signal(self):
        result = _build_fallback_report(_make_sc(signal="SELL", confidence=0.55), "MSFT")
        assert "SELL" in result
        assert "MSFT" in result

    def test_none_returns_markdown(self):
        result = _build_fallback_report(None, "X")
        assert result.startswith("##")

    def test_valid_sc_returns_markdown(self):
        result = _build_fallback_report(_make_sc(), "X")
        assert result.startswith("##")

    def test_returns_string(self):
        assert isinstance(_build_fallback_report(None, "X"), str)
        assert isinstance(_build_fallback_report(_make_sc(), "X"), str)


# ============================================================================
# technical_explanation_node — main node
# ============================================================================

class TestTechnicalExplanationNode:

    @patch("src.langgraph_nodes.node_14_technical_explanation.ANTHROPIC_API_KEY", "fake-key")
    @patch("src.langgraph_nodes.node_14_technical_explanation.anthropic.Anthropic")
    def test_happy_path_writes_report(self, mock_anthropic_cls):
        mock_anthropic_cls.return_value.messages.create.return_value = (
            _mock_anthropic_response("## Report\nDetailed analysis.")
        )
        state = _make_state()
        result = technical_explanation_node(state)
        assert result["technical_explanation"] == "## Report\nDetailed analysis."
        assert result["errors"] == []

    @patch("src.langgraph_nodes.node_14_technical_explanation.ANTHROPIC_API_KEY", "fake-key")
    @patch("src.langgraph_nodes.node_14_technical_explanation.anthropic.Anthropic")
    def test_execution_time_recorded(self, mock_anthropic_cls):
        mock_anthropic_cls.return_value.messages.create.return_value = (
            _mock_anthropic_response()
        )
        state = _make_state()
        result = technical_explanation_node(state)
        assert "node_14" in result["node_execution_times"]
        assert result["node_execution_times"]["node_14"] >= 0.0

    def test_signal_components_none_returns_fallback(self):
        state = _make_state(signal_components=None)
        result = technical_explanation_node(state)
        assert "technical_explanation" in result
        assert len(result["technical_explanation"]) > 0
        assert any("signal_components" in e for e in result["errors"])

    @patch("src.langgraph_nodes.node_14_technical_explanation.ANTHROPIC_API_KEY", "fake-key")
    @patch("src.langgraph_nodes.node_14_technical_explanation.anthropic.Anthropic")
    def test_llm_failure_returns_fallback_no_crash(self, mock_anthropic_cls):
        mock_anthropic_cls.return_value.messages.create.side_effect = RuntimeError("Timeout")
        state = _make_state()
        result = technical_explanation_node(state)
        assert "technical_explanation" in result
        assert len(result["technical_explanation"]) > 0
        assert any("LLM failed" in e for e in result["errors"])

    @patch("src.langgraph_nodes.node_14_technical_explanation.ANTHROPIC_API_KEY", "fake-key")
    @patch("src.langgraph_nodes.node_14_technical_explanation.anthropic.Anthropic")
    def test_technical_indicators_none_no_crash(self, mock_anthropic_cls):
        mock_anthropic_cls.return_value.messages.create.return_value = (
            _mock_anthropic_response("OK")
        )
        state = _make_state(technical_indicators=None)
        result = technical_explanation_node(state)
        assert result["technical_explanation"] == "OK"

    @patch("src.langgraph_nodes.node_14_technical_explanation.ANTHROPIC_API_KEY", "fake-key")
    @patch("src.langgraph_nodes.node_14_technical_explanation.anthropic.Anthropic")
    def test_adaptive_weights_none_no_crash(self, mock_anthropic_cls):
        mock_anthropic_cls.return_value.messages.create.return_value = (
            _mock_anthropic_response("OK")
        )
        state = _make_state(adaptive_weights=None)
        result = technical_explanation_node(state)
        assert result["technical_explanation"] == "OK"

    @patch("src.langgraph_nodes.node_14_technical_explanation.ANTHROPIC_API_KEY", "fake-key")
    @patch("src.langgraph_nodes.node_14_technical_explanation.anthropic.Anthropic")
    def test_all_optional_dicts_none_no_crash(self, mock_anthropic_cls):
        mock_anthropic_cls.return_value.messages.create.return_value = (
            _mock_anthropic_response("OK")
        )
        state = _make_state(
            technical_indicators=None,
            adaptive_weights=None,
            sentiment_breakdown=None,
            sentiment_analysis=None,
            news_impact_verification=None,
            market_context=None,
            monte_carlo_results=None,
            behavioral_anomaly_detection=None,
            content_analysis_summary=None,
        )
        result = technical_explanation_node(state)
        assert result["technical_explanation"] == "OK"
        assert result["errors"] == []

    @patch("src.langgraph_nodes.node_14_technical_explanation.ANTHROPIC_API_KEY", "fake-key")
    @patch("src.langgraph_nodes.node_14_technical_explanation.anthropic.Anthropic")
    def test_returns_state_dict(self, mock_anthropic_cls):
        mock_anthropic_cls.return_value.messages.create.return_value = (
            _mock_anthropic_response()
        )
        state = _make_state()
        result = technical_explanation_node(state)
        assert isinstance(result, dict)
        assert "ticker" in result

    @patch("src.langgraph_nodes.node_14_technical_explanation.ANTHROPIC_API_KEY", "fake-key")
    @patch("src.langgraph_nodes.node_14_technical_explanation.anthropic.Anthropic")
    def test_strips_whitespace_from_llm_output(self, mock_anthropic_cls):
        mock_anthropic_cls.return_value.messages.create.return_value = (
            _mock_anthropic_response("  \n  ## Report\n  \n  ")
        )
        state = _make_state()
        result = technical_explanation_node(state)
        assert result["technical_explanation"] == "## Report"

    @patch("src.langgraph_nodes.node_14_technical_explanation.ANTHROPIC_API_KEY", "fake-key")
    @patch("src.langgraph_nodes.node_14_technical_explanation.anthropic.Anthropic")
    def test_system_prompt_passed_correctly(self, mock_anthropic_cls):
        mock_instance = mock_anthropic_cls.return_value
        mock_instance.messages.create.return_value = _mock_anthropic_response()
        state = _make_state()
        technical_explanation_node(state)
        call_kwargs = mock_instance.messages.create.call_args
        kwargs = call_kwargs[1] if call_kwargs[1] else {}
        assert kwargs.get("system") == SYSTEM_PROMPT

    @patch("src.langgraph_nodes.node_14_technical_explanation.ANTHROPIC_API_KEY", "fake-key")
    @patch("src.langgraph_nodes.node_14_technical_explanation.anthropic.Anthropic")
    def test_existing_errors_preserved(self, mock_anthropic_cls):
        mock_anthropic_cls.return_value.messages.create.return_value = (
            _mock_anthropic_response()
        )
        state = _make_state()
        state["errors"] = ["pre-existing error"]
        result = technical_explanation_node(state)
        assert "pre-existing error" in result["errors"]

    @patch("src.langgraph_nodes.node_14_technical_explanation.ANTHROPIC_API_KEY", "fake-key")
    @patch("src.langgraph_nodes.node_14_technical_explanation.anthropic.Anthropic")
    def test_disagree_state_no_crash(self, mock_anthropic_cls):
        mock_anthropic_cls.return_value.messages.create.return_value = (
            _mock_anthropic_response("OK")
        )
        state = _make_state(signal_components=_make_sc(job2_agreement="disagree"))
        result = technical_explanation_node(state)
        assert result["technical_explanation"] == "OK"

    @patch("src.langgraph_nodes.node_14_technical_explanation.ANTHROPIC_API_KEY", "fake-key")
    @patch("src.langgraph_nodes.node_14_technical_explanation.anthropic.Anthropic")
    def test_sell_signal_no_crash(self, mock_anthropic_cls):
        mock_anthropic_cls.return_value.messages.create.return_value = (
            _mock_anthropic_response("OK")
        )
        state = _make_state(signal_components=_make_sc(signal="SELL", confidence=0.30))
        result = technical_explanation_node(state)
        assert result["technical_explanation"] == "OK"
