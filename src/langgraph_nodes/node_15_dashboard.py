"""
Node 15: Dashboard Data Preparation & Report Generation

Aggregates all outputs from Nodes 1-14 into a structured `dashboard_data`
dict optimised for LangSmith/LangGraph UI display and future Streamlit rendering.

From a Goldman Sachs senior analyst perspective this node has one job:
surface EVERY data point that drives the investment recommendation — clearly
separated between (a) what clients see, (b) what the analyst uses to form the
view, and (c) model quality / risk flags.

Runs AFTER: Node 14 (technical_explanation)
Runs BEFORE: END
Can run in PARALLEL with: Nothing (terminal node)
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from src.utils.logger import get_node_logger

logger = get_node_logger("node_15")


# ============================================================================
# SECTION BUILDERS
# ============================================================================

def _executive_summary(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tier 1 — what the client sees on page 1.

    Condenses the final recommendation, confidence, price targets, and a
    one-line risk verdict into a format suitable for a client cover page.
    """
    sc = state.get("signal_components") or {}
    risk = sc.get("risk_summary") or {}
    pt = sc.get("price_targets") or {}
    pg = sc.get("prediction_graph_data") or {}
    sa = sc.get("signal_agreement") or {}
    mc = state.get("monte_carlo_results") or {}

    final_signal: str = state.get("final_signal") or "HOLD"
    final_confidence: float = float(state.get("final_confidence") or 0.0)

    # signal_agreement in Node 12 state is the integer count of streams that agree
    # Job 1/2 agreement string is in pattern_prediction["agreement_with_job1"]
    pp = sc.get("pattern_prediction") or {}
    streams_agreeing: int = int(sa) if isinstance(sa, (int, float)) else int(sc.get("streams_agreeing", 0))
    agreement_label: str = pp.get("agreement_with_job1") or (sa.get("agreement") if isinstance(sa, dict) else "N/A")

    current_price: Optional[float] = pt.get("current_price") or mc.get("current_price")
    expected_return: Optional[float] = pg.get("blended_expected_return") or pt.get("expected_return_pct")
    prob_up: Optional[float] = mc.get("probability_up")

    # Derive a simple one-liner for the client
    if not risk.get("trading_safe", True):
        risk_verdict = f"CAUTION — {risk.get('overall_risk_level','UNKNOWN')} behavioral risk detected"
    else:
        risk_verdict = f"Clear — risk level {risk.get('overall_risk_level','UNKNOWN')}"

    return {
        "ticker":                 state.get("ticker",         "N/A"),
        "signal_strength":        int(state.get("signal_strength") or 0),
        "analysis_timestamp":     state.get("timestamp", datetime.now()).isoformat()
                                  if hasattr(state.get("timestamp", ""), "isoformat")
                                  else str(state.get("timestamp", datetime.now())),
        "recommendation":         final_signal,
        "confidence_pct":         f"{final_confidence * 100:.1f}%",
        "confidence_raw":         round(final_confidence, 4),
        "streams_agreeing":       streams_agreeing,
        "streams_total":          4,
        "signal_agreement":       agreement_label,
        "current_price_usd":      round(current_price, 2) if current_price else None,
        "expected_return_7d_pct": round(expected_return, 2) if expected_return is not None else None,
        "probability_up_pct":     f"{prob_up * 100:.1f}%" if prob_up is not None else None,
        "probability_up_raw":     round(prob_up, 4) if prob_up is not None else None,
        "trading_safe":           risk.get("trading_safe", True),
        "risk_verdict":           risk_verdict,
        "related_companies":      state.get("related_companies", []),
    }


def _price_intelligence(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tier 1 — price targets and forecast parameters.

    Blended (Job2 + GBM) return as primary; GBM-only as fallback.
    """
    mc = state.get("monte_carlo_results") or {}
    sc = state.get("signal_components") or {}
    pt = sc.get("price_targets") or {}
    pg = sc.get("prediction_graph_data") or {}
    ci95 = mc.get("confidence_95") or {}
    ci75 = mc.get("confidence_75") or {}

    current_price: float = float(mc.get("current_price") or 0.0)

    return {
        "current_price_usd":          current_price,
        "forecasted_price_7d_usd":    state.get("forecasted_price"),
        "expected_return_gbm_pct":    mc.get("expected_return"),
        "expected_return_blended_pct": pg.get("blended_expected_return"),
        "expected_return_empirical_pct": pg.get("empirical_expected_return"),
        "forecast_data_source":       pg.get("data_source", "gbm_only"),
        "price_range_lower_usd":      pt.get("price_range_lower"),
        "price_range_upper_usd":      pt.get("price_range_upper"),
        "confidence_95_lower_usd":    ci95.get("lower"),
        "confidence_95_upper_usd":    ci95.get("upper"),
        "confidence_75_lower_usd":    ci75.get("lower"),
        "confidence_75_upper_usd":    ci75.get("upper"),
        "probability_up":             mc.get("probability_up"),
        "probability_down":           mc.get("probability_down"),
        "mean_forecast_usd":          mc.get("mean_forecast"),
        "simulations_run":            mc.get("num_simulations", 1000),
        "forecast_horizon_days":      mc.get("forecast_days", 7),
        "gbm_spread_lower_pct":       pg.get("gbm_spread_lower"),
        "gbm_spread_upper_pct":       pg.get("gbm_spread_upper"),
        "empirical_lower_pct":        pg.get("empirical_lower"),
        "empirical_upper_pct":        pg.get("empirical_upper"),
    }


def _technical_analysis(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tier 2 — IC/Ridge model output and raw indicator readings.

    technical_alpha is the primary decision metric (IC/Ridge regression).
    Heuristic normalized_score is preserved as secondary reference.
    """
    indicators: Dict[str, Any] = state.get("technical_indicators") or {}
    sc   = state.get("signal_components") or {}
    ss   = sc.get("stream_scores") or {}
    tech = ss.get("technical") or {}
    aw   = state.get("adaptive_weights") or {}

    macd_dict = indicators.get("macd") or {}

    return {
        # IC / Ridge regression (primary alpha signal)
        "technical_alpha":         indicators.get("technical_alpha"),
        "ic_score":                indicators.get("ic_score"),
        "regression_valid":        indicators.get("regression_valid", False),
        "n_training_samples":      indicators.get("n_training_samples"),
        "predicted_return_pct":    indicators.get("predicted_return_pct"),
        "market_regime":           indicators.get("market_regime"),
        # Heuristic score (legacy, for reference)
        "normalized_score_0_100":  indicators.get("normalized_score"),
        "hold_band_low":           indicators.get("hold_low"),
        "hold_band_high":          indicators.get("hold_high"),
        # Raw indicators
        "rsi_14":                  indicators.get("rsi"),
        "macd":                    macd_dict.get("macd") if isinstance(macd_dict, dict) else indicators.get("macd"),
        "macd_signal":             macd_dict.get("signal") if isinstance(macd_dict, dict) else None,
        "macd_hist":               macd_dict.get("histogram") if isinstance(macd_dict, dict) else indicators.get("macd_hist"),
        "bb_pct_b":                indicators.get("bb_pct_b"),
        "sma_20":                  indicators.get("sma20"),
        "sma_50":                  indicators.get("sma50"),
        "adx":                     indicators.get("adx"),
        "volume_ratio":            indicators.get("volume_ratio"),
        "sma20_slope_5d":          indicators.get("sma20_slope_5d"),
        "persistent_pressure":     indicators.get("persistent_pressure"),
        "technical_summary":       indicators.get("technical_summary"),
        # Stream-level aggregates
        "stream_raw_score":        tech.get("raw_score"),
        "stream_weight":           aw.get("technical_weight"),
        "stream_weighted_score":   tech.get("weighted_score"),
        "stream_signal":           state.get("technical_signal"),
        "stream_confidence":       state.get("technical_confidence"),
    }


def _sentiment_intelligence(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tier 2 — FinBERT sentiment across all three news streams.

    Breakdown keys: stock / market / related / overall.
    """
    sb = state.get("sentiment_breakdown") or {}
    sc = state.get("signal_components") or {}
    ss = sc.get("stream_scores") or {}
    aw = state.get("adaptive_weights") or {}

    stock_stream   = ss.get("stock_news")   or {}
    market_stream  = ss.get("market_news")  or {}
    related_stream = ss.get("related_news") or {}

    def _score_from_stream(stream_dict: Dict[str, Any]) -> Optional[float]:
        return stream_dict.get("raw_score")

    return {
        # Aggregated output
        "aggregated_sentiment":    state.get("aggregated_sentiment"),
        "sentiment_signal":        state.get("sentiment_signal"),
        "sentiment_confidence":    state.get("sentiment_confidence"),
        # Breakdown by news type — each key is a stream dict; extract scalar sentiment
        "stock_sentiment":         sb.get("stock", {}).get("weighted_sentiment") if isinstance(sb.get("stock"), dict) else sb.get("stock"),
        "market_sentiment":        sb.get("market", {}).get("weighted_sentiment") if isinstance(sb.get("market"), dict) else sb.get("market"),
        "related_sentiment":       sb.get("related", {}).get("weighted_sentiment") if isinstance(sb.get("related"), dict) else sb.get("related"),
        "overall_sentiment":       sb.get("overall", {}).get("combined_sentiment") if isinstance(sb.get("overall"), dict) else sb.get("overall"),
        # Per-stream scores (used by Node 12)
        "stock_stream_score":      _score_from_stream(stock_stream),
        "market_stream_score":     _score_from_stream(market_stream),
        "related_stream_score":    _score_from_stream(related_stream),
        # Stream weights
        "stock_news_weight":       aw.get("stock_news_weight"),
        "market_news_weight":      aw.get("market_news_weight"),
        "related_news_weight":     aw.get("related_news_weight"),
        # Article counts — prefer stream dict counts, fall back to state list lengths
        "stock_articles_analyzed":   (sb.get("stock", {}).get("article_count") if isinstance(sb.get("stock"), dict) else None)
                                     or len(state.get("cleaned_stock_news") or []),
        "market_articles_analyzed":  (sb.get("market", {}).get("article_count") if isinstance(sb.get("market"), dict) else None)
                                     or len(state.get("cleaned_market_news") or []),
        "related_articles_analyzed": (sb.get("related", {}).get("article_count") if isinstance(sb.get("related"), dict) else None)
                                     or len(state.get("cleaned_related_company_news") or []),
        "total_articles":            (sb.get("overall", {}).get("article_count") if isinstance(sb.get("overall"), dict) else None)
                                     or (len(state.get("cleaned_stock_news") or []) +
                                         len(state.get("cleaned_market_news") or []) +
                                         len(state.get("cleaned_related_company_news") or [])),
    }


def _market_context(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tier 2 — macro / sector context and market correlation.

    Node 6 returns a nested structure; flatten it here for downstream display.
    Sub-dicts: stock_classification, market_regime, sector_industry_context,
               macro_factor_exposure, market_correlation_profile, news_sentiment_context.
    """
    mc_data   = state.get("market_context") or {}
    sc        = state.get("signal_components") or {}
    ss        = sc.get("stream_scores") or {}
    mkt_st    = ss.get("market_context") or {}
    aw        = state.get("adaptive_weights") or {}

    # Node 6 nested sub-dicts
    stk_cls   = mc_data.get("stock_classification")    or {}
    regime    = mc_data.get("market_regime")            or {}
    sector_cx = mc_data.get("sector_industry_context")  or {}
    corr_prof = mc_data.get("market_correlation_profile") or {}
    news_ctx  = mc_data.get("news_sentiment_context")  or {}

    return {
        # Sector / industry (from sector_industry_context + stock_classification)
        "sector":              stk_cls.get("sector"),
        "sector_trend":        sector_cx.get("sector_trend"),
        "sector_return":       sector_cx.get("sector_return_5d"),
        "relative_strength":   sector_cx.get("relative_strength_label"),
        # Market regime (from market_regime)
        "market_trend":        regime.get("spy_trend_label"),
        "spy_return":          regime.get("spy_return_5d"),
        "vix_level":           regime.get("vix_level"),
        "vix_trend":           regime.get("vix_category"),
        "regime_label":        regime.get("regime_label"),
        # Correlation (from market_correlation_profile)
        "market_correlation":  corr_prof.get("market_correlation"),
        "beta":                corr_prof.get("beta_calculated"),
        # Legacy flat keys — kept for backward compatibility with any consumers
        # that were written before the Node 6 refactor
        "context_signal":      mc_data.get("context_signal"),
        "context_score":       mc_data.get("context_score"),
        # Stream aggregate
        "stream_raw_score":    mkt_st.get("raw_score"),
        "stream_weight":       aw.get("market_news_weight"),
        "stream_weighted_score": mkt_st.get("weighted_score"),
    }


def _monte_carlo(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tier 2 — GBM Monte Carlo simulation parameters.
    """
    mc = state.get("monte_carlo_results") or {}
    sc = state.get("signal_components") or {}
    ss = sc.get("stream_scores") or {}
    mc_st = ss.get("monte_carlo") or {}
    aw    = state.get("adaptive_weights") or {}
    ci95  = mc.get("confidence_95") or {}
    ci75  = mc.get("confidence_75") or {}

    return {
        "current_price":         mc.get("current_price"),
        "mean_forecast":         mc.get("mean_forecast"),
        "expected_return_pct":   mc.get("expected_return"),
        "probability_up":        mc.get("probability_up"),
        "probability_down":      mc.get("probability_down"),
        "volatility_daily":      mc.get("volatility"),
        "drift_daily":           mc.get("drift"),
        "ci_95_lower":           ci95.get("lower"),
        "ci_95_upper":           ci95.get("upper"),
        "ci_75_lower":           ci75.get("lower"),
        "ci_75_upper":           ci75.get("upper"),
        "num_simulations":       mc.get("num_simulations", 1000),
        "forecast_days":         mc.get("forecast_days", 7),
        # Stream aggregate
        "stream_raw_score":      mc_st.get("raw_score"),
        "stream_weight":         aw.get("related_news_weight"),  # note: monte_carlo has no separate weight key
        "stream_weighted_score": mc_st.get("weighted_score"),
    }


def _risk_assessment(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tier 3 — behavioral anomaly and content risk flags.

    A high pump_and_dump_score or HIGH/CRITICAL behavioral risk overrides
    the recommendation with a mandatory CAUTION flag.

    Exposes two distinct risk dimensions:
      - overall_risk_level  (pump/manipulation risk from Node 9)
      - volatility_risk     (investment/market risk derived from beta + VIX)
    """
    sc = state.get("signal_components") or {}
    risk = sc.get("risk_summary") or {}
    bad = state.get("behavioral_anomaly_detection") or {}
    cas = state.get("content_analysis_summary") or {}

    # Volatility risk — separate concept from manipulation/pump risk
    _mc = state.get("market_context") or {}
    beta = float((_mc.get("market_correlation_profile") or {}).get("beta_calculated") or 1.0)
    vix  = float((_mc.get("market_regime") or {}).get("vix_level") or 20.0)
    if beta > 1.3 and vix > 22:
        volatility_risk = "HIGH"
    elif beta > 1.0 or vix > 22:
        volatility_risk = "MODERATE"
    else:
        volatility_risk = "LOW"

    return {
        "overall_risk_level":       risk.get("overall_risk_level", "UNKNOWN"),
        "volatility_risk":          volatility_risk,
        "beta":                     beta,
        "vix_level":                vix,
        "behavioral_risk":          risk.get("behavioral_risk", "UNKNOWN"),
        "content_risk":             risk.get("content_risk", "UNKNOWN"),
        "trading_safe":             risk.get("trading_safe", True),
        "trading_recommendation":   risk.get("trading_recommendation", "NORMAL"),
        "pump_and_dump_score":      risk.get("pump_and_dump_score", 0),
        "behavioral_summary":       risk.get("behavioral_summary", ""),
        "primary_risk_factors":     risk.get("primary_risk_factors", []),
        "alerts":                   risk.get("alerts", []),
        "detection_breakdown":      risk.get("detection_breakdown", {}),
        # Raw Node 9B fields for full detail
        "composite_risk_score":     bad.get("composite_risk_score"),
        "seven_detector_scores":    bad.get("detector_scores"),
        # Node 9A content analysis
        "early_risk_level":         cas.get("early_risk_level"),
        "manipulation_flags":       cas.get("manipulation_flags", []),
        "total_articles_screened":  cas.get("total_articles"),
        "flagged_articles":         cas.get("flagged_articles"),
    }


def _model_performance(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tier 3 — adaptive weights, backtest accuracy, stream reliability.

    The weights encode how much the system trusts each signal stream based on
    180 days of backtested directional accuracy (Bayesian-smoothed, recency-
    weighted, Node 8-adjusted).
    """
    aw = state.get("adaptive_weights") or {}
    br = state.get("backtest_results") or {}

    def _stream_accuracy(key: str) -> Optional[float]:
        stream = br.get(key) or {}
        return stream.get("full_accuracy")

    def _stream_recent(key: str) -> Optional[float]:
        stream = br.get(key) or {}
        return stream.get("recent_accuracy")

    def _stream_ic(key: str) -> Optional[float]:
        stream = br.get(key) or {}
        return stream.get("ic_score")

    def _stream_ic_p(key: str) -> Optional[float]:
        stream = br.get(key) or {}
        return stream.get("ic_p_value")

    def _stream_ic_sig(key: str) -> Optional[bool]:
        stream = br.get(key) or {}
        return stream.get("ic_significant")

    # Use the explicit flag stored by calculate_news_ic — no need to scan daily results
    market_decomposed: bool = bool((br.get("market_news") or {}).get("ic_used_idiosyncratic"))

    return {
        # Normalized weights (sum to 1.0)
        "technical_weight":          aw.get("technical_weight"),
        "stock_news_weight":         aw.get("stock_news_weight"),
        "market_news_weight":        aw.get("market_news_weight"),
        "related_news_weight":       aw.get("related_news_weight"),
        # Quality metadata
        "streams_reliable":          aw.get("streams_reliable", 0),
        "fallback_equal_weights":    aw.get("fallback_equal_weights", False),
        "hold_threshold_pct":        aw.get("hold_threshold_pct"),
        "sample_period_days":        aw.get("sample_period_days", 180),
        "per_stream_adjustments":    aw.get("per_stream_adjustments", {}),
        "weight_sources":            aw.get("weight_sources", {}),
        "weighted_accuracies":       aw.get("weighted_accuracies", {}),
        "learning_adjustment_applied": aw.get("learning_adjustment_applied"),
        "historical_correlation":    aw.get("historical_correlation"),
        # Raw backtest accuracy per stream (full period)
        "technical_accuracy_full":   _stream_accuracy("technical"),
        "technical_accuracy_recent": _stream_recent("technical"),
        "stock_news_accuracy_full":  _stream_accuracy("stock_news"),
        "stock_news_accuracy_recent": _stream_recent("stock_news"),
        "market_news_accuracy_full": _stream_accuracy("market_news"),
        "market_news_accuracy_recent": _stream_recent("market_news"),
        "related_news_accuracy_full":  _stream_accuracy("related_news"),
        "related_news_accuracy_recent": _stream_recent("related_news"),
        # Information Coefficient per stream — continuous predictive power measure
        "technical_ic":              _stream_ic("technical"),
        "technical_ic_p":            _stream_ic_p("technical"),
        "technical_ic_significant":  _stream_ic_sig("technical"),
        "stock_news_ic":             _stream_ic("stock_news"),
        "stock_news_ic_p":           _stream_ic_p("stock_news"),
        "stock_news_ic_significant": _stream_ic_sig("stock_news"),
        "market_news_ic":            _stream_ic("market_news"),
        "market_news_ic_p":          _stream_ic_p("market_news"),
        "market_news_ic_significant": _stream_ic_sig("market_news"),
        "market_news_decomposed":    market_decomposed,
        "related_news_ic":           _stream_ic("related_news"),
        "related_news_ic_p":         _stream_ic_p("related_news"),
        "related_news_ic_significant": _stream_ic_sig("related_news"),
    }


def _news_intelligence(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tier 3 — Node 8 learning: source credibility and sentiment-price correlation.

    learning_adjustment > 1 means news has historically been a reliable signal.
    """
    niv = state.get("news_impact_verification") or {}

    return {
        "learning_adjustment":       niv.get("learning_adjustment", 1.0),
        "news_accuracy_score":       niv.get("news_accuracy_score"),
        "historical_correlation":    niv.get("historical_correlation"),
        "sample_size":               niv.get("sample_size", 0),
        "insufficient_data":         niv.get("insufficient_data", True),
        "news_type_effectiveness":   niv.get("news_type_effectiveness", {}),
        "source_reliability":        niv.get("source_reliability", {}),
        "sentiment_analysis":        state.get("sentiment_analysis"),
    }


def _historical_pattern(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tier 3 — Node 12 Job 2: signal-conditioned historical pattern matching.

    Empirical outcomes from the k most similar historical days (inverse-distance
    weighted kNN across 4-dimensional signal space).
    """
    sc = state.get("signal_components") or {}
    pp = sc.get("pattern_prediction") or {}
    sa = sc.get("signal_agreement") or {}

    return {
        "sufficient_data":          pp.get("sufficient_data", False),
        "similar_days_found":       pp.get("similar_days_found"),
        "avg_similarity":           pp.get("avg_similarity"),
        "similarity_threshold":     pp.get("similarity_threshold"),
        "prob_up_empirical":        pp.get("prob_up"),
        "expected_return_7d_pct":   pp.get("expected_return_7d"),
        "best_case_7d_pct":         pp.get("best_case_7d"),
        "worst_case_7d_pct":        pp.get("worst_case_7d"),
        "p25_return_pct":           pp.get("p25_return"),
        "p75_return_pct":           pp.get("p75_return"),
        "reason":                   pp.get("reason"),
        # Job 1/2 agreement — stored in pattern_prediction under "agreement_with_job1"
        "signal_agreement":         pp.get("agreement_with_job1"),
        "streams_agreeing_count":   sa if isinstance(sa, int) else None,
    }


def _llm_analysis(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tier 1 — LLM-generated explanations (Node 13 beginner, Node 14 technical).
    """
    beginner_text: Optional[str] = state.get("beginner_explanation")
    technical_text: Optional[str] = state.get("technical_explanation")

    return {
        "beginner_explanation":    beginner_text,
        "technical_explanation":   technical_text,
        "beginner_available":      beginner_text is not None and len(beginner_text) > 0,
        "technical_available":     technical_text is not None and len(technical_text) > 0,
        "beginner_word_count":     len(beginner_text.split()) if beginner_text else 0,
        "technical_word_count":    len(technical_text.split()) if technical_text else 0,
    }


def _pipeline_integrity(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tier 4 — execution health: timing, errors, data completeness.
    """
    execution_times: Dict[str, float] = state.get("node_execution_times") or {}
    errors: List[str] = state.get("errors") or []
    total_time: Optional[float] = (
        sum(execution_times.values()) if execution_times else None
    )

    # Count how many optional state fields are populated
    optional_fields = [
        "raw_price_data", "technical_indicators", "aggregated_sentiment",
        "market_context", "monte_carlo_results", "news_impact_verification",
        "behavioral_anomaly_detection", "backtest_results", "adaptive_weights",
        "final_signal", "beginner_explanation", "technical_explanation",
    ]
    populated = sum(1 for f in optional_fields if state.get(f) is not None)
    completeness = populated / len(optional_fields)

    if completeness >= 0.90 and len(errors) == 0:
        data_quality = "EXCELLENT"
    elif completeness >= 0.75 and len(errors) <= 2:
        data_quality = "GOOD"
    elif completeness >= 0.50:
        data_quality = "DEGRADED"
    else:
        data_quality = "POOR"

    return {
        "total_execution_time_s": round(total_time, 2) if total_time else None,
        "node_execution_times":   {k: round(v, 3) for k, v in execution_times.items()},
        "slowest_node":           max(execution_times, key=execution_times.get)
                                  if execution_times else None,
        "error_count":            len(errors),
        "errors":                 errors,
        "data_completeness_pct":  f"{completeness * 100:.0f}%",
        "data_quality":           data_quality,
        "price_days_available":   len(state["raw_price_data"]) if state.get("raw_price_data") is not None else 0,
        "related_companies_found": len(state.get("related_companies") or []),
    }


# ============================================================================
# ADDITIONAL SECTION BUILDERS
# ============================================================================

def _trustworthiness(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tier 1 — signal trustworthiness and per-stream hit rates.

    Exposes the trustworthiness_breakdown produced by Node 12 so the
    Streamlit app can display hit rates, reliability, and prior flags
    without touching signal_components directly.
    """
    sc = state.get("signal_components") or {}
    tw = sc.get("trustworthiness_breakdown") or {}
    return {
        "reliability_score":    tw.get("reliability_score", 0),
        "agreement_score":      tw.get("agreement_score",   0),
        "pattern_score":        tw.get("pattern_score",     0),
        "insufficient_history": tw.get("insufficient_history", True),
        "using_prior":          tw.get("using_prior", False),
        "stream_hit_rates":     tw.get("stream_hit_rates", {}),
    }


def _visualization(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tier 1 — raw chart data for Streamlit.

    Bundles the bulk arrays and nested dicts that the four chart
    builders need (MC paths, similar-day records, stream scores,
    forecast graph parameters) so the app never has to reach into
    signal_components or monte_carlo_results directly.
    """
    mc = state.get("monte_carlo_results") or {}
    sc = state.get("signal_components")   or {}
    pp = sc.get("pattern_prediction")     or {}
    pg = sc.get("prediction_graph_data")  or {}
    ss = sc.get("stream_scores")          or {}
    return {
        # build_monte_carlo_chart — needs the raw path arrays
        "mc_visualization":      mc.get("visualization_data") or {},
        # build_similar_days_chart — needs individual day records
        "similar_days_detail":   pp.get("similar_days_detail", []),
        # build_stream_bars — needs per-stream raw_score + weight
        "stream_scores":         ss,
        # build_forecast_chart — needs blend weights and spread bounds
        "prediction_graph_data": pg,
        # shared baseline for forecast chart
        "current_price":         float(mc.get("current_price") or 0),
    }


def _macro_factors(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tier 2 — commodity / macro factor exposure table.

    Extracted from Node 6 market_context so the app renders the
    factor table without touching the raw market_context dict.
    """
    mc_data   = state.get("market_context") or {}
    macro_exp = mc_data.get("macro_factor_exposure") or {}
    return {
        "identified_factors": macro_exp.get("identified_factors", []),
        "macro_summary":      macro_exp.get("macro_summary", ""),
    }


def _peer_companies(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tier 2 — related companies (peers, suppliers, customers, competitors).

    Passes through the Node 3 related_companies list verbatim so the
    Streamlit app can render a peer table without touching raw state.
    Each entry is a dict with keys: ticker, relationship, reason.
    """
    related: List[Dict[str, Any]] = state.get("related_companies") or []
    return {
        "companies": related,
        "count":     len(related),
    }


def _sec_fundamentals(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tier 2 — SEC filing analysis from Node 16.

    Passes through the fundamental_context dict with a thin reshaping so
    the Streamlit app never reads state directly.  Returns an empty-but-safe
    dict when Node 16 did not run or returned data_quality='unavailable'.
    """
    fc = state.get("fundamental_context") or {}
    if not fc or fc.get("data_quality") == "unavailable":
        return {"available": False}

    target      = fc.get("target") or {}
    peers_raw   = fc.get("peers") or {}
    divergences = fc.get("divergences") or []

    # Flatten peers dict → list for easier iteration in the UI
    peers_list = [
        {
            "ticker":        pt,
            "relationship":  pd.get("relationship", ""),
            "recent_events": pd.get("recent_events") or [],
            "event_sentiment": pd.get("event_sentiment"),
            "filing_dates":  pd.get("filing_dates") or [],
        }
        for pt, pd in peers_raw.items()
    ]

    return {
        "available":              True,
        "fundamental_signal":     fc.get("fundamental_signal", "NEUTRAL"),
        "fundamental_confidence": fc.get("fundamental_confidence", 0.0),
        "data_quality":           fc.get("data_quality", "partial"),
        "revenue_trend":          target.get("revenue_trend", "stable"),
        "margin_trend":           target.get("margin_trend", "stable"),
        "management_sentiment":   target.get("management_sentiment"),
        "recent_events":          target.get("recent_events") or [],
        "filing_dates":           (target.get("filing_dates") or [])[:3],
        "quarters_covered":       target.get("quarters_covered", 0),
        "peers":                  peers_list,
        "divergences":            divergences,
    }


def _diagnostics(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tier 4 — data-availability and backtest-sufficiency counters.

    Pre-computes the row/article counts and backtest stream dicts
    that the signal-diagnostics widget needs, so the widget never
    reads raw state lists.
    """
    price_data = state.get("raw_price_data")
    br         = state.get("backtest_results") or {}
    return {
        "price_rows": len(price_data) if price_data is not None else 0,
        "stock_articles":   len(
            state.get("cleaned_stock_news")
            or state.get("stock_news", [])
        ),
        "market_articles":  len(
            state.get("cleaned_market_news")
            or state.get("market_news", [])
        ),
        "related_articles": len(
            state.get("cleaned_related_company_news")
            or state.get("related_company_news", [])
        ),
        "backtest_results": br,
    }


# ============================================================================
# FORMATTED PRINT REPORT (visible in LangSmith trace stdout)
# ============================================================================

def _print_gs_report(data: Dict[str, Any]) -> None:
    """
    Print a Goldman Sachs-style investment report to stdout.

    This appears in the LangSmith node trace as the node's log output,
    giving the analyst an at-a-glance view of the full recommendation.
    """
    es   = data["executive_summary"]
    pi   = data["price_intelligence"]
    ta   = data["technical_analysis"]
    si   = data["sentiment_intelligence"]
    mc   = data["monte_carlo"]
    risk = data["risk_assessment"]
    mp   = data["model_performance"]
    ni   = data["news_intelligence"]
    hp   = data["historical_pattern"]
    pipe = data["pipeline_integrity"]

    def _f(v: Any, fmt: str = ".2f", fallback: str = "N/A") -> str:
        """Format a float safely; returns fallback when v is None."""
        if v is None:
            return fallback
        try:
            return f"{v:{fmt}}"
        except (TypeError, ValueError):
            return str(v)

    ticker       = es["ticker"]
    rec          = es["recommendation"]
    conf         = es["confidence_pct"]
    cur_price    = es.get("current_price_usd")
    exp_ret      = es.get("expected_return_7d_pct")
    prob_up      = es.get("probability_up_pct")

    SEP  = "=" * 72
    SEP2 = "-" * 72

    lines = [
        "",
        SEP,
        f"  GOLDMAN SACHS EQUITY RESEARCH — {ticker}",
        f"  Generated: {es['analysis_timestamp']}",
        SEP,
        "",
        # ─── EXECUTIVE SUMMARY ──────────────────────────────────────────────
        "  INVESTMENT RECOMMENDATION",
        SEP2,
        f"  Signal   : {rec}",
        f"  Confidence: {conf}  ({es['streams_agreeing']}/4 streams agree, Job1/2: {es.get('signal_agreement','N/A')})",
        f"  Current Price: ${cur_price:.2f}"  if cur_price else "  Current Price: N/A",
        f"  7-Day Expected Return: {exp_ret:+.2f}%"  if exp_ret is not None else "  7-Day Expected Return: N/A",
        f"  Prob. Up (7d): {prob_up}"  if prob_up else "  Prob. Up (7d): N/A",
        f"  Price Range: ${pi.get('price_range_lower_usd') or pi.get('ci_95_lower_usd', '?'):.2f}"
        f" – ${pi.get('price_range_upper_usd') or pi.get('ci_95_upper_usd', '?'):.2f}"
        if (pi.get("price_range_lower_usd") or pi.get("ci_95_lower_usd")) else
        "  Price Range: N/A",
        f"  Risk Status: {es['risk_verdict']}",
        "",
        # ─── FOUR-STREAM BREAKDOWN ───────────────────────────────────────────
        "  SIGNAL STREAMS (weighted contribution)",
        SEP2,
    ]

    # Technical
    alpha = ta.get("technical_alpha")
    ic    = ta.get("ic_score")
    rsi   = ta.get("rsi_14")
    tw    = mp.get("technical_weight")
    lines += [
        f"  [1] TECHNICAL ANALYSIS  (weight: {tw*100:.1f}%)" if tw else "  [1] TECHNICAL ANALYSIS",
        f"      Alpha (IC/Ridge): {alpha:+.3f}  |  IC Score: {ic:.3f}" if (alpha is not None and ic is not None) else
        f"      Alpha: N/A",
        f"      Regression Valid: {ta.get('regression_valid')}  |  Training Samples: {ta.get('n_training_samples')}",
        f"      Predicted 7d Return: {ta.get('predicted_return_pct'):+.2f}%" if ta.get("predicted_return_pct") is not None else "      Predicted 7d Return: N/A",
        f"      RSI(14): {_f(rsi,'.1f')}  |  ADX: {_f(ta.get('adx'),'.1f')}  |  BB%B: {_f(ta.get('bb_pct_b'),'.2f')}"
        if rsi else "      RSI: N/A",
        f"      Market Regime: {ta.get('market_regime','N/A')}",
        f"      Stream Score: {ta.get('stream_raw_score'):+.3f}" if ta.get("stream_raw_score") is not None else "",
        "",
    ]

    # Sentiment
    agg = si.get("aggregated_sentiment")
    snw = mp.get("stock_news_weight")
    mnw = mp.get("market_news_weight")
    rnw = mp.get("related_news_weight")
    lines += [
        f"  [2] SENTIMENT ANALYSIS  (stock {snw*100:.1f}% / market {mnw*100:.1f}% / related {rnw*100:.1f}%)"
        if (snw and mnw and rnw) else "  [2] SENTIMENT ANALYSIS",
        f"      Aggregated Sentiment: {_f(agg,'+.3f')}  |  Signal: {si.get('sentiment_signal','N/A')}  |  Conf: {_f((si.get('sentiment_confidence') or 0)*100,'.1f')}%"
        if agg is not None else "      Aggregated Sentiment: N/A",
        f"      Stock: {_f(si.get('stock_sentiment'),'+.3f')}  |  Market: {_f(si.get('market_sentiment'),'+.3f')}  |  Related: {_f(si.get('related_sentiment'),'+.3f')}",
        f"      Articles: {si.get('total_articles',0)} total  ({si.get('stock_articles_analyzed',0)} stock / {si.get('market_articles_analyzed',0)} market / {si.get('related_articles_analyzed',0)} related)",
        "",
    ]

    # Market context
    mc_d = data["market_context"]
    lines += [
        "  [3] MARKET CONTEXT",
        f"      Sector: {mc_d.get('sector','N/A')}  |  Trend: {mc_d.get('sector_trend','N/A')}  |  Regime: {mc_d.get('regime_label','N/A')}",
        f"      Market Trend: {mc_d.get('market_trend','N/A')}  |  Correlation: {mc_d.get('market_correlation', 0):.2f}  |  Beta: {_f(mc_d.get('beta'),'.2f')}"
        if mc_d.get("market_correlation") is not None else
        "      Market Trend: N/A  |  Correlation: N/A",
        f"      VIX: {_f(mc_d.get('vix_level'),'.2f')} ({mc_d.get('vix_trend','N/A')})  |  Relative Strength: {mc_d.get('relative_strength','N/A')}",
        "",
    ]

    # Monte Carlo
    prob_u = mc.get("probability_up")
    exp_r  = mc.get("expected_return_pct")
    lines += [
        "  [4] MONTE CARLO FORECAST (GBM, 1000 simulations, 7-day horizon)",
        f"      Prob. Up: {prob_u*100:.1f}%  |  Expected Return: {exp_r:+.2f}%"
        if (prob_u and exp_r is not None) else "      Prob. Up: N/A",
        f"      95% CI: ${_f(mc.get('ci_95_lower'),'.2f')} – ${_f(mc.get('ci_95_upper'),'.2f')}"
        if mc.get("ci_95_lower") else "      95% CI: N/A",
        "",
    ]

    # Historical pattern
    lines += [
        "  HISTORICAL PATTERN MATCHING (kNN on 4-dim signal space)",
        SEP2,
        f"  Sufficient Data: {hp.get('sufficient_data')}  |  Similar Days Found: {hp.get('similar_days_found','N/A')}",
        f"  Empirical Prob. Up: {_f((hp.get('prob_up_empirical') or 0)*100,'.1f')}%  |  Exp. Return 7d: {_f(hp.get('expected_return_7d_pct'),'+.2f')}%"
        if hp.get("prob_up_empirical") is not None else "  Empirical data: insufficient",
        f"  Case Range: {_f(hp.get('worst_case_7d_pct'),'+.1f')}% / {_f(hp.get('best_case_7d_pct'),'+.1f')}%  |  Agreement: {hp.get('signal_agreement','N/A')}"
        if hp.get("worst_case_7d_pct") is not None else "",
        "",
    ]

    # Risk
    rd = data["risk_assessment"]
    alerts = rd.get("alerts") or []
    lines += [
        "  RISK ASSESSMENT",
        SEP2,
        f"  Behavioral Risk: {rd.get('overall_risk_level','N/A')}  |  Pump & Dump Score: {rd.get('pump_and_dump_score',0)}/100",
        f"  Content Risk: {rd.get('content_risk','N/A')}  |  Trading Safe: {rd.get('trading_safe',True)}",
        f"  Recommendation: {rd.get('trading_recommendation','NORMAL')}",
    ]
    if alerts:
        lines.append(f"  Alerts ({len(alerts)}): " + "; ".join(str(a) for a in alerts[:3]))
    rf = rd.get("primary_risk_factors") or []
    if rf:
        lines.append(f"  Risk Factors: " + ", ".join(str(r) for r in rf[:4]))
    lines.append("")

    # Model performance — use _f() for all floats that may be None
    def _wt(key: str) -> str:
        v = mp.get(key)
        return f"{v*100:.1f}%" if v is not None else "N/A"

    def _acc(key: str) -> str:
        v = mp.get(key)
        return f"{v*100:.1f}%" if v is not None else "N/A"

    def _ic(ic_key: str, p_key: str) -> str:
        ic_val = mp.get(ic_key)
        ic_p   = mp.get(p_key)
        if ic_val is None:
            return "N/A"
        sig_flag = " *" if (ic_p is not None and ic_p < 0.05) else ""
        return f"{ic_val:+.3f}{sig_flag} (p={_f(ic_p, '.3f')})"

    mkt_decomp = "decomposed" if mp.get("market_news_decomposed") else "raw return"

    lines += [
        "  MODEL PERFORMANCE (180-day backtest, Bayesian-smoothed)",
        SEP2,
        f"  {'Stream':<14}  {'Weight':>8}  {'Full acc':>10}  {'Recent':>10}  {'IC Score':>20}",
        f"  {'-'*14}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*20}",
        f"  {'Technical':<14}  {_wt('technical_weight'):>8}  {_acc('technical_accuracy_full'):>10}  {_acc('technical_accuracy_recent'):>10}  {_ic('technical_ic','technical_ic_p'):>20}",
        f"  {'Stock news':<14}  {_wt('stock_news_weight'):>8}  {_acc('stock_news_accuracy_full'):>10}  {_acc('stock_news_accuracy_recent'):>10}  {_ic('stock_news_ic','stock_news_ic_p'):>20}",
        f"  {'Market news':<14}  {_wt('market_news_weight'):>8}  {_acc('market_news_accuracy_full'):>10}  {_acc('market_news_accuracy_recent'):>10}  {_ic('market_news_ic','market_news_ic_p'):>20}  [{mkt_decomp}]",
        f"  {'Related':<14}  {_wt('related_news_weight'):>8}  {_acc('related_news_accuracy_full'):>10}  {_acc('related_news_accuracy_recent'):>10}  {_ic('related_news_ic','related_news_ic_p'):>20}",
        f"  (* IC significant at p<0.05)",
        f"  Streams Reliable: {mp.get('streams_reliable',0)}/4  |  Equal-weight Fallback: {mp.get('fallback_equal_weights',False)}",
        f"  News Learning Adj.: {_f(ni.get('learning_adjustment',1.0),'.3f')}  |  Hist. Correlation: {_f(ni.get('historical_correlation'),'.3f')}",
        "",
    ]

    # LLM Analysis (Node 13 + 14)
    llm = data["llm_analysis"]
    lines += [
        "  LLM ANALYSIS",
        SEP2,
    ]
    beginner = llm.get("beginner_explanation") or ""
    technical = llm.get("technical_explanation") or ""
    if beginner:
        lines += [
            f"  [Node 13 — Beginner Explanation]  ({llm.get('beginner_word_count',0)} words)",
            "",
        ]
        for line in beginner.strip().splitlines():
            lines.append(f"  {line}")
        lines.append("")
    else:
        lines.append("  [Node 13 — Beginner Explanation]  (not available)")
    if technical:
        lines += [
            f"  [Node 14 — Technical Explanation]  ({llm.get('technical_word_count',0)} words)",
            "",
        ]
        for line in technical.strip().splitlines():
            lines.append(f"  {line}")
        lines.append("")
    else:
        lines.append("  [Node 14 — Technical Explanation]  (not available)")
    lines.append("")

    # Pipeline integrity
    lines += [
        "  PIPELINE INTEGRITY",
        SEP2,
        f"  Data Quality: {pipe.get('data_quality','N/A')}  |  Completeness: {pipe.get('data_completeness_pct','N/A')}",
        f"  Total Runtime: {pipe.get('total_execution_time_s','N/A')}s  |  Errors: {pipe.get('error_count',0)}",
        f"  Price History: {pipe.get('price_days_available',0)} days  |  Related Co.: {pipe.get('related_companies_found',0)}",
    ]
    if pipe.get("errors"):
        for err in pipe["errors"][:3]:
            lines.append(f"  ! {err}")
    lines += ["", SEP, ""]

    print("\n".join(lines))


# ============================================================================
# MAIN NODE FUNCTION
# ============================================================================

def dashboard_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 15: Dashboard Data Preparation & Report Generation

    Reads the final state from Nodes 1-14 and assembles a structured
    `dashboard_data` dict with four tiers of information:

    Tier 1 — Executive Summary: recommendation, price targets, risk verdict
    Tier 2 — Decision Framework: 4-stream breakdown with weighted scores
    Tier 3 — Model Quality: weights, backtest accuracy, pattern matching
    Tier 4 — Pipeline Integrity: timing, errors, data completeness

    Also prints a Goldman Sachs-style formatted report to stdout, which
    appears in the LangSmith trace for immediate at-a-glance review.

    Args:
        state: Final LangGraph state from Node 14

    Returns:
        Updated state with dashboard_data populated
    """
    start_time = datetime.now()
    ticker = state.get("ticker", "UNKNOWN")

    try:
        logger.info(f"Node 15: Assembling dashboard for {ticker}")

        dashboard_data: Dict[str, Any] = {
            # Tier 1: what clients see
            "executive_summary":      _executive_summary(state),
            "price_intelligence":     _price_intelligence(state),
            "llm_analysis":           _llm_analysis(state),
            "trustworthiness":        _trustworthiness(state),
            "visualization":          _visualization(state),
            # Tier 2: what drives the call
            "technical_analysis":     _technical_analysis(state),
            "sentiment_intelligence": _sentiment_intelligence(state),
            "market_context":         _market_context(state),
            "monte_carlo":            _monte_carlo(state),
            "macro_factors":          _macro_factors(state),
            "peer_companies":         _peer_companies(state),
            "sec_fundamentals":       _sec_fundamentals(state),
            # Tier 3: model quality & validation
            "model_performance":      _model_performance(state),
            "news_intelligence":      _news_intelligence(state),
            "historical_pattern":     _historical_pattern(state),
            "risk_assessment":        _risk_assessment(state),
            # Tier 4: pipeline health
            "pipeline_integrity":     _pipeline_integrity(state),
            "diagnostics":            _diagnostics(state),
        }

        # Compute total_execution_time and stamp it before printing
        execution_time = (datetime.now() - start_time).total_seconds()
        state.setdefault("node_execution_times", {})["node_15"] = execution_time

        # Refresh pipeline_integrity with node_15 included
        dashboard_data["pipeline_integrity"] = _pipeline_integrity(state)

        state["dashboard_data"]     = dashboard_data
        state["total_execution_time"] = sum(
            state["node_execution_times"].values()
        )

        _print_gs_report(dashboard_data)

        logger.info(
            f"Node 15: Dashboard assembled in {execution_time:.3f}s — "
            f"recommendation={dashboard_data['executive_summary']['recommendation']} "
            f"confidence={dashboard_data['executive_summary']['confidence_pct']}"
        )

        return state

    except Exception as e:
        logger.error(f"Node 15 failed: {e}")
        logger.exception("Full traceback:")
        state.setdefault("errors", []).append(f"Node 15 (dashboard) failed: {e}")
        state["dashboard_data"] = None
        state.setdefault("node_execution_times", {})["node_15"] = (
            datetime.now() - start_time
        ).total_seconds()
        return state
