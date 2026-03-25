"""
Node 14: Technical Explanation (LLM)

Generates a structured markdown research report for a sophisticated investor.
Calls the Anthropic Claude LLM with a hardcoded system prompt and a dynamically
assembled user prompt built entirely from state data — no external knowledge is used.

Reads from state (all tiers):
  TIER 1  — signal_components       (Node 12: full signal breakdown, risk, pattern, blending)
  TIER 2  — technical_indicators    (Node 4: RSI, MACD, Bollinger, SMA, volume, ATR)
  TIER 3  — adaptive_weights        (Node 11: per-stream weights and reliability)
  TIER 4  — sentiment_breakdown     (Node 5: per-stream scores, top articles, credibility)
            sentiment_analysis      (Node 8 adjusted: per-stream sentiment values)
  TIER 5  — news_impact_verification(Node 8: source reliability, learning adjustment)
  TIER 6  — market_context          (Node 6: sector, correlation, market trend)
  TIER 7  — monte_carlo_results     (Node 7: probability_up, CI, simulation params)
  TIER 8  — behavioral_anomaly_detection (Node 9B: full detection_breakdown)
  TIER 9  — content_analysis_summary    (Node 9A: anomaly flags)
  TIER 10 — ticker (direct field; technical_signal/confidence removed — see ti dict)

Writes to state:
  technical_explanation    — markdown string, 600-900 words
  node_execution_times     — records "node_14" duration

Runs AFTER:  Node 13 (beginner_explanation)
Runs BEFORE: Node 15 (dashboard)
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import anthropic

from src.utils.config import ANTHROPIC_API_KEY
from src.utils.logger import get_node_logger

logger = get_node_logger("node_14")


# ============================================================================
# CONSTANTS
# ============================================================================

CLAUDE_MODEL: str = "claude-sonnet-4-5"
MAX_TOKENS: int = 3200  # ~1100 words with per-section budgets and headroom


def _fmt(value: Any, fmt: str = "+.4f", fallback: str = "N/A") -> str:
    """Safe number formatter — returns fallback string if value is None or unformattable."""
    if value is None:
        return fallback
    try:
        return format(value, fmt)
    except (TypeError, ValueError):
        return fallback


# ============================================================================
# SYSTEM PROMPT (hardcoded — never changes between runs)
# ============================================================================

SYSTEM_PROMPT: str = """You are a quantitative analyst generating a technical research \
report on a stock trading signal for a sophisticated investor or portfolio manager.

Rules you must follow:
- Use ONLY the data provided in the user message. Do not supplement with any external \
knowledge about the company, sector, or market from your training data.
- Report exact numbers as given. Do not round unless the number is already rounded.
- If a data field is None or marked UNAVAILABLE, explicitly note the gap — do not skip \
silently and do not invent values.
- When Job 1 and Job 2 disagree (agreement_with_job1 = "disagree"), lead the Executive \
Summary with this finding prominently.
- When insufficient_history is True, note in the Signal Quality section that \
trustworthiness reflects a statistical prior (0.55), not measured accuracy.
- When insufficient_history is False, interpret trustworthiness as a backtested hit rate \
and note which streams are driving it up or down based on stream_hit_rates.
- Follow these per-section word budgets as hard upper limits. Total output \
must not exceed 1100 words excluding markdown headers: \
Section 1 Executive Summary: 180 words. \
Section 2 Signal Decomposition and Quality: 160 words. \
Section 3 Technical and Quantitative Analysis: 220 words. \
Section 4 Sentiment and News Analysis: 160 words. \
Section 5 Market Context and Risk: 160 words. \
Section 6 Anomaly Detection: 120 words. \
Section 7 Methodology Notes: 100 words minimum, 120 words maximum. \
Do not exceed any individual section budget. If a section has less to say, \
use fewer words — do not pad. These budgets exist to ensure every section \
is completed and none is truncated.
- Use markdown headers (##) for each section. Follow the 7-section order exactly.
- Synthesise — do not just list data points. A sophisticated reader wants your \
interpretation of what the combination of signals means, not a table of values.
- Before writing any section, identify the single most important tension or \
confirmation in the data — the finding that most affects whether the signal \
should be trusted. Lead the Executive Summary with that finding. Every \
subsequent section should either support or complicate it.
- If the technical regression direction contradicts price-action indicators \
(e.g. regression predicts a negative return while MACD is bullish, or vice \
versa), this contradiction must be the first sentence of the Executive Summary. \
Label it explicitly as 'internal technical disagreement', quantify the gap \
between what regression predicts and what price-action indicators show, and \
re-reference it in section 3.
- In section 4, after presenting source accuracy data, write one explicit \
conclusion sentence in this exact form: 'Sentiment signal for [TICKER] is \
[reliable / unreliable / low-reliability] based on [X]% measured accuracy — \
[weight accordingly / treat as corroborating evidence only / discount \
substantially].' Do not leave the sentiment reliability conclusion implicit.
- In section 5, state explicitly: 'If SPY drops 1%, [TICKER] is expected to \
drop approximately [beta]% based on its calculated beta.' Then combine beta \
and VIX into a single volatility risk statement for the reader. Do not \
mention beta and VIX in separate sentences — synthesise them.
- Section 7 must explicitly address each of the following if present in the \
data: (a) weights_are_fallback=True — state that adaptive stream weights are \
prior-based, not learned from this ticker's history; (b) \
insufficient_history=True — state that trustworthiness reflects a statistical \
prior of 0.55, not a measured hit rate; (c) streams_missing — name each \
missing stream and state what analytical dimension is blind as a result; \
(d) avg_news_accuracy below 40% — flag sentiment as a low-reliability stream \
for this ticker. If none of these conditions apply, write 'No material data \
gaps or reliability flags detected.' Never write section 7 in fewer than \
80 words.
- Do not add sections not listed below."""


# ============================================================================
# USER PROMPT BUILDER
# ============================================================================

def _build_user_prompt(
    ticker: str,
    sc: Dict[str, Any],
    ti: Dict[str, Any],
    aw: Dict[str, Any],
    sb: Dict[str, Any],
    sa: Dict[str, Any],
    niv: Dict[str, Any],
    mc_ctx: Dict[str, Any],
    mc: Dict[str, Any],
    ba: Dict[str, Any],
    ca: Dict[str, Any],
    related_companies: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Assemble the full data block from pre-extracted state dicts.

    Every argument is already guarded to be a dict (possibly empty) or a
    scalar. The prompt builder never accesses state directly.

    Node 4 no longer emits discrete technical_signal / technical_confidence
    fields. All technical context (normalized_score, hold_low, hold_high,
    market_regime, persistent_pressure, technical_summary) is carried inside
    the ``ti`` dict and rendered automatically in the TECHNICAL INDICATORS
    block below. The final trading signal is sourced exclusively from Node 12
    via ``sc`` (signal_components).

    Args:
        ticker:  Stock ticker symbol.
        sc:      signal_components (Node 12) — authoritative final signal.
        ti:      technical_indicators (Node 4) — full numeric context.
        aw:      adaptive_weights (Node 11).
        sb:      sentiment_breakdown (Node 5).
        sa:      sentiment_analysis (Node 8 adjusted).
        niv:     news_impact_verification (Node 8).
        mc_ctx:  market_context (Node 6).
        mc:      monte_carlo_results (Node 7).
        ba:      behavioral_anomaly_detection (Node 9B).
        ca:      content_analysis_summary (Node 9A).

    Returns:
        Formatted user prompt string.
    """
    pp: Dict[str, Any] = sc.get("pattern_prediction") or {}
    pt: Dict[str, Any] = sc.get("price_targets") or {}
    rs: Dict[str, Any] = sc.get("risk_summary") or {}
    pg: Dict[str, Any] = sc.get("prediction_graph_data") or {}
    bc: Dict[str, Any] = sc.get("backtest_context") or {}
    ss: Dict[str, Any] = sc.get("stream_scores") or {}

    # -------------------------------------------------------------------------
    # STREAM SCORES TABLE
    # -------------------------------------------------------------------------
    stream_rows: List[str] = []
    for stream in ("technical", "sentiment", "market", "related_news"):
        d = ss.get(stream) or {}
        stream_rows.append(
            f"  {stream:<12} | raw={d.get('raw_score')}  "
            f"weight={d.get('weight')}  contribution={d.get('contribution')}"
        )
    stream_table = "\n".join(stream_rows)

    # -------------------------------------------------------------------------
    # TECHNICAL INDICATORS BLOCK
    # -------------------------------------------------------------------------
    if ti:
        _ti_keep = [
            "normalized_score", "rsi", "macd_histogram", "macd_histogram_slope",
            "sma_relationship", "volume_ratio", "adx", "bollinger_position",
            "predicted_return", "market_regime", "persistent_pressure",
            "technical_summary", "hold_low", "hold_high",
        ]
        ti_lines = [f"  {k}: {v}" for k, v in ti.items() if k in _ti_keep]
        if not ti_lines:  # fallback: schema change → dump all
            ti_lines = [f"  {k}: {v}" for k, v in ti.items()]
        ti_block = "\n".join(ti_lines)
    else:
        ti_block = "UNAVAILABLE — Node 4 did not produce data"

    # -------------------------------------------------------------------------
    # SENTIMENT BLOCK
    # -------------------------------------------------------------------------
    if sb:
        # Correct keys: sentiment_breakdown uses 'stock'/'market'/'related'/'overall'
        sn = sb.get("stock")  or {}
        mn = sb.get("market") or {}
        rn = sb.get("related") or {}
        overall = sb.get("overall") or {}

        # Collect top articles from all three streams.
        # Node 05 already caps each stream to top 3, so max 9 articles total here.
        all_top_articles = (
            sn.get("top_articles", []) +
            mn.get("top_articles", []) +
            rn.get("top_articles", [])
        )
        article_lines = [
            f'  - [{stream}] "{a.get("title")}" | source={a.get("source")} '
            f'| sentiment={a.get("sentiment_score")} | credibility={a.get("credibility_weight")}'
            for stream, articles in [
                ("stock",   sn.get("top_articles", [])),
                ("market",  mn.get("top_articles", [])),
                ("related", rn.get("top_articles", [])),
            ]
            for a in articles
        ]

        # Credibility aggregate — from the correct key sb['overall']['credibility']
        cred = overall.get("credibility") or {}
        cred_block = (
            f"  avg_source_credibility: {cred.get('avg_source_credibility')}\n"
            f"  high_credibility_articles: {cred.get('high_credibility_articles')}\n"
            f"  medium_credibility_articles: {cred.get('medium_credibility_articles')}\n"
            f"  low_credibility_articles: {cred.get('low_credibility_articles')}\n"
            f"  avg_composite_anomaly: {cred.get('avg_composite_anomaly')}"
        )

        sb_block = (
            f"  stock   score={sn.get('weighted_sentiment')} articles={sn.get('article_count')} dominant={sn.get('dominant_label')}\n"
            f"  market  score={mn.get('weighted_sentiment')} articles={mn.get('article_count')} dominant={mn.get('dominant_label')}\n"
            f"  related score={rn.get('weighted_sentiment')} articles={rn.get('article_count')} dominant={rn.get('dominant_label')}\n"
            f"  overall combined_sentiment={overall.get('combined_sentiment')} confidence={overall.get('confidence')}\n"
            f"  credibility:\n{cred_block}\n"
            f"  top_articles (top 3 per stream, already pre-selected by Node 5):\n" + "\n".join(article_lines)
        )
    else:
        sb_block = "UNAVAILABLE"

    # Node 8 adjusted per-stream values
    if sa:
        sa_block = (
            f"  aggregated_sentiment: {sa.get('aggregated_sentiment')}\n"
            f"  stock_news_sentiment: {sa.get('stock_news_sentiment')}\n"
            f"  market_news_sentiment: {sa.get('market_news_sentiment')}\n"
            f"  related_news_sentiment: {sa.get('related_news_sentiment')}"
        )
    else:
        sa_block = "UNAVAILABLE — Node 8 did not adjust sentiment"

    # News impact verification
    if niv:
        src_rel = niv.get("source_reliability") or {}
        src_filtered = {
            src: v for src, v in src_rel.items()
            if isinstance(v, dict) and int(v.get("total_articles") or 0) >= 10
        }
        src_top5 = sorted(
            src_filtered.items(),
            key=lambda x: int(x[1].get("total_articles") or 0),
            reverse=True,
        )[:5]

        if src_top5:
            avg_accuracy = sum(
                float(v.get("accuracy_rate") or 0) for _, v in src_top5
            ) / len(src_top5)
        else:
            avg_accuracy = 0.0

        src_lines = [
            f"    {src}: accuracy={v.get('accuracy_rate')} "
            f"articles={v.get('total_articles')} multiplier={v.get('confidence_multiplier')}"
            for src, v in src_top5
        ]

        niv_block = (
            f"  historical_correlation: {niv.get('historical_correlation')}\n"
            f"  news_accuracy_score: {niv.get('news_accuracy_score')}\n"
            f"  learning_adjustment: {niv.get('learning_adjustment')}\n"
            f"  avg_accuracy_top5_sources: {avg_accuracy:.1%} "
            f"({'low-reliability — discount sentiment stream' if avg_accuracy < 0.40 else 'moderate reliability' if avg_accuracy < 0.60 else 'high reliability'})\n"
            f"  source_reliability (top 5 by volume, min 10 articles):\n"
            + "\n".join(src_lines)
        )
    else:
        niv_block = "UNAVAILABLE — Node 8 learning data not present"

    # -------------------------------------------------------------------------
    # MARKET CONTEXT BLOCK (specific fields only — never dump raw dict)
    # Node 6 stores a nested dict; extract from the correct sub-keys.
    # -------------------------------------------------------------------------
    if mc_ctx:
        regime       = mc_ctx.get("market_regime") or {}
        sector_ctx   = mc_ctx.get("sector_industry_context") or {}
        corr_profile = mc_ctx.get("market_correlation_profile") or {}
        stock_cls    = mc_ctx.get("stock_classification") or {}
        news_ctx     = mc_ctx.get("news_sentiment_context") or {}
        macro_exp    = mc_ctx.get("macro_factor_exposure") or {}

        # Commodity factors block (only factors with a fetched price)
        identified_factors: List[Dict[str, Any]] = macro_exp.get("identified_factors") or []
        _trend_arrow = {"UP": "↑", "DOWN": "↓", "FLAT": "→"}
        commodity_lines: List[str] = []
        for fac in identified_factors:
            if not isinstance(fac, dict):
                continue
            f_name  = fac.get("factor_name", "Unknown")
            f_type  = fac.get("exposure_type", "")
            f_expl  = fac.get("exposure_explanation", "")
            f_price = fac.get("current_price")
            f_trend = fac.get("price_trend")
            price_str = f"${f_price:,.4f}" if f_price is not None else "N/A"
            trend_str = (
                f" | trend: {_trend_arrow.get(f_trend, '')} {f_trend}"
                if f_trend else ""
            )
            commodity_lines.append(
                f"    {f_name} ({f_type}): {price_str}{trend_str} — {f_expl}"
            )
        commodity_block_str = (
            "\n".join(commodity_lines) if commodity_lines else "    none fetched"
        )

        mc_ctx_block = (
            f"  regime_label: {regime.get('regime_label')}\n"
            f"  regime_description: {regime.get('regime_description')}\n"
            f"  spy_trend_label: {regime.get('spy_trend_label')}\n"
            f"  spy_return_5d: {_fmt(regime.get('spy_return_5d'))}%\n"
            f"  spy_return_21d: {_fmt(regime.get('spy_return_21d'))}%\n"
            f"  vix_level: {regime.get('vix_level')} ({regime.get('vix_category')})\n"
            f"  sector: {stock_cls.get('sector')}\n"
            f"  sector_return_5d: {_fmt(sector_ctx.get('sector_return_5d'))}%"
            f" | trend: {sector_ctx.get('sector_trend')}\n"
            f"  relative_strength_label: {sector_ctx.get('relative_strength_label')}\n"
            f"  sector_context_note: {sector_ctx.get('sector_context_note')}\n"
            f"  market_correlation: {_fmt(corr_profile.get('market_correlation'))}\n"
            f"  beta_calculated: {_fmt(corr_profile.get('beta_calculated'))}\n"
            f"  beta_interpretation: {corr_profile.get('beta_interpretation')}\n"
            f"  market_news_sentiment: {_fmt(news_ctx.get('market_news_sentiment'))}"
            f" ({news_ctx.get('sentiment_label')})\n"
            f"  macro_summary: {macro_exp.get('macro_summary')}\n"
            f"  commodity_factors (use to assess cost/revenue impact on price):\n"
            + commodity_block_str
        )

        # Append peer companies if available
        if related_companies:
            _rel_lines: List[str] = []
            for c in related_companies:
                if not isinstance(c, dict):
                    continue
                _rel_lines.append(
                    f"    {c.get('ticker', '')} ({c.get('relationship', 'SAME_SECTOR')}): {c.get('reason', '')}"
                )
            if _rel_lines:
                mc_ctx_block += (
                    "\n  peer_companies (suppliers / customers / competitors — use in"
                    " section 5 Market Context & Risk):\n" + "\n".join(_rel_lines)
                )

        # Pre-computed beta/VIX volatility risk statement
        _beta = float(corr_profile.get("beta_calculated") or 1.0)
        _vix  = float(regime.get("vix_level") or 20.0)

        if _beta > 1.3 and _vix > 22:
            _vol_risk = "HIGH — amplified market sensitivity in elevated volatility"
            _vol_note = (
                f"A 1% SPY decline implies approximately {_beta:.2f}% decline in "
                f"{ticker} under current VIX={_vix:.1f} conditions."
            )
        elif _beta > 1.0 or _vix > 22:
            _vol_risk = "MODERATE"
            _vol_note = (
                f"Beta of {_beta:.2f} with VIX at {_vix:.1f} — above-average "
                f"sensitivity to market moves."
            )
        else:
            _vol_risk = "LOW"
            _vol_note = (
                f"Beta {_beta:.2f} and VIX {_vix:.1f} indicate contained "
                f"volatility exposure."
            )

        mc_ctx_block += (
            f"\n  PRE-COMPUTED VOLATILITY RISK: {_vol_risk}\n"
            f"  {_vol_note}\n"
            f"  INSTRUCTION: Use this pre-computed statement verbatim as the "
            f"opening of the volatility risk paragraph in section 5. Do not "
            f"restate beta and VIX separately — use the synthesised statement."
        )
    else:
        mc_ctx_block = "UNAVAILABLE"

    # -------------------------------------------------------------------------
    # MONTE CARLO BLOCK (specific fields only — exclude simulation_paths arrays)
    # -------------------------------------------------------------------------
    if mc:
        ci95 = mc.get("confidence_95") or {}
        mc_block = (
            f"  PRIMARY (lead with these):\n"
            f"  probability_up: {mc.get('probability_up')}\n"
            f"  expected_return: {mc.get('expected_return')}\n"
            f"  confidence_95_lower: {ci95.get('lower')}\n"
            f"  confidence_95_upper: {ci95.get('upper')}\n"
            f"  volatility: {mc.get('volatility')}\n"
            f"  SECONDARY (reference only):\n"
            f"  mean_forecast: {mc.get('mean_forecast')}\n"
            f"  current_price: {mc.get('current_price')}\n"
            f"  METHODOLOGY (section 7 only):\n"
            f"  simulation_count: {mc.get('simulation_count')}\n"
            f"  time_horizon_days: {mc.get('time_horizon_days')}"
        )
    else:
        mc_block = "UNAVAILABLE"

    # -------------------------------------------------------------------------
    # PATTERN PREDICTION BLOCK (full detail)
    # -------------------------------------------------------------------------
    if pp.get("sufficient_data"):
        detail = pp.get("similar_days_detail") or []
        detail_top3 = sorted(
            [d for d in detail if d.get("actual_change_7d") is not None],
            key=lambda d: float(d.get("similarity_score") or 0),
            reverse=True,
        )[:3]

        detail_rows = "\n".join(
            f"  {d.get('date')} | sim={_fmt(d.get('similarity_score'), '.3f')} "
            f"| change={_fmt(d.get('actual_change_7d'), '+.2f')}% | {d.get('direction')}"
            for d in detail_top3
        )

        if len(detail_top3) >= 2:
            changes = [float(d.get("actual_change_7d") or 0) for d in detail_top3]
            dispersion = max(changes) - min(changes)
            dispersion_note = (
                f"\n  outcome_dispersion_top3: {dispersion:+.2f}% range — "
                + ("high variance, low predictive value" if dispersion > 10
                   else "moderate variance" if dispersion > 5
                   else "low variance, higher predictive value")
            )
        else:
            dispersion_note = ""

        pattern_block = (
            f"  sufficient_data: True\n"
            f"  similar_days_found: {pp.get('similar_days_found')} "
            f"(threshold: {pp.get('similarity_threshold_used')})\n"
            f"  prob_up: {_fmt(pp.get('prob_up'), '.4f')} | prob_down: {_fmt(pp.get('prob_down'), '.4f')}\n"
            f"  expected_return_7d: {_fmt(pp.get('expected_return_7d'))}%\n"
            f"  worst_case_7d (10th pct): {_fmt(pp.get('worst_case_7d'))}%\n"
            f"  best_case_7d (90th pct): {_fmt(pp.get('best_case_7d'))}%\n"
            f"  median_return_7d: {_fmt(pp.get('median_return_7d'))}%\n"
            f"  agreement_with_job1: {pp.get('agreement_with_job1')}\n"
            f"  similar_days_detail (top 3 by similarity):\n{detail_rows}"
            + dispersion_note
        )
    else:
        pattern_block = (
            f"  sufficient_data: False\n"
            f"  reason: {pp.get('reason', 'unknown')}"
        )

    # -------------------------------------------------------------------------
    # BLENDING BLOCK
    # -------------------------------------------------------------------------
    if pg:
        blend_lower = pg.get("gbm_spread_lower")
        blend_upper = pg.get("gbm_spread_upper")
        blend_block = (
            f"  data_source: {pg.get('data_source')}\n"
            f"  blended_expected_return: {_fmt(pg.get('blended_expected_return'))}%\n"
            f"  gbm_expected_return: {_fmt(pg.get('gbm_expected_return'))}%\n"
            f"  empirical_expected_return: {pg.get('empirical_expected_return')}\n"
            f"  gbm_spread: {_fmt(blend_lower)}% to {_fmt(blend_upper)}%\n"
            f"  empirical_spread: {pg.get('empirical_lower')} to {pg.get('empirical_upper')}\n"
            f"  blend_formula: {pg.get('job2_blend_weight', 0):.0%} empirical + "
            f"{pg.get('gbm_blend_weight', 1.0):.0%} GBM "
            f"({int(pg.get('similar_days_used', 0))} similar days → dynamic weight)"
        )

        _similar_days_used = int(pg.get("similar_days_used") or 0)
        _job2_weight = float(pg.get("job2_blend_weight") or 0)

        if _similar_days_used >= 20 and _job2_weight >= 0.6:
            _blend_interp = (
                "High empirical weight — model has strong historical precedent. "
                "Empirical distribution is the more reliable range estimate."
            )
        elif _similar_days_used >= 10:
            _blend_interp = (
                "Moderate empirical weight — reasonable historical sample. "
                "Blend is balanced between theory and precedent."
            )
        else:
            _blend_interp = (
                "Low empirical weight — limited historical precedent. "
                "GBM theoretical distribution dominates. Wider uncertainty."
            )

        blend_block += (
            f"\n  blend_interpretation: {_blend_interp}\n"
            f"  INSTRUCTION: Use blend_interpretation when describing what "
            f"the blended forecast means for signal confidence in section 3."
        )
    else:
        blend_block = "UNAVAILABLE"

    # -------------------------------------------------------------------------
    # ANOMALY BLOCKS (specific fields only)
    # -------------------------------------------------------------------------
    if ba:
        dd = ba.get("detection_breakdown") or {}
        _NUMERIC_DD_KEYS = {
            "price_velocity", "volume_anomaly",
            "news_divergence", "momentum_score",
        }
        dd_formatted = "\n".join(
            f"    {k}: {round(float(v), 4) if isinstance(v, (int, float)) else str(v)[:60]}"
            for k, v in dd.items()
            if k in _NUMERIC_DD_KEYS
        )
        ba_block = (
            f"  risk_level: {ba.get('risk_level')}\n"
            f"  pump_and_dump_score: {ba.get('pump_and_dump_score')}\n"
            f"  trading_recommendation: {ba.get('trading_recommendation')}\n"
            f"  behavioral_summary: {ba.get('behavioral_summary')}\n"
            f"  primary_risk_factors: {ba.get('primary_risk_factors')}\n"
            f"  alerts: {ba.get('alerts')}\n"
            f"  key_detection_metrics:\n{dd_formatted}"
        )
    else:
        ba_block = "UNAVAILABLE"

    if ca:
        ca_block = (
            f"  early_risk_level: {ca.get('early_risk_level')}\n"
            f"  anomaly_flags: {ca.get('anomaly_flags')}\n"
            f"  suspicious_patterns: {ca.get('suspicious_patterns')}"
        )
    else:
        ca_block = "UNAVAILABLE"

    # -------------------------------------------------------------------------
    # SIGNAL QUALITY METRICS BLOCK
    # -------------------------------------------------------------------------
    tw: Dict = sc.get("trustworthiness_breakdown") or {}
    hit_rates_block = "\n".join([
        f"  {stream}: hit_rate={tw.get('stream_hit_rates', {}).get(stream, {}).get('hit_rate')} "
        f"using_prior={tw.get('stream_hit_rates', {}).get(stream, {}).get('using_prior')}"
        for stream in ("technical", "sentiment", "market", "related_news")
    ])

    # -------------------------------------------------------------------------
    # ASSEMBLE
    # -------------------------------------------------------------------------
    return f"""Generate a technical research report for the following stock analysis.
Use markdown (##) headers. Follow this section order exactly — do not skip, merge, or add sections:
1. Executive Summary (signal, key conviction drivers, any Job1/Job2 disagreement)
2. Signal Decomposition & Quality (stream scores, signal_strength, trustworthiness, hit rates)
3. Technical & Quantitative Analysis (technical indicators, Monte Carlo, price targets, blending)
4. Sentiment & News Analysis (all three streams, credibility, Node 8 learning adjustment)
5. Market Context & Risk (sector, correlation, VIX, beta)
6. Anomaly Detection (Node 9A content, Node 9B behavioral, pump score, trading_safe)
7. Methodology Notes (weights, fallback flags, data gaps, caveats)

TICKER: {ticker}

--- SIGNAL ---
  final_signal: {sc.get('final_signal')}
  final_confidence: {sc.get('final_confidence')}
  final_score: {sc.get('final_score')}
  signal_agreement: {sc.get('signal_agreement')}/4
  streams_missing: {sc.get('streams_missing') or 'none'}

--- STREAM SCORES ---
{stream_table}

--- TECHNICAL INDICATORS (Node 4) ---
{ti_block}

--- ADAPTIVE WEIGHTS (Node 11) ---
  technical_weight: {aw.get('technical_weight')}
  stock_news_weight: {aw.get('stock_news_weight')}
  market_news_weight: {aw.get('market_news_weight')}
  related_news_weight: {aw.get('related_news_weight')}
  hold_threshold_pct: {aw.get('hold_threshold_pct')}
  streams_reliable: {aw.get('streams_reliable')}
  weights_are_fallback: {bc.get('weights_are_fallback')}

--- SIGNAL QUALITY METRICS ---
  signal_strength: {sc.get('signal_strength')}/100
  trustworthiness: {sc.get('trustworthiness')}
  insufficient_history: {tw.get('insufficient_history')}
  reliability_score: {tw.get('reliability_score')}
  agreement_score: {tw.get('agreement_score')}
  pattern_score: {tw.get('pattern_score')}
  stream_hit_rates:
{hit_rates_block}

--- SENTIMENT BREAKDOWN (Node 5) ---
{sb_block}

--- SENTIMENT ADJUSTMENT (Node 8) ---
{sa_block}

--- NEWS IMPACT VERIFICATION (Node 8 learning) ---
{niv_block}

--- MARKET CONTEXT (Node 6) ---
{mc_ctx_block}

--- MONTE CARLO (Node 7) ---
{mc_block}

--- PRICE TARGETS ---
  current_price: {pt.get('current_price')}
  forecasted_price: {pt.get('forecasted_price')}
  price_range_lower: {pt.get('price_range_lower')}
  price_range_upper: {pt.get('price_range_upper')}
  expected_return_pct: {pt.get('expected_return_pct')}

--- HISTORICAL PATTERN MATCHING (Job 2) ---
{pattern_block}

--- PREDICTION BLENDING ---
{blend_block}

--- RISK SUMMARY ---
  overall_risk_level: {rs.get('overall_risk_level')}
  trading_safe: {rs.get('trading_safe')}
  trading_recommendation: {rs.get('trading_recommendation')}
  (full detail — pump_and_dump_score, behavioral_summary, primary_risk_factors, alerts, detection_breakdown — is in the BEHAVIORAL ANOMALY section below)

--- CONTENT ANOMALY (Node 9A) ---
{ca_block}

--- BEHAVIORAL ANOMALY (Node 9B) ---
{ba_block}
"""


# ============================================================================
# FALLBACK REPORT (used when LLM call fails)
# ============================================================================

def _build_fallback_report(
    sc: Optional[Dict[str, Any]],
    ticker: str,
) -> str:
    """
    Produce a minimal templated technical report without an LLM call.

    Args:
        sc:     signal_components dict (may be None).
        ticker: Stock ticker symbol.

    Returns:
        A markdown string suitable for technical_explanation.
    """
    if sc is None:
        return (
            f"## {ticker} — Technical Report Unavailable\n\n"
            "The analysis pipeline did not complete successfully. "
            "`signal_components` was not produced by Node 12.\n\n"
            "**Action:** Check Node 12 logs for errors."
        )

    signal     = sc.get("final_signal", "HOLD")
    confidence = float(sc.get("final_confidence") or 0.0)
    score      = float(sc.get("final_score") or 0.0)
    rs         = sc.get("risk_summary") or {}

    return (
        f"## {ticker} — Technical Report (LLM Unavailable)\n\n"
        f"**Signal:** {signal} | **Confidence:** {confidence:.4f} | "
        f"**Score:** {score:+.4f}\n\n"
        f"**Risk Level:** {rs.get('overall_risk_level', 'UNKNOWN')} | "
        f"**Trading Safe:** {rs.get('trading_safe', True)}\n\n"
        "_LLM call failed — full narrative report could not be generated. "
        "Raw signal data above is accurate._"
    )


# ============================================================================
# MAIN NODE FUNCTION
# ============================================================================

def technical_explanation_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 14: Generate detailed technical research report via Anthropic Claude.

    Extracts all tier data from state (all fields guarded with .get()),
    assembles a structured prompt, calls the LLM, and writes the markdown
    report to state["technical_explanation"]. Falls back to a templated
    string on any failure — never raises.

    Args:
        state: LangGraph state dict.

    Returns:
        Updated state with technical_explanation populated.
    """
    start_time = datetime.now()
    ticker: str = state.get("ticker", "UNKNOWN")

    logger.info(f"Node 14: Generating technical explanation for {ticker}")

    # =========================================================================
    # STEP 1: Extract all state fields — guarded, never raw state to LLM
    # =========================================================================
    sc:  Optional[Dict[str, Any]] = state.get("signal_components")
    ti:  Dict[str, Any]           = state.get("technical_indicators") or {}
    aw:  Dict[str, Any]           = state.get("adaptive_weights") or {}
    sb:  Dict[str, Any]           = state.get("sentiment_breakdown") or {}
    sa:  Dict[str, Any]           = state.get("sentiment_analysis") or {}
    niv: Dict[str, Any]           = state.get("news_impact_verification") or {}
    mc_ctx: Dict[str, Any]        = state.get("market_context") or {}
    mc:  Dict[str, Any]           = state.get("monte_carlo_results") or {}
    ba:  Dict[str, Any]           = state.get("behavioral_anomaly_detection") or {}
    ca:  Dict[str, Any]           = state.get("content_analysis_summary") or {}
    related_companies: List[Dict[str, Any]] = state.get("related_companies") or []

    # =========================================================================
    # STEP 2: Guard — signal_components is required
    # =========================================================================
    if sc is None:
        logger.error("Node 14: signal_components is None — using fallback report")
        state.setdefault("errors", []).append(
            "Node 14: signal_components missing, used fallback report"
        )
        state["technical_explanation"] = _build_fallback_report(None, ticker)
        state.setdefault("node_execution_times", {})["node_14"] = (
            datetime.now() - start_time
        ).total_seconds()
        return state

    # =========================================================================
    # STEP 3: Build prompt
    # =========================================================================
    user_prompt = _build_user_prompt(
        ticker=ticker,
        sc=sc,
        ti=ti,
        aw=aw,
        sb=sb,
        sa=sa,
        niv=niv,
        mc_ctx=mc_ctx,
        mc=mc,
        ba=ba,
        ca=ca,
        related_companies=related_companies,
    )

    logger.debug(f"  Prompt built ({len(user_prompt)} chars), calling Anthropic...")

    # =========================================================================
    # STEP 4: LLM call
    # =========================================================================
    try:
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is not set in environment")

        llm_start = datetime.now()
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0.35,
        )
        llm_elapsed = (datetime.now() - llm_start).total_seconds()

        report: str = response.content[0].text.strip()
        logger.info(
            f"  Anthropic call succeeded in {llm_elapsed:.2f}s "
            f"({len(report.split())} words)"
        )

    except Exception as exc:
        logger.error(f"  Anthropic call failed for {ticker}: {exc}")
        state.setdefault("errors", []).append(
            f"Node 14 (technical explanation) LLM failed: {exc}"
        )
        report = _build_fallback_report(sc, ticker)

    # =========================================================================
    # STEP 5: Write to state
    # =========================================================================
    state["technical_explanation"] = report
    state.setdefault("node_execution_times", {})["node_14"] = (
        datetime.now() - start_time
    ).total_seconds()

    logger.info(f"Node 14 completed in {state['node_execution_times']['node_14']:.3f}s")
    return state
