"""
Node 13: Beginner Explanation (LLM)

Generates a plain-English explanation of the final trading signal for a
non-expert retail investor. Calls the Anthropic Claude LLM with a hardcoded
system prompt and a dynamically assembled user prompt built entirely from
state data — no external knowledge is used.

Reads from state:
  PRIMARY   — signal_components  (Node 12 output: signal, streams, risk, pattern, prices)
  SECONDARY — sentiment_breakdown (Node 5: top articles, per-stream labels)
  TERTIARY  — technical_indicators, monte_carlo_results, ticker (direct fields)

Writes to state:
  beginner_explanation     — plain string, 250-400 words
  node_execution_times     — records "node_13" duration

Runs AFTER:  Node 12 (signal_generation)
Runs BEFORE: Node 14 (technical_explanation) / Node 15 (dashboard)
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import anthropic

from src.utils.config import ANTHROPIC_API_KEY
from src.utils.logger import get_node_logger

logger = get_node_logger("node_13")


# ============================================================================
# CONSTANTS
# ============================================================================

CLAUDE_MODEL: str = "claude-sonnet-4-5"
MAX_TOKENS: int = 900   # ~500 words with headroom

# Score thresholds for translating raw_score to plain English
STRONG_POSITIVE: float = 0.4
MILD_POSITIVE:   float = 0.1
MILD_NEGATIVE:   float = -0.1
STRONG_NEGATIVE: float = -0.4

DISCLAIMER: str = (
    "This is an automated analysis, not financial advice. "
    "Always conduct your own research before making investment decisions."
)


# ============================================================================
# SYSTEM PROMPT (hardcoded — never changes between runs)
# ============================================================================

SYSTEM_PROMPT: str = """You are a financial analysis assistant generating a plain-English \
explanation of a stock trading recommendation for a non-expert retail investor.

Rules you must follow:
- Use ONLY the data provided in the user message. Do not add any facts, news, or context \
from your training data or general knowledge about the company or market.
- Never mention company products, leadership, history, or events unless they appear \
verbatim in the provided article titles.
- If a data field is marked as unavailable or None, omit that section entirely. \
Do not estimate or infer missing values.
- Write in plain English. Avoid financial jargon. If you must use a term like RSI, \
explain it in one sentence immediately after.
- Keep the response between 350 and 500 words.
- Signal Strength tells the reader how loudly the indicators are agreeing — explain \
it as a simple score out of 100. Trustworthiness tells the reader how often this \
type of signal has been right historically — explain it as a percentage if available, \
or note that the system is still building its track record if marked insufficient.
- Use the exact 7-section structure provided. Do not add sections or change their order.
- Do not give specific investment advice beyond what the signal says.
- Always end with the disclaimer exactly as provided — word for word.
- When signal_strength is below 40, the opening of section 1 (Headline) must \
convey that evidence is mixed or indicators are conflicting. Use language like \
'signals are mixed', 'no clear direction yet', or 'indicators are not strongly \
aligned'. Never open a low-conviction HOLD with language that implies the \
recommendation is clear or confident.
- Never use the phrase 'current pricing data isn't available' or similar — \
if data is missing, describe the concept without referencing the gap.
- The risk section must always distinguish between manipulation risk (is this \
stock being artificially traded?) and investment risk (how volatile is this \
stock likely to be?). These are different things and beginners confuse them."""


# ============================================================================
# HELPER: TRANSLATE RAW SCORE TO PLAIN LANGUAGE
# ============================================================================

def _score_to_label(raw_score: float, stream: str) -> str:
    """
    Convert a continuous [-1, +1] score to a plain-English direction phrase.

    Args:
        raw_score: The stream's raw_score from signal_components.
        stream:    One of 'technical', 'sentiment', 'market', 'monte_carlo'.

    Returns:
        A short readable phrase describing the direction and strength.
    """
    labels = {
        "technical": {
            "strong_pos": "Technical momentum is strongly bullish",
            "mild_pos":   "Technical indicators lean slightly bullish",
            "neutral":    "Technical indicators are mixed",
            "mild_neg":   "Technical indicators lean slightly bearish",
            "strong_neg": "Technical momentum is strongly bearish",
        },
        "sentiment": {
            "strong_pos": "News sentiment is strongly positive",
            "mild_pos":   "News sentiment is slightly positive",
            "neutral":    "News sentiment is neutral",
            "mild_neg":   "News sentiment is slightly negative",
            "strong_neg": "News sentiment is strongly negative",
        },
        "market": {
            "strong_pos": "The broader market environment is clearly supportive",
            "mild_pos":   "The broader market environment is mildly supportive",
            "neutral":    "The broader market environment is neutral",
            "mild_neg":   "The broader market environment is slightly unfavourable",
            "strong_neg": "The broader market environment is clearly unfavourable",
        },
        "monte_carlo": {
            "strong_pos": "Our price simulation strongly favours upward movement",
            "mild_pos":   "Our price simulation leans upward",
            "neutral":    "Our price simulation shows no clear direction",
            "mild_neg":   "Our price simulation leans downward",
            "strong_neg": "Our price simulation strongly favours downward movement",
        },
    }
    bucket = labels.get(stream, {})
    if raw_score >= STRONG_POSITIVE:
        return bucket.get("strong_pos", f"{stream} is strongly positive")
    if raw_score >= MILD_POSITIVE:
        return bucket.get("mild_pos",   f"{stream} is mildly positive")
    if raw_score > MILD_NEGATIVE:
        return bucket.get("neutral",    f"{stream} is neutral")
    if raw_score > STRONG_NEGATIVE:
        return bucket.get("mild_neg",   f"{stream} is mildly negative")
    return bucket.get("strong_neg", f"{stream} is strongly negative")


# ============================================================================
# USER PROMPT BUILDER
# ============================================================================

def _build_user_prompt(
    sc: Dict[str, Any],
    sb: Dict[str, Any],
    ticker: str,
    macro_factors: Optional[List[Dict[str, Any]]] = None,
    related_companies: Optional[List[Dict[str, Any]]] = None,
    mc_ctx: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Assemble the dynamic data block from pre-extracted state dicts.

    Args:
        sc:                signal_components dict from Node 12.
        sb:                sentiment_breakdown dict from Node 5 (may be empty).
        ticker:            Stock ticker symbol.
        macro_factors:     identified_factors list from Node 6 (may be None/empty).
        related_companies: related_companies list from Node 3 (may be None/empty).
        mc_ctx:            market_context dict from Node 6 (may be None/empty).

    Returns:
        Formatted user prompt string ready to send to the LLM.
    """
    pp: Dict[str, Any] = sc.get("pattern_prediction") or {}
    pt: Dict[str, Any] = sc.get("price_targets") or {}
    rs: Dict[str, Any] = sc.get("risk_summary") or {}
    ss: Dict[str, Any] = sc.get("stream_scores") or {}
    bc: Dict[str, Any] = sc.get("backtest_context") or {}

    final_signal:      str   = sc.get("final_signal", "HOLD")
    final_confidence:  float = float(sc.get("final_confidence") or 0.0)
    signal_strength:   int   = int(sc.get("signal_strength") or 0)
    trustworthiness:   float = float(sc.get("trustworthiness") or 0.0)
    tw_breakdown: Dict        = sc.get("trustworthiness_breakdown") or {}
    insufficient_history: bool = tw_breakdown.get("insufficient_history", True)
    signal_agreement:  int   = int(sc.get("signal_agreement") or 0)
    streams_missing:   List  = sc.get("streams_missing") or []

    # --- STREAM DIRECTIONS ---
    stream_lines: List[str] = []
    for stream in ("technical", "sentiment", "market", "related_news"):
        data = ss.get(stream) or {}
        raw  = float(data.get("raw_score") or 0.0)
        wt   = float(data.get("weight") or 0.25)
        label = _score_to_label(raw, stream)
        stream_lines.append(f"- {label} (importance: {wt:.0%})")
    streams_block = "\n".join(stream_lines)

    # --- TOP ARTICLES ---
    # sentiment_breakdown keys are 'stock'/'market'/'related' — no top-level 'top_articles'.
    # Node 05 already pre-selected top 3 per stream; collect the most credible ones across
    # all streams and take the top 3 by credibility_weight for the beginner explanation.
    top_articles_block = ""
    if sb:
        all_stream_articles = (
            (sb.get("stock")   or {}).get("top_articles", []) +
            (sb.get("market")  or {}).get("top_articles", []) +
            (sb.get("related") or {}).get("top_articles", [])
        )
        # Sort by credibility_weight descending so the most reliable headlines come first
        all_stream_articles = sorted(
            all_stream_articles,
            key=lambda a: float(a.get("credibility_weight") or 0.0),
            reverse=True,
        )
        lines = [
            f'- "{a.get("title", "")}" ({a.get("source", "unknown source")})'
            for a in all_stream_articles[:3]
            if a.get("title")
        ]
        if lines:
            top_articles_block = "TOP NEWS HEADLINES (use to explain sentiment):\n" + "\n".join(lines)

    # --- PATTERN SECTION ---
    if pp.get("sufficient_data"):
        agreement = pp.get("agreement_with_job1", "neutral")
        caution = ""
        if agreement == "disagree":
            caution = (
                f"\nCAUTION: Historical pattern does NOT support the {final_signal} signal — "
                "add a caution note in the Historical Pattern section."
            )
        pattern_block = f"""HISTORICAL PATTERN DATA:
- Similar past trading days found: {pp.get('similar_days_found')}
- Probability price goes up next week: {float(pp.get('prob_up', 0)):.0%}
- Expected 7-day return: {float(pp.get('expected_return_7d', 0)):+.1f}%
- Worst case (10th percentile): {float(pp.get('worst_case_7d', 0)):+.1f}%
- Best case (90th percentile): {float(pp.get('best_case_7d', 0)):+.1f}%
- Agreement with main signal: {agreement}{caution}"""
    else:
        pattern_block = "HISTORICAL PATTERN DATA: Insufficient data — omit section 4 entirely."

    # --- PRICE TARGETS ---
    current_price    = pt.get("current_price")
    forecasted_price = pt.get("forecasted_price")
    range_lower      = pt.get("price_range_lower")
    range_upper      = pt.get("price_range_upper")

    if current_price is not None:
        price_block = (
            f"- Current price: ${float(current_price):.2f}\n"
            f"- Forecasted price (30-day): "
            f"{'${:.2f}'.format(float(forecasted_price)) if forecasted_price is not None else 'unavailable'}\n"
            f"- Expected range: "
            f"{'${:.2f} – ${:.2f}'.format(float(range_lower), float(range_upper)) if range_lower is not None and range_upper is not None else 'unavailable'}"
        )
        price_block += (
            "\nINSTRUCTION: When describing the price range in section 5, call it a "
            "'range of simulated outcomes, not a price target'. Emphasise that most "
            "real outcomes will land near the middle of this range. The outer edges "
            "represent unlikely but statistically possible scenarios. The forecasted "
            "price is the expected midpoint, not a guarantee."
        )
    else:
        price_block = "Price targets: unavailable — omit section 5."

    # --- RISK ---
    trading_safe = rs.get("trading_safe", True)
    pump_score   = int(rs.get("pump_and_dump_score") or 0)

    # Volatility risk: derived from beta and VIX — separate from manipulation/pump risk
    _mc = mc_ctx or {}
    beta = float((_mc.get("market_correlation_profile") or {}).get("beta_calculated") or 1.0)
    vix  = float((_mc.get("market_regime") or {}).get("vix_level") or 20.0)
    if beta > 1.3 and vix > 22:
        volatility_risk = "HIGH"
    elif beta > 1.0 or vix > 22:
        volatility_risk = "MODERATE"
    else:
        volatility_risk = "LOW"
    alerts       = rs.get("alerts") or []
    # Truncate to 200 chars — this is a free-form Node 9B narrative; the full version
    # is for Node 14. Beginners only need the gist.
    raw_summary  = rs.get("behavioral_summary") or ""
    beh_summary  = (raw_summary[:200] + "…") if len(raw_summary) > 200 else raw_summary

    risk_note = ""
    if not trading_safe:
        risk_note = "\nWARNING: trading_safe=False — move the Risk section to position 2 and warn prominently."
    if pump_score > 50:
        risk_note += f"\nWARNING: pump_and_dump_score={pump_score} — explicitly warn the user about unusual trading patterns."

    alerts_str = "\n".join(f"  • {a}" for a in alerts[:3]) if alerts else "  None"

    # --- STREAMS MISSING NOTE ---
    missing_note = ""
    if streams_missing:
        missing_note = f"\nNOTE: The following data streams were unavailable: {', '.join(streams_missing)}. Briefly mention this."

    # --- ADAPTIVE WEIGHTS NOTE ---
    weights_note = ""
    if bc.get("weights_are_fallback"):
        weights_note = "\nNOTE: Historical learning data was insufficient — do not mention 'historical learning' in the explanation."

    # --- COMMODITY & MACRO FACTORS BLOCK ---
    commodity_section = ""
    if macro_factors:
        trend_arrow = {"UP": "↑", "DOWN": "↓", "FLAT": "→"}
        lines: List[str] = []
        for f in macro_factors:
            if not isinstance(f, dict):
                continue
            name     = f.get("factor_name", "Unknown")
            exp_type = f.get("exposure_type", "")
            expl     = f.get("exposure_explanation", "")
            price    = f.get("current_price")
            trend    = f.get("price_trend")
            price_str = f"${price:,.2f}" if price is not None else "price unavailable"
            trend_str = (
                f" {trend_arrow.get(trend, '')} {trend}" if trend else ""
            )
            lines.append(f"- {name} ({exp_type}): {price_str}{trend_str} — {expl}")
        if lines:
            commodity_section = (
                "\nCOMMODITY & MACRO FACTOR PRICES"
                " (incorporate into section 3 — What's Driving This):\n"
                + "\n".join(lines)
                + "\nInterpretation guide: For factors where a live price is available, mention it. "
                "For factors where price = N/A, describe only the directional relationship between "
                "that factor and the company's business — do NOT mention that pricing data is "
                "unavailable, just omit the price reference and explain the relationship in plain "
                "English (e.g. 'Higher defense spending typically means more government contracts "
                "for this company'). The goal is useful context, not a data dump.\n"
            )

    # --- PEER COMPANIES BLOCK ---
    peers_section = ""
    if related_companies:
        rel_label = {
            "COMPETITOR": "Direct Competitor",
            "SUPPLIER": "Supplier",
            "CUSTOMER": "Customer",
            "SAME_SECTOR": "Same Sector",
            "PARTNER": "Partner",
        }
        peer_lines: List[str] = []
        for c in related_companies:
            if not isinstance(c, dict):
                continue
            t = c.get("ticker", "")
            rel = c.get("relationship", "SAME_SECTOR")
            reason = c.get("reason", "")
            label = rel_label.get(rel, rel.replace("_", " ").title())
            peer_lines.append(f"- {t} ({label}): {reason}")
        if peer_lines:
            peers_section = (
                "\nPEER COMPANIES (mention in section 3 — What's Driving This,"
                " explain how each peer's relationship affects the stock):\n"
                + "\n".join(peer_lines)
                + "\n"
            )

    # --- CONVICTION LABEL (FIX 3) ---
    if signal_strength < 34:
        conviction_label = "weak — indicators are pulling in different directions"
    elif signal_strength < 67:
        conviction_label = "moderate — indicators are broadly aligned"
    else:
        conviction_label = "strong — indicators are well-aligned"

    return f"""WORD BUDGET PER SECTION (treat as approximate upper bounds):
1. Headline: 15 words max
2. Signal strength & reliability: 60 words
3. What's driving this (including macro/peers): 100 words
4. Historical pattern: 60 words (omit entirely if insufficient data)
5. Price expectation: 50 words
6. Risk: 80 words — must cover BOTH manipulation risk and volatility risk
7. Disclaimer: verbatim only, no additions

Generate a beginner explanation for the following stock analysis.
Follow the 7-section structure exactly in this order:
1. Headline
2. How Strong and Reliable is This Signal
3. What's Driving This (incorporate commodity/macro factor prices and peer companies if provided)
4. Historical Pattern (skip entirely if marked insufficient)
5. Price Expectation (skip if unavailable)
6. Risk
7. Disclaimer (use the exact text provided)

TICKER: {ticker}

SIGNAL:
- Recommendation: {final_signal}
- Confidence: {final_confidence:.0%}
- Signal strength: {signal_strength}/100 ({conviction_label})
- Trustworthiness: {"building track record — not enough history yet" if insufficient_history else f"{trustworthiness:.0%} (based on historical stream accuracy)"}
- Number of indicators agreeing: {signal_agreement} out of 4{missing_note}{weights_note}

STREAM DIRECTIONS (translate these to plain English in section 3):
{streams_block}
{commodity_section}{peers_section}
{top_articles_block}

PRICE TARGETS:
{price_block}

RISK:
- Manipulation risk (pump-and-dump score): {pump_score}/100
- Investment/volatility risk: {volatility_risk} (beta={beta:.1f}, VIX={vix:.1f})
- Trading safe: {trading_safe}{risk_note}
- Active alerts:
{alerts_str}
- Behavioral summary: {beh_summary or 'none'}

INSTRUCTION: In section 6, explain BOTH risk types separately in plain English.
Manipulation risk = is the stock being artificially pumped or showing suspicious trading patterns?
Investment/volatility risk = how much could the price swing based on the stock's historical behavior relative to the market?
Never describe investment/volatility risk as LOW if volatility_risk = HIGH.
Never conflate these two concepts.

{pattern_block}

DISCLAIMER TO USE VERBATIM AS THE FINAL SENTENCE:
"{DISCLAIMER}"
"""


# ============================================================================
# FALLBACK EXPLANATION (used when LLM call fails)
# ============================================================================

def _build_fallback_explanation(
    sc: Optional[Dict[str, Any]],
    ticker: str,
) -> str:
    """
    Produce a minimal templated explanation without an LLM call.

    Used when signal_components is None or the Anthropic call raises an exception.

    Args:
        sc:     signal_components dict (may be None).
        ticker: Stock ticker symbol.

    Returns:
        A plain string suitable for beginner_explanation.
    """
    if sc is None:
        return (
            f"{ticker} — Analysis Unavailable\n\n"
            "The analysis pipeline did not complete successfully. "
            "Please try again later.\n\n"
            f"{DISCLAIMER}"
        )

    signal     = sc.get("final_signal", "HOLD")
    confidence = float(sc.get("final_confidence") or 0.0)
    rs         = sc.get("risk_summary") or {}
    risk_level = rs.get("overall_risk_level", "UNKNOWN")

    return (
        f"{ticker} — {signal} with {confidence:.0%} confidence\n\n"
        f"Our analysis recommends {signal} for {ticker}. "
        f"Current risk level is {risk_level}.\n\n"
        f"{DISCLAIMER}"
    )


# ============================================================================
# MAIN NODE FUNCTION
# ============================================================================

def beginner_explanation_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 13: Generate plain-English beginner explanation via Anthropic Claude.

    Orchestrates data extraction from state, builds a structured prompt,
    calls the LLM, and writes the result to state["beginner_explanation"].
    Falls back to a templated string on any failure — never raises.

    Args:
        state: LangGraph state dict.

    Returns:
        Updated state with beginner_explanation populated.
    """
    start_time = datetime.now()
    ticker: str = state.get("ticker", "UNKNOWN")

    logger.info(f"Node 13: Generating beginner explanation for {ticker}")

    # =========================================================================
    # STEP 1: Extract state fields (all guarded — fail early, not inside prompt)
    # =========================================================================
    sc: Optional[Dict[str, Any]] = state.get("signal_components")
    sb: Dict[str, Any]           = state.get("sentiment_breakdown") or {}
    mc_ctx: Dict[str, Any]       = state.get("market_context") or {}
    macro_factors: List[Dict[str, Any]] = (
        mc_ctx.get("macro_factor_exposure", {}).get("identified_factors", []) or []
    )
    related_companies: List[Dict[str, Any]] = state.get("related_companies") or []

    # =========================================================================
    # STEP 2: Guard — signal_components is required
    # =========================================================================
    if sc is None:
        logger.error("Node 13: signal_components is None — using fallback explanation")
        state.setdefault("errors", []).append(
            "Node 13: signal_components missing, used fallback explanation"
        )
        state["beginner_explanation"] = _build_fallback_explanation(None, ticker)
        state.setdefault("node_execution_times", {})["node_13"] = (
            datetime.now() - start_time
        ).total_seconds()
        return state

    # =========================================================================
    # STEP 3: Build prompts
    # =========================================================================
    user_prompt = _build_user_prompt(sc, sb, ticker, macro_factors, related_companies, mc_ctx)

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
            temperature=0.55,
        )
        llm_elapsed = (datetime.now() - llm_start).total_seconds()

        explanation: str = response.content[0].text.strip()
        logger.info(
            f"  Anthropic call succeeded in {llm_elapsed:.2f}s "
            f"({len(explanation.split())} words)"
        )

    except Exception as exc:
        logger.error(f"  Anthropic call failed for {ticker}: {exc}")
        state.setdefault("errors", []).append(
            f"Node 13 (beginner explanation) LLM failed: {exc}"
        )
        explanation = _build_fallback_explanation(sc, ticker)

    # =========================================================================
    # STEP 5: Write to state
    # =========================================================================
    state["beginner_explanation"] = explanation
    state.setdefault("node_execution_times", {})["node_13"] = (
        datetime.now() - start_time
    ).total_seconds()

    logger.info(f"Node 13 completed in {state['node_execution_times']['node_13']:.3f}s")
    return state
