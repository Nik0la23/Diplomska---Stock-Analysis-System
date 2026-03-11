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
MAX_TOKENS: int = 700   # ~400 words with headroom

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
- Keep the response between 250 and 400 words.
- Use the exact 7-section structure provided. Do not add sections or change their order.
- Do not give specific investment advice beyond what the signal says.
- Always end with the disclaimer exactly as provided — word for word."""


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
) -> str:
    """
    Assemble the dynamic data block from pre-extracted state dicts.

    Args:
        sc:     signal_components dict from Node 12.
        sb:     sentiment_breakdown dict from Node 5 (may be empty).
        ticker: Stock ticker symbol.

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
    signal_agreement:  int   = int(sc.get("signal_agreement") or 0)
    streams_missing:   List  = sc.get("streams_missing") or []

    # --- STREAM DIRECTIONS ---
    stream_lines: List[str] = []
    for stream in ("technical", "sentiment", "market", "monte_carlo"):
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
    else:
        price_block = "Price targets: unavailable — omit section 5."

    # --- RISK ---
    risk_level  = rs.get("overall_risk_level", "UNKNOWN")
    trading_safe = rs.get("trading_safe", True)
    pump_score   = int(rs.get("pump_and_dump_score") or 0)
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

    return f"""Generate a beginner explanation for the following stock analysis.
Follow the 7-section structure exactly in this order:
1. Headline
2. Signal Strength Context
3. What's Driving This
4. Historical Pattern (skip entirely if marked insufficient)
5. Price Expectation (skip if unavailable)
6. Risk
7. Disclaimer (use the exact text provided)

TICKER: {ticker}

SIGNAL:
- Recommendation: {final_signal}
- Confidence: {final_confidence:.0%}
- Number of indicators agreeing: {signal_agreement} out of 4{missing_note}{weights_note}

STREAM DIRECTIONS (translate these to plain English in section 3):
{streams_block}

{top_articles_block}

PRICE TARGETS:
{price_block}

RISK:
- Overall risk level: {risk_level}
- Trading safe: {trading_safe}
- Pump-and-dump score: {pump_score}/100{risk_note}
- Active alerts:
{alerts_str}
- Behavioral summary: {beh_summary or 'none'}

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
    user_prompt = _build_user_prompt(sc, sb, ticker)

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
            temperature=0.3,   # low temperature → consistent, factual output
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
