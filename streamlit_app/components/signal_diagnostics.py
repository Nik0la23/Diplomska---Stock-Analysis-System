"""
Signal Diagnostics Widget

Shows data availability and signal counts per stream after the pipeline runs.
Use this to spot which stream is sparse or producing too few directional signals
(BUY/SELL) to be statistically meaningful.
"""

from typing import Any, Dict, Optional
import streamlit as st


_MIN_PRICE_ROWS_REGRESSION = 60   # Node 4 needs ≥60 rows for Ridge regression
_MIN_PRICE_ROWS_INDICATORS = 50   # Node 4 fallback needs ≥50 rows
_MIN_SIGNALS_SUFFICIENT    = 20   # Node 10 MIN_SIGNALS_FOR_SUFFICIENCY


def _badge(label: str, style: str) -> str:
    """
    style: 'green' | 'yellow' | 'red' | 'grey'
    Returns an inline-styled HTML badge consistent with the app's aesthetic.
    """
    styles = {
        "green":  "color:#065f46;background:#ecfdf5",
        "yellow": "color:#92400e;background:#fffbeb",
        "red":    "color:#991b1b;background:#fef2f2",
        "grey":   "color:#374151;background:#f3f4f6",
    }
    css = styles.get(style, styles["grey"])
    return (
        f'<span style="{css};padding:2px 8px;border-radius:4px;'
        f'font-size:11px;font-weight:600">{label}</span>'
    )


def render_signal_diagnostics(diag: Dict[str, Any]) -> None:
    """
    Render the signal diagnostics section.

    Section 1 — Data availability: raw counts per source before backtesting.
    Section 2 — Backtest signal breakdown: BUY/SELL/HOLD per stream with
                sufficiency status after Node 10.

    Args:
        diag: dashboard_data["diagnostics"] dict produced by Node 15.
    """
    st.markdown(
        '<div class="section-header">Signal diagnostics</div>',
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------ #
    # Unpack pre-computed counts from Node 15                             #
    # ------------------------------------------------------------------ #
    price_rows       = diag.get("price_rows", 0)
    stock_articles   = diag.get("stock_articles", 0)
    market_articles  = diag.get("market_articles", 0)
    related_articles = diag.get("related_articles", 0)
    br               = diag.get("backtest_results") or {}

    # ------------------------------------------------------------------ #
    # Section 1: Data availability metrics                                 #
    # ------------------------------------------------------------------ #
    st.markdown(
        '<div style="font-size:12px;font-weight:600;color:#555;'
        'text-transform:uppercase;letter-spacing:.05em;margin-bottom:.5rem">'
        'Data availability</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    if price_rows >= _MIN_PRICE_ROWS_REGRESSION:
        c1.metric("Price rows", price_rows, help=f"Ridge regression active (needs ≥{_MIN_PRICE_ROWS_REGRESSION})")
    elif price_rows >= _MIN_PRICE_ROWS_INDICATORS:
        c1.metric("Price rows", price_rows,
                  delta=f"Heuristic only — needs ≥{_MIN_PRICE_ROWS_REGRESSION} for ML",
                  delta_color="off")
    else:
        c1.metric("Price rows", price_rows,
                  delta="Insufficient for technical analysis",
                  delta_color="inverse")

    c2.metric("Stock news",   stock_articles,   help="Cleaned articles entering sentiment analysis")
    c3.metric("Market news",  market_articles,  help="Broad market articles")
    c4.metric("Related news", related_articles, help="Peer / competitor articles")

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ------------------------------------------------------------------ #
    # Section 2: Backtest signal breakdown table                           #
    # ------------------------------------------------------------------ #
    st.markdown(
        '<div style="font-size:12px;font-weight:600;color:#555;'
        'text-transform:uppercase;letter-spacing:.05em;margin-bottom:.5rem">'
        'Backtest signal counts &nbsp;'
        '<span style="font-size:10px;font-weight:400;color:#aaa">'
        '(last 180 days)</span></div>',
        unsafe_allow_html=True,
    )

    def _ic_cell(metrics: Optional[Dict[str, Any]], note: str = "") -> str:
        """Render the IC score cell with colour coding and p-value tooltip."""
        if metrics is None:
            return '<td style="color:#aaa;font-size:12px;padding:5px 8px">—</td>'
        ic  = metrics.get("ic_score")
        ic_p = metrics.get("ic_p_value")
        sig  = metrics.get("ic_significant", False)
        if ic is None:
            return '<td style="color:#aaa;font-size:12px;padding:5px 8px">N/A</td>'
        if sig:
            color = "#065f46" if ic > 0 else "#991b1b"
            star  = " ✦"
        else:
            color = "#92400e" if abs(ic) > 0.03 else "#999"
            star  = ""
        p_str = f" p={ic_p:.3f}" if ic_p is not None else ""
        note_html = (
            f'<span style="font-size:10px;color:#aaa;margin-left:3px">{note}</span>'
            if note else ""
        )
        return (
            f'<td style="font-family:\'DM Mono\',monospace;font-size:12px;'
            f'color:{color};padding:5px 8px" title="IC={ic:+.4f}{p_str}">'
            f'{ic:+.3f}{star}{note_html}</td>'
        )

    def _stream_row(
        name: str,
        data_pts: str,
        metrics: Optional[Dict[str, Any]],
        ic_note: str = "",
    ) -> str:
        if metrics is None:
            return (
                f"<tr>"
                f'<td style="color:#333;font-size:12px">{name}</td>'
                f'<td style="color:#aaa;font-size:12px">{data_pts}</td>'
                f'<td colspan="6" style="color:#aaa;font-size:12px">—</td>'
                f'<td>{_badge("No data", "red")}</td>'
                f"</tr>"
            )

        total  = metrics.get("total_days_evaluated", 0)
        buys   = metrics.get("buy_count",   0)
        sells  = metrics.get("sell_count",  0)
        holds  = metrics.get("hold_count",  0)
        sigs   = metrics.get("signal_count", buys + sells)
        is_suf = metrics.get("is_sufficient",     False)
        is_sig = metrics.get("is_significant",    False)
        is_ant = metrics.get("is_anti_predictive", False)

        if is_ant:
            badge = _badge("Anti-pred", "red")
        elif not is_suf:
            badge = _badge(f"Sparse ({sigs})", "red")
        elif is_sig:
            badge = _badge("Significant", "green")
        else:
            badge = _badge("Near-random", "yellow")

        return (
            f"<tr>"
            f'<td style="color:#333;font-size:12px;padding:5px 8px">{name}</td>'
            f'<td style="color:#777;font-size:12px;padding:5px 8px">{data_pts}</td>'
            f'<td style="font-family:\'DM Mono\',monospace;font-size:12px;padding:5px 8px">{total}</td>'
            f'<td style="font-family:\'DM Mono\',monospace;font-size:12px;color:#065f46;padding:5px 8px">{buys}</td>'
            f'<td style="font-family:\'DM Mono\',monospace;font-size:12px;color:#991b1b;padding:5px 8px">{sells}</td>'
            f'<td style="font-family:\'DM Mono\',monospace;font-size:12px;color:#92400e;padding:5px 8px">{holds}</td>'
            f'<td style="font-family:\'DM Mono\',monospace;font-size:12px;padding:5px 8px">{sigs}</td>'
            + _ic_cell(metrics, ic_note) +
            f'<td style="padding:5px 8px">{badge}</td>'
            f"</tr>"
        )

    # Use the explicit ic_used_idiosyncratic flag stored by calculate_news_ic
    market_metrics_raw = br.get("market_news") or {}
    market_ic_note = "idiosyn." if market_metrics_raw.get("ic_used_idiosyncratic") else "raw ret"

    rows_html = "".join([
        _stream_row("Technical",    f"{price_rows} price rows",   br.get("technical")),
        _stream_row("Stock news",   f"{stock_articles} articles",  br.get("stock_news")),
        _stream_row("Market news",  f"{market_articles} articles", br.get("market_news"),
                    ic_note=market_ic_note),
        _stream_row("Related news", f"{related_articles} articles",br.get("related_news")),
        _stream_row("Combined",     "all 4 streams",               br.get("combined_stream")),
    ])

    st.markdown(f"""
    <table class="sim-table">
        <thead>
            <tr>
                <th>Stream</th><th>Input</th><th>Days eval.</th>
                <th>BUY</th><th>SELL</th><th>HOLD</th>
                <th>Directional</th>
                <th title="Information Coefficient — Pearson correlation between score and 7d return. ✦ = p&lt;0.05">IC score</th>
                <th>Status</th>
            </tr>
        </thead>
        <tbody>{rows_html}</tbody>
    </table>
    <div style="font-size:10px;color:#aaa;margin-top:4px">
      IC score: Pearson correlation between stream score and 7-day return
      (✦ = significant at p&lt;0.05 · ≥10 samples required).
      Market news IC uses idiosyncratic return when SPY history is available.
    </div>""", unsafe_allow_html=True)

    # ------------------------------------------------------------------ #
    # Issues summary                                                       #
    # ------------------------------------------------------------------ #
    issues = []

    if price_rows < _MIN_PRICE_ROWS_REGRESSION:
        issues.append(
            f"Price data: only {price_rows} rows — Ridge regression inactive "
            f"(needs ≥{_MIN_PRICE_ROWS_REGRESSION})"
        )

    for key, label in [
        ("technical",      "Technical"),
        ("stock_news",     "Stock news"),
        ("market_news",    "Market news"),
        ("related_news",   "Related news"),
        ("combined_stream","Combined"),
    ]:
        m = br.get(key)
        if m is None:
            issues.append(f"{label}: no backtest data returned — check DB / price history")
        elif m.get("is_anti_predictive"):
            issues.append(
                f"{label}: anti-predictive "
                f"(p={m.get('p_value_less', '?'):.3f}) — worse than random"
            )
        elif not m.get("is_sufficient"):
            sc = m.get("signal_count", 0)
            issues.append(
                f"{label}: only {sc} directional signals "
                f"(needs ≥{_MIN_SIGNALS_SUFFICIENT})"
            )

