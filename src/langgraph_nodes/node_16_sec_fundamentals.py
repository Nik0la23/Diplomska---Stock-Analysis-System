"""
Node 16: SEC Fundamentals Analysis

Fetches SEC EDGAR filings for the target ticker and its peers, extracts
structured parameters, runs FinBERT sentiment analysis on narrative text,
and writes a clean fundamental_context dict into state.

DATA ISOLATION CONTRACT:
    This node runs AFTER Node 12 (signal generation) and writes ONLY to
    state['fundamental_context'], which is consumed exclusively by Nodes 13
    and 14 (LLM explanation layer). It MUST NOT be read by Nodes 4-12.
    Historical filing documents describe past events — allowing them to
    influence signal generation or backtesting would introduce look-ahead
    bias and invalidate quantitative results.

What is fetched:
    Target ticker : 2 × 10-Q  (MD&A text + revenue/margin trend)
                    3 × 8-K   (event type + event sentiment)
    Each peer     : 2 × 8-K   (event type + event sentiment)

Runs AFTER: Node 12 (signal_generation) — needs state['final_signal'] for
            divergence comparison
Runs BEFORE: Node 13 (beginner_explanation), Node 14 (technical_explanation)
Can run in PARALLEL with: Nothing (sequential, between Node 12 and Node 13)
"""

import json
import re
import time
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests

from src.database.db_manager import get_connection
from src.utils.config import DATABASE_PATH
from src.langgraph_nodes.node_05_sentiment_analysis import (
    load_finbert_model,
    analyze_text_with_finbert,
)

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

SEC_HEADERS = {"User-Agent": "nikolanedeski23@gmail.com"}
SEC_RATE_LIMIT_SLEEP = 0.1      # Stay within SEC's 10 req/s limit
MAX_TEXT_CHARS = 8000           # Hard cap on text passed to FinBERT pipeline
FINBERT_CHUNK_CHARS = 1400      # ~420 tokens — safely under FinBERT's 512-token limit
                                # (financial text averages ~3 chars/token; 2000 chars → 606
                                # tokens which triggers a tensor size error in the model)
CACHE_HOURS = 24

# 8-K Item number → human-readable event description
ITEM_MAPPING: Dict[str, str] = {

    "1.01": "material agreement",
    "1.02": "termination of agreement",
    "2.01": "acquisition or disposal",
    "2.02": "earnings results",
    "5.02": "executive change",
    "7.01": "regulation FD disclosure",
}

# Revenue/margin keyword sets for directional trend extraction
_GROWING_REVENUE = [
    "revenue increased", "revenue grew", "revenue growth",
    "net revenue increased", "net sales increased", "revenues increased",
]
_DECLINING_REVENUE = [
    "revenue decreased", "revenue declined", "revenue fell",
    "net revenue decreased", "net sales decreased", "revenues decreased",
]
_GROWING_MARGIN = [
    "gross margin improved", "gross margin expanded", "gross margin increased",
    "operating margin improved", "operating margin expanded",
    "margin improvement", "margin expansion",
]
_DECLINING_MARGIN = [
    "gross margin declined", "gross margin compressed", "gross margin decreased",
    "operating margin declined", "operating margin compressed",
    "margin compression", "margin contraction",
]

# Divergence thresholds
_SENTIMENT_DIVERGENCE_THRESHOLD = 0.4   # Flag if gap between target and peer > this
_FUNDAMENTAL_BULLISH_THRESHOLD  = 0.15
_FUNDAMENTAL_BEARISH_THRESHOLD  = -0.15


# ============================================================================
# HELPER: CIK RESOLUTION
# ============================================================================

def _fetch_all_company_tickers() -> Dict[str, Any]:
    """
    Fetch the full SEC company-tickers JSON in one request.

    This covers all ~10 000 registered companies. The result is passed to
    _resolve_cik() for each ticker so the total cost is one network call
    regardless of how many peers need resolution.

    Returns:
        Raw JSON dict from https://www.sec.gov/files/company_tickers.json,
        or an empty dict if the request fails.

    Example:
        >>> data = _fetch_all_company_tickers()
        >>> len(data)  # ~10000
    """
    url = "https://www.sec.gov/files/company_tickers.json"
    try:
        resp = requests.get(url, headers=SEC_HEADERS, timeout=10)
        time.sleep(SEC_RATE_LIMIT_SLEEP)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error(f"Failed to fetch company tickers JSON: {e}")
        return {}


def _resolve_cik(ticker: str, all_tickers_data: Dict[str, Any]) -> Optional[str]:
    """
    Resolve a ticker symbol to a zero-padded 10-digit SEC CIK string.

    Args:
        ticker: Stock ticker symbol, e.g. 'AAPL'
        all_tickers_data: Raw dict from _fetch_all_company_tickers()

    Returns:
        Zero-padded CIK string (e.g. '0000320193'), or None if not found.

    Example:
        >>> data = _fetch_all_company_tickers()
        >>> _resolve_cik('AAPL', data)
        '0000320193'
    """
    ticker_upper = ticker.upper()
    for entry in all_tickers_data.values():
        if entry.get("ticker", "").upper() == ticker_upper:
            cik_raw = entry.get("cik_str") or entry.get("cik")
            if cik_raw is not None:
                return str(cik_raw).zfill(10)
    logger.warning(f"CIK not found for ticker {ticker}")
    return None


# ============================================================================
# HELPER: FILING LIST FROM SUBMISSIONS JSON
# ============================================================================

def _get_filings_list(
    cik: str,
    form_type: str,
    count: int,
) -> List[Dict[str, str]]:
    """
    Fetch the N most recent filings of a given form type for a CIK.

    Uses the EDGAR submissions JSON endpoint which returns a complete filing
    history with accession numbers and document metadata.

    Args:
        cik: Zero-padded 10-digit CIK string.
        form_type: SEC form type, e.g. '10-Q' or '8-K'.
        count: How many recent filings to return.

    Returns:
        List of dicts with keys 'accession_number', 'filed_date',
        'primary_document'. Empty list if fetch fails or no filings found.

    Example:
        >>> filings = _get_filings_list('0000320193', '10-Q', 2)
        >>> filings[0]['filed_date']
        '2024-11-01'
    """
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    try:
        resp = requests.get(url, headers=SEC_HEADERS, timeout=10)
        time.sleep(SEC_RATE_LIMIT_SLEEP)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error(f"Failed to fetch submissions for CIK {cik}: {e}")
        return []

    recent = data.get("filings", {}).get("recent", {})
    forms        = recent.get("form", [])
    accessions   = recent.get("accessionNumber", [])
    filed_dates  = recent.get("filingDate", [])
    primary_docs = recent.get("primaryDocument", [])

    results: List[Dict[str, str]] = []
    for i, form in enumerate(forms):
        if form == form_type:
            results.append({
                "accession_number": accessions[i].replace("-", ""),
                "filed_date": filed_dates[i],
                "primary_document": primary_docs[i],
            })
            if len(results) >= count:
                break

    logger.debug(f"Found {len(results)} {form_type} filings for CIK {cik}")
    return results


# ============================================================================
# HELPER: FILING INDEX — FIND BEST FILENAME
# ============================================================================

def _fetch_filing_index(cik: str, accession_number: str, form_type: str = "") -> Optional[str]:
    """
    Fetch the filing index page and return the best document filename.

    Parses the index table to find the most readable narrative document:
        - For 10-Q: prefers the complete submission `.txt` file (contains the
          full plain-text document including MD&A deep in the filing).
        - For 8-K: prefers `EX-99.1` (earnings/press release) which is the
          actual human-readable narrative. Falls back to the first non-XBRL
          `.htm` file. Avoids iXBRL-embedded primary 8-K documents and any
          `.xml` / `.xsd` files.

    Args:
        cik: Zero-padded 10-digit CIK string.
        accession_number: Dash-stripped accession number (20 chars).
        form_type: '10-Q' or '8-K' — guides document preference.

    Returns:
        Filename string to use in the document URL, or None on failure.

    Example:
        >>> _fetch_filing_index('0000320193', '000032019326000006', '10-Q')
        '0000320193-26-000006.txt'
    """
    acc_dashed = f"{accession_number[:10]}-{accession_number[10:12]}-{accession_number[12:]}"
    index_url = (
        f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
        f"{accession_number}/{acc_dashed}-index.htm"
    )
    try:
        resp = requests.get(index_url, headers=SEC_HEADERS, timeout=10)
        time.sleep(SEC_RATE_LIMIT_SLEEP)
        resp.raise_for_status()

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")

        # Build list of (description, filename) from the index table
        documents: List[Tuple[str, str]] = []
        for row in soup.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) >= 3:
                desc = cells[1].get_text(strip=True).upper()
                href_tag = cells[2].find("a", href=True)
                if href_tag:
                    fname = href_tag["href"].split("/")[-1]
                    documents.append((desc, fname))

        # --- 10-Q: full submission .txt is best (contains complete plain text)
        if form_type == "10-Q":
            for desc, fname in documents:
                if fname.endswith(".txt") and "SUBMISSION" not in desc:
                    return fname
            # Fall back to first non-XBRL .htm
            for desc, fname in documents:
                if fname.endswith(".htm") and "XBRL" not in desc and "_htm.xml" not in fname:
                    return fname

        # --- 8-K: prefer the primary 8-K document (contains Item numbers needed
        # for event extraction). iXBRL-embedded documents are readable after
        # stripping ix:hidden/ix:header tags in _fetch_filing_text().
        # Note: EX-99.1 (press releases) do NOT contain Item numbers, so we
        # use the primary document first. If it is excessively short (<200 chars
        # after stripping), the caller falls back to primary_document from the
        # submissions JSON.
        if form_type == "8-K":
            # Primary 8-K document (desc == '8-K', any .htm variant)
            for desc, fname in documents:
                if desc == "8-K" and fname.endswith(".htm"):
                    return fname
            # Fall back to first non-XML .htm
            for desc, fname in documents:
                if (fname.endswith(".htm")
                        and not fname.endswith((".xml", ".xsd"))
                        and "SCHEMA" not in desc
                        and "LINKBASE" not in desc):
                    return fname

        # Generic fallback for any form type
        for desc, fname in documents:
            if fname.endswith(".htm") and "XBRL" not in desc and "IXBRL" not in desc:
                return fname

    except Exception as e:
        logger.debug(f"Could not fetch filing index for {accession_number}: {e}")

    return None


# ============================================================================
# HELPER: FETCH AND CLEAN FILING TEXT
# ============================================================================

def _fetch_filing_text(
    cik: str,
    accession_number: str,
    filename: str,
) -> Optional[str]:
    """
    Download a single SEC filing document and return clean plain text.

    Does NOT truncate — the full cleaned text is returned so that callers
    can extract the relevant section (e.g. MD&A) before truncating. A 10-Q
    can be ~60KB cleaned; an 8-K is typically < 5KB.

    If the file is HTML or iXBRL-embedded, tags are stripped with
    BeautifulSoup. Plain-text `.txt` files are returned as-is.

    Args:
        cik: Zero-padded 10-digit CIK string.
        accession_number: Dash-stripped 20-char accession number.
        filename: Document filename from the filing index.

    Returns:
        Clean plain text (full length), or None on failure.

    Example:
        >>> text = _fetch_filing_text('0000320193', '000032019326000006', 'aapl-20251227.htm')
        >>> len(text) > 8000
        True
    """
    url = (
        f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
        f"{accession_number}/{filename}"
    )
    try:
        resp = requests.get(url, headers=SEC_HEADERS, timeout=30)
        time.sleep(SEC_RATE_LIMIT_SLEEP)
        resp.raise_for_status()
        raw = resp.text

        # Strip HTML/iXBRL if needed
        if filename.lower().endswith((".htm", ".html")) or "<html" in raw[:500].lower():
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(raw, "html.parser")
                # Remove iXBRL hidden data blocks (contain structured XBRL values,
                # not human-readable text — without this step the output starts with
                # "true true true true NASDAQ NASDAQ..." garbage)
                for tag in soup.find_all(["ix:hidden", "ix:header"]):
                    tag.decompose()
                text = soup.get_text(separator=" ", strip=True)
            except Exception:
                text = re.sub(r"<[^>]+>", " ", raw)
                text = re.sub(r"\s+", " ", text).strip()
        else:
            text = raw

        return text

    except Exception as e:
        logger.warning(f"Failed to fetch filing text from {url}: {e}")
        return None


# ============================================================================
# HELPER: SQLITE CACHE FOR FILINGS
# ============================================================================

def _get_cached_filing(
    conn,
    accession_number: str,
    prefer_mda: bool = False,
) -> Optional[str]:
    """
    Return cached text for an accession number if it is < 24 hours old.

    Args:
        conn: Open SQLite connection.
        accession_number: Unique filing identifier (dash-stripped).
        prefer_mda: If True, return md_and_a_text when available (10-Q use
            case). Falls back to raw_text if md_and_a_text is NULL.

    Returns:
        Cached text string, or None if not cached / expired.
    """
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT raw_text, md_and_a_text FROM sec_filings
            WHERE accession_number = ?
              AND fetched_at > datetime('now', ?)
            """,
            (accession_number, f"-{CACHE_HOURS} hours"),
        )
        row = cursor.fetchone()
        if row:
            if prefer_mda and row[1]:
                logger.debug(f"Cache hit (md_and_a) for accession {accession_number}")
                return row[1]
            if row[0]:
                logger.debug(f"Cache hit (raw_text) for accession {accession_number}")
                return row[0]
    except Exception as e:
        logger.debug(f"Cache lookup failed for {accession_number}: {e}")
    return None


def _cache_filing(
    conn,
    ticker: str,
    form_type: str,
    filed_date: str,
    accession_number: str,
    raw_text: str,
    mda_text: Optional[str],
) -> None:
    """
    Insert or replace a filing record in sec_filings.

    Args:
        conn: Open SQLite connection.
        ticker: Stock ticker symbol.
        form_type: '10-Q' or '8-K'.
        filed_date: ISO date string when filing was submitted.
        accession_number: Unique filing identifier (dash-stripped).
        raw_text: Full truncated plain text of the filing.
        mda_text: Extracted MD&A section text (10-Q only), or None.
    """
    try:
        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO sec_filings
                    (ticker, form_type, filed_date, accession_number, raw_text, md_and_a_text)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (ticker, form_type, filed_date, accession_number, raw_text, mda_text),
            )
        logger.debug(f"Cached filing {accession_number} for {ticker}")
    except Exception as e:
        logger.warning(f"Failed to cache filing {accession_number}: {e}")


# ============================================================================
# HELPER: TEXT EXTRACTION
# ============================================================================

def _extract_mda_text(text: str) -> str:
    """
    Locate and extract the MD&A section from a 10-Q plain text document.

    Search strategy:
        1. Find the first occurrence of "Management" or "Discussion and Analysis"
           (case-insensitive) to locate the section start.
        2. Extract text from that point until the next major section header
           (a line in ALL CAPS with ≥ 5 characters, or "ITEM" prefix).
        3. Fallback: first 3000 chars if no MD&A header is found.
        4. Truncate result to MAX_TEXT_CHARS.

    Args:
        text: Full plain text of the 10-Q filing.

    Returns:
        Extracted MD&A text, truncated to MAX_TEXT_CHARS chars.

    Example:
        >>> mda = _extract_mda_text(filing_text)
        >>> "revenue" in mda.lower()
        True
    """
    lower = text.lower()

    # Search for MD&A header occurrences and pick the actual section body,
    # not a table of contents reference.
    #
    # TOC entries look like:
    #   "Management's Discussion and Analysis of ... 13 Item 3."
    #   (header title, then a page number, then the next item — all on one line)
    # Actual section headers are followed by substantive paragraph text.
    #
    # Detection: after the header title, if the next ~60 chars contain a
    # page-number + "Item" pattern (\d+ \s+ Item), it is a TOC entry.
    candidates = []
    for pattern in [
        "management\u2019s discussion and analysis",  # unicode apostrophe
        "management's discussion and analysis",
        "management\u2019s discussion",
        "management's discussion",
        "discussion and analysis",
    ]:
        pos = 0
        while True:
            idx = lower.find(pattern, pos)
            if idx == -1:
                break
            candidates.append(idx)
            pos = idx + len(pattern)

    if not candidates:
        logger.debug("MD&A header not found, using first 3000 chars as fallback")
        return text[:3000]

    # Sort candidates and pick the first non-TOC occurrence
    candidates = sorted(set(candidates))
    start_idx = candidates[-1]  # safe default: last occurrence is deepest in doc
    for idx in candidates:
        after = text[idx: idx + 120]
        # TOC pattern: page number immediately followed by "Item" keyword
        is_toc = bool(re.search(r'\s\d{1,3}\s+[Ii]tem\s+\d', after))
        if not is_toc:
            start_idx = idx
            break

    section = text[start_idx:]

    # Find next major section header after the first 200 chars (skip the title itself)
    next_section = re.search(
        r"\n(?:ITEM\s+\d|[A-Z][A-Z\s]{4,})\b",
        section[200:],
    )
    if next_section:
        section = section[: 200 + next_section.start()]

    return section[:MAX_TEXT_CHARS]


def _extract_revenue_trend(text: str) -> str:
    """
    Determine directional revenue trend from MD&A or financial table text.

    Scans for positive/negative revenue keyword phrases. Growing takes
    precedence over declining when both appear in the same document.

    Args:
        text: Plain text from the 10-Q filing.

    Returns:
        'growing' | 'declining' | 'stable'

    Example:
        >>> _extract_revenue_trend("Net revenue increased 12% year over year.")
        'growing'
    """
    lower = text.lower()
    growing  = any(kw in lower for kw in _GROWING_REVENUE)
    declining = any(kw in lower for kw in _DECLINING_REVENUE)

    if growing and not declining:
        return "growing"
    if declining and not growing:
        return "declining"
    if growing and declining:
        # Both present — count occurrences to pick dominant signal
        grow_count = sum(lower.count(kw) for kw in _GROWING_REVENUE)
        decl_count = sum(lower.count(kw) for kw in _DECLINING_REVENUE)
        return "growing" if grow_count >= decl_count else "declining"
    return "stable"


def _extract_margin_trend(text: str) -> str:
    """
    Determine directional gross/operating margin trend from filing text.

    Args:
        text: Plain text from the 10-Q filing.

    Returns:
        'growing' | 'declining' | 'stable'

    Example:
        >>> _extract_margin_trend("Gross margin expanded to 42.3% from 39.1%.")
        'growing'
    """
    lower = text.lower()
    growing  = any(kw in lower for kw in _GROWING_MARGIN)
    declining = any(kw in lower for kw in _DECLINING_MARGIN)

    if growing and not declining:
        return "growing"
    if declining and not growing:
        return "declining"
    if growing and declining:
        grow_count = sum(lower.count(kw) for kw in _GROWING_MARGIN)
        decl_count = sum(lower.count(kw) for kw in _DECLINING_MARGIN)
        return "growing" if grow_count >= decl_count else "declining"
    return "stable"


def _extract_8k_events(text: str) -> List[str]:
    """
    Extract human-readable event descriptions from an 8-K filing.

    Scans for "Item X.XX" patterns and maps matched numbers to
    ITEM_MAPPING descriptions. Deduplicates the result.

    Args:
        text: Plain text of the 8-K filing.

    Returns:
        List of unique event description strings (may be empty).

    Example:
        >>> events = _extract_8k_events("This report contains Item 2.02 and Item 5.02.")
        >>> events
        ['earnings results', 'executive change']
    """
    matches = re.findall(r"[Ii]tem\s+(\d+\.\d+)", text)
    events: List[str] = []
    seen: set = set()
    for item_num in matches:
        description = ITEM_MAPPING.get(item_num)
        if description and description not in seen:
            events.append(description)
            seen.add(description)
    return events


# ============================================================================
# HELPER: FINBERT CHUNKED SCORING
# ============================================================================

def _chunk_and_score_finbert(text: str, model) -> Optional[float]:
    """
    Score a long text with FinBERT by splitting into FINBERT_CHUNK_CHARS
    chunks (~420 tokens each, well under FinBERT's 512-token hard limit).

    SEC filings are up to MAX_TEXT_CHARS long; chunking gives better coverage
    than a single truncated pass.  Each chunk is scored independently and the
    final score is the mean sentiment across all successful chunks.

    Args:
        text: Plain text to score (any length).
        model: FinBERT pipeline object from load_finbert_model().

    Returns:
        Mean sentiment float in [-1.0, +1.0], or None if model is None
        or all chunks fail.

    Example:
        >>> model = load_finbert_model()
        >>> score = _chunk_and_score_finbert("Revenue grew 15%...", model)
        >>> isinstance(score, float)
        True
    """
    if model is None or not text:
        return None

    chunks = [
        text[i: i + FINBERT_CHUNK_CHARS]
        for i in range(0, len(text), FINBERT_CHUNK_CHARS)
    ]

    scores: List[float] = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            result = analyze_text_with_finbert(chunk, model)
            scores.append(result["sentiment"])
        except Exception as e:
            logger.debug(f"FinBERT chunk scoring failed: {e}")

    if not scores:
        return None
    return sum(scores) / len(scores)


# ============================================================================
# HELPER: PROCESS TARGET FILINGS (10-Q × 2, 8-K × 3)
# ============================================================================

def _process_target_filings(
    ticker: str,
    cik: str,
    conn,
    model,
) -> Dict[str, Any]:
    """
    Fetch and analyse 2 × 10-Q and 3 × 8-K filings for the target ticker.

    For each 10-Q:
        - Cache check (24h) → fetch if needed → extract MD&A → FinBERT
        - Extract revenue trend and margin trend from full text

    For each 8-K:
        - Cache check → fetch if needed → extract Item events → FinBERT

    Management sentiment is the mean FinBERT score across all successfully
    scored filings (both 10-Q and 8-K contribute).

    Args:
        ticker: Target stock ticker symbol.
        cik: Zero-padded 10-digit SEC CIK.
        conn: Open SQLite connection.
        model: FinBERT pipeline (may be None — handled gracefully).

    Returns:
        Dict with keys: revenue_trend, margin_trend, management_sentiment,
        recent_events, filing_dates, quarters_covered.
    """
    sentiment_scores: List[float] = []
    all_events:       List[str]   = []
    filing_dates:     List[str]   = []
    revenue_trends:   List[str]   = []
    margin_trends:    List[str]   = []

    # ---- 10-Q filings -------------------------------------------------------
    # Strategy: cache check returns the extracted MD&A (prefer_mda=True).
    # If not cached: fetch the full document (no truncation), extract MD&A from
    # the full text, then truncate the MD&A to MAX_TEXT_CHARS for FinBERT.
    # Revenue/margin trends are also extracted from the full text.
    tenq_filings = _get_filings_list(cik, "10-Q", 2)
    for filing in tenq_filings:
        acc = filing["accession_number"]
        filed = filing["filed_date"]

        # Try to get the already-extracted MD&A from cache
        mda = _get_cached_filing(conn, acc, prefer_mda=True)
        if mda is None:
            # Fetch full document — MD&A may be 10-15KB into a 60KB+ file
            filename = _fetch_filing_index(cik, acc, "10-Q") or filing["primary_document"]
            full_text = _fetch_filing_text(cik, acc, filename)
            if not full_text:
                logger.warning(f"Could not fetch 10-Q {acc} for {ticker}")
                continue
            mda = _extract_mda_text(full_text)
            # Cache: store first MAX_TEXT_CHARS of full text + extracted MD&A
            _cache_filing(conn, ticker, "10-Q", filed, acc,
                          full_text[:MAX_TEXT_CHARS], mda)
            # Use full text for trend extraction (trends may be outside first 8K chars)
            revenue_trends.append(_extract_revenue_trend(full_text))
            margin_trends.append(_extract_margin_trend(full_text))
        else:
            # Re-derive trends from the cached MD&A (good enough approximation)
            revenue_trends.append(_extract_revenue_trend(mda))
            margin_trends.append(_extract_margin_trend(mda))

        # FinBERT on MD&A (already truncated to MAX_TEXT_CHARS)
        score = _chunk_and_score_finbert(mda, model)
        if score is not None:
            sentiment_scores.append(score)

        filing_dates.append(filed)

    # ---- 8-K filings -------------------------------------------------------
    # Strategy: 8-Ks are short (~4KB cleaned). Prefer EX-99.1 (press release)
    # over the iXBRL-embedded primary document.
    eightk_filings = _get_filings_list(cik, "8-K", 3)
    for filing in eightk_filings:
        acc = filing["accession_number"]
        filed = filing["filed_date"]

        raw_text = _get_cached_filing(conn, acc)
        if raw_text is None:
            filename = _fetch_filing_index(cik, acc, "8-K") or filing["primary_document"]
            raw_text = _fetch_filing_text(cik, acc, filename)
            if raw_text:
                # 8-Ks are short — truncate before caching
                raw_text = raw_text[:MAX_TEXT_CHARS]
                _cache_filing(conn, ticker, "8-K", filed, acc, raw_text, None)
            else:
                logger.warning(f"Could not fetch 8-K {acc} for {ticker}")
                continue

        events = _extract_8k_events(raw_text)
        all_events.extend(events)

        score = _chunk_and_score_finbert(raw_text, model)
        if score is not None:
            sentiment_scores.append(score)

        if filed not in filing_dates:
            filing_dates.append(filed)

    # ---- Aggregate ---------------------------------------------------------
    mgmt_sentiment: Optional[float] = (
        sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else None
    )

    # Majority-vote for trends across 2 quarters
    def _majority(trend_list: List[str], default: str = "stable") -> str:
        if not trend_list:
            return default
        return max(set(trend_list), key=trend_list.count)

    return {
        "revenue_trend":        _majority(revenue_trends),
        "margin_trend":         _majority(margin_trends),
        "management_sentiment": mgmt_sentiment,
        "recent_events":        list(dict.fromkeys(all_events)),  # deduplicated
        "filing_dates":         sorted(filing_dates, reverse=True),
        "quarters_covered":     len(tenq_filings),
    }


# ============================================================================
# HELPER: PROCESS PEER FILINGS (8-K × 2)
# ============================================================================

def _process_peer_filings(
    peer_ticker: str,
    peer_relationship: str,
    cik: str,
    conn,
    model,
) -> Optional[Dict[str, Any]]:
    """
    Fetch and analyse 2 × 8-K filings for a single peer ticker.

    Only 8-K filings are fetched for peers — 10-Q is too much data and
    peers need event-level context only (not full fundamental analysis).

    Args:
        peer_ticker: Peer stock ticker symbol.
        peer_relationship: Relationship type from Node 3, e.g. 'COMPETITOR'.
        cik: Zero-padded 10-digit SEC CIK for this peer.
        conn: Open SQLite connection.
        model: FinBERT pipeline (may be None).

    Returns:
        Dict with keys: ticker, relationship, recent_events, event_sentiment,
        filing_dates. Returns None if no filings could be fetched at all.
    """
    sentiment_scores: List[float] = []
    all_events:       List[str]   = []
    filing_dates:     List[str]   = []

    filings = _get_filings_list(cik, "8-K", 2)
    for filing in filings:
        acc = filing["accession_number"]
        filed = filing["filed_date"]

        raw_text = _get_cached_filing(conn, acc)
        if raw_text is None:
            filename = _fetch_filing_index(cik, acc, "8-K") or filing["primary_document"]
            raw_text = _fetch_filing_text(cik, acc, filename)
            if raw_text:
                raw_text = raw_text[:MAX_TEXT_CHARS]
                _cache_filing(conn, peer_ticker, "8-K", filed, acc, raw_text, None)
            else:
                logger.warning(f"Could not fetch 8-K {acc} for peer {peer_ticker}")
                continue

        events = _extract_8k_events(raw_text)
        all_events.extend(events)

        score = _chunk_and_score_finbert(raw_text, model)
        if score is not None:
            sentiment_scores.append(score)

        filing_dates.append(filed)

    if not filing_dates:
        logger.warning(f"No 8-K filings retrieved for peer {peer_ticker}")
        return None

    avg_sentiment: Optional[float] = (
        sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else None
    )

    return {
        "ticker":          peer_ticker,
        "relationship":    peer_relationship,
        "recent_events":   list(dict.fromkeys(all_events)),
        "event_sentiment": avg_sentiment,
        "filing_dates":    sorted(filing_dates, reverse=True),
    }


# ============================================================================
# HELPER: PERSIST SEC FUNDAMENTALS
# ============================================================================

def _persist_fundamentals(
    conn,
    ticker: str,
    target: Dict[str, Any],
    peers: Dict[str, Any],
    fundamental_signal: str,
    fundamental_confidence: float,
) -> None:
    """
    Write target fundamental analysis and peer events to SQLite.

    Args:
        conn: Open SQLite connection.
        ticker: Target ticker symbol.
        target: Result dict from _process_target_filings().
        peers: Dict of peer_ticker → peer result dicts.
        fundamental_signal: 'BULLISH' | 'BEARISH' | 'NEUTRAL'.
        fundamental_confidence: 0.0 to 1.0.
    """
    today = datetime.now().strftime("%Y-%m-%d")

    # sec_fundamentals — one row per ticker per analysis date
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO sec_fundamentals
                    (ticker, analysis_date, revenue_trend, margin_trend,
                     management_sentiment, recent_events,
                     fundamental_signal, fundamental_confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ticker,
                    today,
                    target.get("revenue_trend"),
                    target.get("margin_trend"),
                    target.get("management_sentiment"),
                    json.dumps(target.get("recent_events", [])),
                    fundamental_signal,
                    fundamental_confidence,
                ),
            )
    except Exception as e:
        logger.warning(f"Failed to persist sec_fundamentals for {ticker}: {e}")

    # sec_peer_events — one row per peer event batch
    for peer_ticker, peer_data in peers.items():
        try:
            filed_date = (peer_data.get("filing_dates") or [""])[0]
            with conn:
                conn.execute(
                    """
                    INSERT INTO sec_peer_events
                        (target_ticker, peer_ticker, filed_date,
                         event_description, event_sentiment)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        ticker,
                        peer_ticker,
                        filed_date,
                        json.dumps(peer_data.get("recent_events", [])),
                        peer_data.get("event_sentiment"),
                    ),
                )
        except Exception as e:
            logger.warning(f"Failed to persist sec_peer_events for {peer_ticker}: {e}")


# ============================================================================
# HELPER: FUNDAMENTAL SIGNAL CALCULATION
# ============================================================================

def _calculate_fundamental_signal(
    management_sentiment: Optional[float],
    revenue_trend: str,
    margin_trend: str,
) -> Tuple[str, float]:
    """
    Derive a directional fundamental signal from document-level evidence.

    Composite score:
        management_sentiment × 0.5  (FinBERT mean across MD&A + 8-K)
        revenue_trend score  × 0.3  (growing=+1, stable=0, declining=-1)
        margin_trend score   × 0.2  (growing=+1, stable=0, declining=-1)

    Thresholds:
        combined > +0.15  →  BULLISH
        combined < -0.15  →  BEARISH
        otherwise         →  NEUTRAL

    Args:
        management_sentiment: Mean FinBERT sentiment, or None.
        revenue_trend: 'growing' | 'stable' | 'declining'.
        margin_trend: 'growing' | 'stable' | 'declining'.

    Returns:
        Tuple of (signal_string, confidence_float).

    Example:
        >>> _calculate_fundamental_signal(0.6, 'growing', 'stable')
        ('BULLISH', 0.5)
    """
    trend_map = {"growing": 1.0, "stable": 0.0, "declining": -1.0}
    revenue_score = trend_map.get(revenue_trend, 0.0)
    margin_score  = trend_map.get(margin_trend,  0.0)

    # If sentiment unavailable, fall back to trend-only (equal weight)
    if management_sentiment is not None:
        combined = (
            management_sentiment * 0.5
            + revenue_score      * 0.3
            + margin_score       * 0.2
        )
    else:
        combined = revenue_score * 0.6 + margin_score * 0.4

    if combined > _FUNDAMENTAL_BULLISH_THRESHOLD:
        signal = "BULLISH"
    elif combined < _FUNDAMENTAL_BEARISH_THRESHOLD:
        signal = "BEARISH"
    else:
        signal = "NEUTRAL"

    confidence = min(1.0, abs(combined))
    return signal, round(confidence, 4)


# ============================================================================
# HELPER: DIVERGENCE STRINGS
# ============================================================================

def _build_divergences(
    target: Dict[str, Any],
    peers: Dict[str, Any],
    final_signal: Optional[str],
    fundamental_signal: str,
) -> List[str]:
    """
    Produce up to 5 plain-English divergence strings for Nodes 13/14.

    Compares:
        - Target management sentiment vs each peer's event sentiment
        - Target negative 8-K events vs peers reporting positive events
        - fundamental_signal vs final_signal from Node 12

    The relationship type (COMPETITOR / SUPPLIER etc.) is included in each
    string so the LLM can contextualise the divergence correctly.

    Args:
        target: Result dict from _process_target_filings().
        peers: Dict of peer_ticker → peer result dicts.
        final_signal: 'BUY' | 'SELL' | 'HOLD' from Node 12, or None.
        fundamental_signal: 'BULLISH' | 'BEARISH' | 'NEUTRAL'.

    Returns:
        List of up to 5 divergence description strings, most significant first.

    Example:
        >>> divs = _build_divergences(target, peers, 'BUY', 'BEARISH')
        >>> 'contradicts' in divs[0].lower()
        True
    """
    divergences: List[Tuple[float, str]] = []  # (priority_score, message)
    target_sentiment = target.get("management_sentiment")
    target_events    = set(target.get("recent_events", []))

    # 1. Signal contradiction: Node 12 vs fundamentals
    if final_signal and fundamental_signal != "NEUTRAL":
        signal_map = {"BUY": "bullish", "SELL": "bearish", "HOLD": "neutral"}
        fundamental_direction = fundamental_signal.lower()
        technical_direction   = signal_map.get(final_signal, "neutral")
        if (
            (fundamental_direction == "bearish" and final_signal == "BUY")
            or (fundamental_direction == "bullish" and final_signal == "SELL")
        ):
            divergences.append((
                1.0,
                f"SEC fundamentals signal is {fundamental_signal} while the "
                f"quantitative model generated a {final_signal} signal — "
                f"management tone and financial trends contradict the technical/sentiment view.",
            ))

    # 2. Peer sentiment vs target sentiment
    if target_sentiment is not None:
        for peer_ticker, peer_data in peers.items():
            peer_sentiment = peer_data.get("event_sentiment")
            if peer_sentiment is None:
                continue
            gap = peer_sentiment - target_sentiment
            relationship = peer_data.get("relationship", "peer").capitalize()
            if abs(gap) >= _SENTIMENT_DIVERGENCE_THRESHOLD:
                direction = "more positive" if gap > 0 else "more negative"
                divergences.append((
                    min(1.0, abs(gap)),
                    f"{relationship} {peer_ticker} filing sentiment is {direction} "
                    f"(score {peer_sentiment:+.2f}) than {target_sentiment:+.2f} "
                    f"in target filings.",
                ))

    # 3. Negative target events vs positive peer events
    negative_target_events = {"executive change", "termination of agreement"}
    if target_events & negative_target_events:
        for peer_ticker, peer_data in peers.items():
            peer_events = set(peer_data.get("recent_events", []))
            positive_peer = {"earnings results", "material agreement", "acquisition or disposal"}
            if peer_events & positive_peer:
                relationship = peer_data.get("relationship", "peer").capitalize()
                hit = (target_events & negative_target_events).pop()
                divergences.append((
                    0.7,
                    f"Target recently filed a '{hit}' event while {relationship} "
                    f"{peer_ticker} reported positive events "
                    f"({', '.join(peer_events & positive_peer)}).",
                ))
                break  # one divergence of this type is enough

    # 4. Revenue / margin trends vs peer events
    if target.get("revenue_trend") == "declining":
        positive_peers = [
            f"{pd['relationship'].capitalize()} {pt}"
            for pt, pd in peers.items()
            if pd.get("event_sentiment") is not None and pd["event_sentiment"] > 0.2
        ]
        if positive_peers:
            divergences.append((
                0.6,
                f"Target revenue is declining while "
                f"{', '.join(positive_peers[:3])} show positive recent events.",
            ))

    # Sort by priority descending, return top 5 message strings
    divergences.sort(key=lambda x: x[0], reverse=True)
    return [msg for _, msg in divergences[:5]]


# ============================================================================
# HELPER: ENSURE SEC TABLES EXIST
# ============================================================================

_SEC_TABLES_DDL = """
CREATE TABLE IF NOT EXISTS sec_filings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    form_type TEXT NOT NULL,
    filed_date TEXT NOT NULL,
    accession_number TEXT UNIQUE,
    raw_text TEXT,
    md_and_a_text TEXT,
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_sec_filings_ticker ON sec_filings(ticker, form_type);
CREATE INDEX IF NOT EXISTS idx_sec_filings_accession ON sec_filings(accession_number);

CREATE TABLE IF NOT EXISTS sec_fundamentals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    analysis_date TEXT NOT NULL,
    revenue_trend TEXT,
    margin_trend TEXT,
    management_sentiment REAL,
    recent_events TEXT,
    fundamental_signal TEXT,
    fundamental_confidence REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_sec_fundamentals_ticker ON sec_fundamentals(ticker, analysis_date DESC);

CREATE TABLE IF NOT EXISTS sec_peer_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    target_ticker TEXT NOT NULL,
    peer_ticker TEXT NOT NULL,
    filed_date TEXT NOT NULL,
    event_description TEXT,
    event_sentiment REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_sec_peer_events_target ON sec_peer_events(target_ticker, peer_ticker);
"""


def _ensure_sec_tables(conn) -> None:
    """
    Create SEC tables if they do not already exist.

    Uses IF NOT EXISTS throughout so it is safe to call on every run without
    touching or conflicting with existing tables or indexes.

    Args:
        conn: Open SQLite connection.
    """
    try:
        conn.executescript(_SEC_TABLES_DDL)
        conn.commit()
        logger.debug("Node 16: SEC tables verified / created")
    except Exception as e:
        logger.warning(f"Node 16: Could not ensure SEC tables: {e}")


# ============================================================================
# MAIN NODE FUNCTION
# ============================================================================

def sec_fundamentals_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 16: SEC Fundamentals Analysis.

    Fetches SEC EDGAR filings, extracts structured fundamental signals, and
    writes narrative context for the LLM explanation nodes (13 and 14).

    DATA ISOLATION: This node sits after Node 12 and writes only to
    state['fundamental_context']. That key is never read by Nodes 4-12.
    This prevents historical filing documents from contaminating the
    quantitative signal pipeline.

    State reads:
        state['ticker']             — target stock symbol
        state['related_companies']  — peer list from Node 3
        state['final_signal']       — Node 12 output (optional, for divergence)

    State writes:
        state['fundamental_context']              — full SEC analysis
        state['node_execution_times']['node_16']  — execution time in seconds

    Args:
        state: Current LangGraph pipeline state.

    Returns:
        Updated state with fundamental_context populated.

    Raises:
        Never. All exceptions are caught, logged, and added to state['errors'].
        The pipeline always continues.

    Example:
        >>> state = create_initial_state('AAPL')
        >>> state = sec_fundamentals_node(state)
        >>> state['fundamental_context']['fundamental_signal']
        'BULLISH'
    """
    start_time = time.time()
    ticker = state.get("ticker", "UNKNOWN")
    logger.info(f"Node 16 starting SEC fundamentals analysis for {ticker}")

    # Accumulate only the NEW errors produced by this node so the fan-in
    # reducer (operator.add) doesn't double-count errors from previous nodes.
    _new_errors: List[str] = []
    _fundamental_context: Optional[Dict[str, Any]] = None

    try:
        related_companies: List[Dict[str, Any]] = state.get("related_companies", [])
        final_signal: Optional[str] = state.get("final_signal")

        # Load FinBERT (module-level singleton — load once, reuse across runs)
        model = load_finbert_model()
        if model is None:
            logger.warning("Node 16: FinBERT unavailable — sentiment scores will be None")

        conn = get_connection(DATABASE_PATH)

        # Ensure SEC tables exist (safe to run every time — all use IF NOT EXISTS)
        _ensure_sec_tables(conn)

        # ------------------------------------------------------------------ #
        # Step 1: Resolve all CIKs (single HTTP request)                      #
        # ------------------------------------------------------------------ #
        logger.info("Node 16: Fetching SEC company tickers for CIK resolution")
        all_tickers_data = _fetch_all_company_tickers()

        target_cik = _resolve_cik(ticker, all_tickers_data)
        if not target_cik:
            raise ValueError(f"CIK not found for target ticker {ticker}")

        # ------------------------------------------------------------------ #
        # Step 2: Process target filings (10-Q × 2, 8-K × 3)                 #
        # ------------------------------------------------------------------ #
        logger.info(f"Node 16: Processing target filings for {ticker} (CIK {target_cik})")
        target_data = _process_target_filings(ticker, target_cik, conn, model)
        logger.info(
            f"Node 16: Target — revenue={target_data['revenue_trend']}, "
            f"margin={target_data['margin_trend']}, "
            f"mgmt_sentiment={target_data['management_sentiment']}, "
            f"events={target_data['recent_events']}"
        )

        # ------------------------------------------------------------------ #
        # Step 3: Process peer filings (8-K × 2 per peer)                    #
        # ------------------------------------------------------------------ #
        peers_data: Dict[str, Any] = {}
        for peer in related_companies:
            peer_ticker = peer.get("ticker") or peer.get("related_ticker")
            if not peer_ticker:
                continue
            peer_relationship = peer.get("relationship", "unknown")
            peer_cik = _resolve_cik(peer_ticker, all_tickers_data)
            if not peer_cik:
                logger.warning(f"Node 16: CIK not found for peer {peer_ticker}, skipping")
                continue
            logger.debug(f"Node 16: Processing peer {peer_ticker} (CIK {peer_cik})")
            peer_result = _process_peer_filings(
                peer_ticker, peer_relationship, peer_cik, conn, model
            )
            if peer_result:
                peers_data[peer_ticker] = peer_result

        logger.info(f"Node 16: Processed {len(peers_data)} peers successfully")

        # ------------------------------------------------------------------ #
        # Step 4: Calculate fundamental signal                                #
        # ------------------------------------------------------------------ #
        fundamental_signal, fundamental_confidence = _calculate_fundamental_signal(
            target_data["management_sentiment"],
            target_data["revenue_trend"],
            target_data["margin_trend"],
        )
        logger.info(
            f"Node 16: Fundamental signal = {fundamental_signal} "
            f"(confidence {fundamental_confidence:.2f})"
        )

        # ------------------------------------------------------------------ #
        # Step 5: Build divergences                                           #
        # ------------------------------------------------------------------ #
        divergences = _build_divergences(
            target_data, peers_data, final_signal, fundamental_signal
        )

        # ------------------------------------------------------------------ #
        # Step 6: Persist to SQLite                                           #
        # ------------------------------------------------------------------ #
        _persist_fundamentals(
            conn, ticker, target_data, peers_data,
            fundamental_signal, fundamental_confidence,
        )

        # ------------------------------------------------------------------ #
        # Step 7: Determine data quality                                      #
        # ------------------------------------------------------------------ #
        has_sentiment  = target_data["management_sentiment"] is not None
        has_filings    = len(target_data["filing_dates"]) > 0
        if has_sentiment and has_filings:
            data_quality = "complete"
        elif has_filings:
            data_quality = "partial"
        else:
            data_quality = "unavailable"

        # ------------------------------------------------------------------ #
        # Step 8: Assemble fundamental_context                                #
        # ------------------------------------------------------------------ #
        _fundamental_context = {
            "target": {
                "ticker":               ticker,
                "revenue_trend":        target_data["revenue_trend"],
                "margin_trend":         target_data["margin_trend"],
                "management_sentiment": target_data["management_sentiment"],
                "recent_events":        target_data["recent_events"],
                "filing_dates":         target_data["filing_dates"],
                "quarters_covered":     target_data["quarters_covered"],
            },
            "peers":                peers_data,
            "divergences":          divergences,
            "fundamental_signal":   fundamental_signal,
            "fundamental_confidence": fundamental_confidence,
            "data_quality":         data_quality,
        }

        logger.info(
            f"Node 16 complete for {ticker} — "
            f"signal={fundamental_signal}, quality={data_quality}, "
            f"divergences={len(divergences)}"
        )

    except Exception as e:
        logger.error(f"Node 16 failed for {ticker}: {e}", exc_info=True)
        _new_errors.append(f"Node 16 (SEC fundamentals) failed: {str(e)}")
        _fundamental_context = {
            "target":                 {},
            "peers":                  {},
            "divergences":            [],
            "fundamental_signal":     "NEUTRAL",
            "fundamental_confidence": 0.0,
            "data_quality":           "unavailable",
            "error":                  str(e),
        }

    finally:
        elapsed = time.time() - start_time
        logger.info(f"Node 16 execution time: {elapsed:.2f}s")

    # Return ONLY the keys this node writes (not full state).  Keeps updates
    # small; workflow runs Node 16 after Node 12 so final_signal is available.
    #   - node_execution_times: operator.or_  reducer — dict union
    #   - errors              : operator.add  reducer — return only NEW errors
    return {
        "fundamental_context":  _fundamental_context,
        "node_execution_times": {"node_16": round(elapsed, 3)},
        "errors":               _new_errors,
    }
