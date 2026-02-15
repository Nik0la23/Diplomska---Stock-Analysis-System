"""
Check sentiment coverage in news_articles.

Alpha Vantage is the only API that provides sentiment per article; we store it
in sentiment_label and sentiment_score. Finnhub articles and any cached before
we added sentiment have NULL there.

Note: We do NOT store which API each row came from. So we infer:
- Rows WITH sentiment_label → likely from Alpha Vantage (API provided it).
- Rows WITHOUT sentiment_label → Finnhub, or Alpha Vantage before we saved it,
  or Alpha Vantage when API only returns ~10 days (so few articles have it).

Usage:
  python -m scripts.check_sentiment_coverage
"""

import sys
from pathlib import Path

# Project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database.db_manager import get_connection, DEFAULT_DB_PATH


def main():
    db_path = DEFAULT_DB_PATH
    if not Path(db_path).exists():
        print(f"Database not found: {db_path}")
        return 1

    with get_connection(db_path) as conn:
        cur = conn.cursor()

        # ---- Overall ----
        cur.execute("""
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN sentiment_label IS NULL OR TRIM(COALESCE(sentiment_label, '')) = '' THEN 1 ELSE 0 END) AS without_label,
                SUM(CASE WHEN sentiment_label IS NOT NULL AND TRIM(sentiment_label) != '' THEN 1 ELSE 0 END) AS with_label,
                SUM(CASE WHEN sentiment_score IS NULL THEN 1 ELSE 0 END) AS without_score,
                SUM(CASE WHEN sentiment_score IS NOT NULL THEN 1 ELSE 0 END) AS with_score
            FROM news_articles
        """)
        row = cur.fetchone()
        total = row['total'] or 0
        without_label = row['without_label'] or 0
        with_label = row['with_label'] or 0
        without_score = row['without_score'] or 0
        with_score = row['with_score'] or 0

        # ---- By news_type (stock = Alpha Vantage + Finnhub; market = Finnhub only) ----
        cur.execute("""
            SELECT
                news_type,
                COUNT(*) AS total,
                SUM(CASE WHEN sentiment_label IS NULL OR TRIM(COALESCE(sentiment_label, '')) = '' THEN 1 ELSE 0 END) AS without_label,
                SUM(CASE WHEN sentiment_label IS NOT NULL AND TRIM(sentiment_label) != '' THEN 1 ELSE 0 END) AS with_label
            FROM news_articles
            GROUP BY news_type
            ORDER BY news_type
        """)
        by_type = cur.fetchall()

        # ---- By ticker (optional) ----
        cur.execute("""
            SELECT
                ticker,
                COUNT(*) AS total,
                SUM(CASE WHEN sentiment_label IS NOT NULL AND TRIM(sentiment_label) != '' THEN 1 ELSE 0 END) AS with_label
            FROM news_articles
            GROUP BY ticker
            ORDER BY total DESC
            LIMIT 15
        """)
        by_ticker = cur.fetchall()

    # Print report
    print()
    print("=" * 70)
    print("NEWS_ARTICLES — Sentiment coverage (sentiment_label / sentiment_score)")
    print("=" * 70)
    print()
    print("Overall (all articles in news_articles):")
    print(f"  Total articles:              {total}")
    print(f"  WITH sentiment_label:        {with_label}  ({100 * with_label / total:.1f}%)" if total else "  (no rows)")
    print(f"  WITHOUT sentiment_label:      {without_label}  ({100 * without_label / total:.1f}%)" if total else "  (no rows)")
    print(f"  WITH sentiment_score:        {with_score}  ({100 * with_score / total:.1f}%)" if total else "")
    print(f"  WITHOUT sentiment_score:     {without_score}  ({100 * without_score / total:.1f}%)" if total else "")
    print()
    print("By news_type (stock = AV + Finnhub; market = Finnhub only):")
    print("-" * 70)
    for r in by_type:
        t = r['total'] or 0
        w = r['with_label'] or 0
        wo = r['without_label'] or 0
        pct = 100 * w / t if t else 0
        print(f"  {r['news_type']:10}  total: {t:5}   with sentiment: {w:5} ({pct:5.1f}%)   without: {wo:5}")
    print()
    print("By ticker (top 15 by article count):")
    print("-" * 70)
    for r in by_ticker:
        t = r['total'] or 0
        w = r['with_label'] or 0
        pct = 100 * w / t if t else 0
        print(f"  {r['ticker']:8}  total: {t:5}   with sentiment: {w:5} ({pct:5.1f}%)")
    print()
    print("=" * 70)
    print("Note: Only Alpha Vantage provides sentiment per article. If the API")
    print("only returns ~10 days of data, most rows will be Finnhub → no sentiment.")
    print("=" * 70)
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
