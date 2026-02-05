---
description: "SQLite database patterns - context managers and parameterized queries"
alwaysApply: true
---

# Database Operations Standards

## Required Pattern: Context Manager + Parameterized Queries

```python
import sqlite3

def get_cached_data(ticker: str, max_age_hours: int = 24) -> Optional[pd.DataFrame]:
    """Get cached data from database"""
    try:
        with sqlite3.connect('data/stock_prices.db') as conn:
            query = """
                SELECT * FROM price_data 
                WHERE ticker = ? 
                AND created_at > datetime('now', ?)
                ORDER BY date DESC
            """
            params = (ticker, f'-{max_age_hours} hours')
            df = pd.read_sql_query(query, conn, params=params)
            return df if not df.empty else None
    except Exception as e:
        logger.error(f"Database read failed: {str(e)}")
        return None
```

## Rules

- **ALWAYS use context managers** (`with` statement)
- **ALWAYS use parameterized queries** (`?` placeholders)
- **NEVER use f-strings in SQL queries** (SQL injection risk)
- Commit transactions explicitly when writing
- Handle exceptions gracefully (return None on failure)

## What NOT to Do

```python
# ❌ BAD - SQL injection risk
query = f"SELECT * FROM price_data WHERE ticker = '{ticker}'"

# ❌ BAD - No context manager
conn = sqlite3.connect('data.db')
cursor = conn.cursor()
# ... forgot to close
```

## Reference

See `@src/database/db_manager.py` for all database operations.
