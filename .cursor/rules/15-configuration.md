---
description: "Environment variables and configuration management - never hardcode secrets"
alwaysApply: true
---

# Configuration Management

**Rule:** Never hardcode API keys or secrets. Use environment variables.

## Required Setup

Create `.env` file in project root:

```bash
# API Keys
FINNHUB_API_KEY=your_key_here
NEWS_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here

# Optional Settings
DATABASE_PATH=data/stock_prices.db
CACHE_HOURS=24
LOG_LEVEL=INFO
```

## Loading Configuration

```python
from dotenv import load_dotenv
import os

# Load environment variables (do once at startup)
load_dotenv()

# Access configuration
FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY')
NEWS_API_KEY = os.getenv('NEWS_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

# Provide defaults for optional settings
DATABASE_PATH = os.getenv('DATABASE_PATH', 'data/stock_prices.db')
CACHE_HOURS = int(os.getenv('CACHE_HOURS', '24'))

# Validate required keys
if not FINNHUB_API_KEY:
    raise ValueError("FINNHUB_API_KEY not found in .env file")
```

## .gitignore

**CRITICAL:** Add to `.gitignore`:

```
.env
*.db
__pycache__/
*.pyc
.pytest_cache/
```

## What to Store in .env

- ✅ API keys
- ✅ Database paths
- ✅ Optional settings (cache duration, log level)
- ❌ Code
- ❌ Version controlled files
- ❌ Large data files
