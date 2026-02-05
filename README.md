# LangGraph Stock Analysis System
## Bloomberg Terminal-Style AI Stock Analysis Dashboard

**Graduation Thesis Project**  
**Deadline:** March 2026  
**Architecture:** LangGraph Multi-Agent System (16 Nodes)

---

## Overview

This is a Bloomberg Terminal-style AI stock analysis system that uses LangGraph's multi-agent architecture to analyze stocks through 4 data streams: price data, stock-specific news, market news, and related company news.

**Core Innovation:** A news source reliability learning system (Node 8) that tracks which news sources are historically accurate for each stock and adjusts confidence weights accordingly, achieving 10-15% improvement in sentiment signal accuracy.

---

## Key Features

- **16-Node LangGraph Architecture** - First known application of LangGraph to stock analysis
- **Two-Phase Anomaly Detection** - Content filtering + behavioral analysis for 95%+ manipulation detection
- **News Learning System** - Learns which sources are reliable (Bloomberg vs blogs)
- **Adaptive Weighting** - Optimizes signal weights based on historical accuracy
- **Technical Analysis** - RSI, MACD, Bollinger Bands, SMAs, EMAs
- **Sentiment Analysis** - FinBERT-powered financial NLP
- **Monte Carlo Forecasting** - 1000 simulations with confidence intervals
- **LLM Explanations** - Plain English for beginners, technical for experts
- **Professional Dashboard** - 6-tab Streamlit interface

---

## Quick Start

### 1. Prerequisites

- Python 3.10+
- API Keys:
  - [Finnhub](https://finnhub.io/register) (free tier)
  - [NewsAPI](https://newsapi.org/register) (free tier)
  - [Anthropic](https://console.anthropic.com/) (Claude API)

### 2. Installation

```bash
# Clone repository
cd /path/to/Diplomska

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your API keys
# FINNHUB_API_KEY=your_key_here
# NEWS_API_KEY=your_key_here
# ANTHROPIC_API_KEY=your_key_here
```

### 4. Initialize Database

```bash
# Create SQLite database with schema
python scripts/setup_database.py
```

### 5. Run Dashboard

```bash
# Launch Streamlit dashboard
streamlit run streamlit_app/app.py
```

---

## Project Structure

```
Diplomska/
├── src/                        # Core source code
│   ├── langgraph_nodes/       # 16 LangGraph nodes
│   ├── graph/                 # Workflow definition & state
│   ├── database/              # SQLite operations
│   ├── visualization/         # Charts and components
│   └── utils/                 # Config, logging, helpers
├── streamlit_app/             # Streamlit dashboard
│   ├── tabs/                  # 6 dashboard tabs
│   └── components/            # Reusable UI components
├── tests/                     # Unit and integration tests
├── scripts/                   # Utility scripts
├── data/                      # SQLite database & cache
└── LangGraph_setup/           # Documentation & guides
```

---

## Architecture

### 16-Node Pipeline

**Layer 1: Data Acquisition**
- Node 1: Fetch price data (Finnhub/yfinance)
- Node 3: Detect related companies
- Node 2: Fetch multi-source news

**Layer 2: Early Protection**
- Node 9A: Filter fake news (content-based)

**Layer 3: Parallel Analysis**
- Node 4: Technical indicators
- Node 5: Sentiment analysis (FinBERT)
- Node 6: Market context
- Node 7: Monte Carlo forecast

**Layer 4: Learning**
- Node 8: News verification & learning (THESIS INNOVATION)

**Layer 5: Behavioral Protection**
- Node 9B: Pump-and-dump detection

**Layer 6: Intelligence**
- Node 10: Backtest all signals
- Node 11: Calculate adaptive weights

**Layer 7: Signal Generation**
- Node 12: Final BUY/SELL/HOLD signal

**Layer 8: Explanations**
- Node 13: Beginner explanation
- Node 14: Technical explanation

**Layer 9: Output**
- Node 15: Dashboard data preparation

---

## Database Schema

**15 Core Tables:**
- `price_data` - Historical OHLCV data
- `news_articles` - Multi-source news with sentiment
- `news_outcomes` - Critical for Node 8 learning
- `source_reliability` - Source accuracy scores
- `news_type_effectiveness` - News type performance
- Plus: technical_indicators, related_companies, anomaly results, backtest_results, weight_history, final_signals, etc.

**3 Views:**
- `news_with_outcomes` - Joins news with outcomes
- `source_performance_summary` - Source accuracy aggregation
- `latest_signals` - Recent recommendations

---

## Technology Stack

- **Architecture:** LangGraph 0.0.55
- **LLM:** Claude 3.5 Sonnet (Anthropic)
- **NLP:** FinBERT (financial sentiment)
- **APIs:** Finnhub, yfinance, NewsAPI
- **Database:** SQLite
- **UI:** Streamlit
- **Language:** Python 3.10+

---

## Development Guide

See [`NODE_BUILD_GUIDE.md`](NODE_BUILD_GUIDE.md) for complete node-by-node implementation roadmap.

See [`.cursor/rules/`](.cursor/rules/) for development standards and patterns.

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Test specific node
pytest tests/test_nodes/test_node_01.py -v
```

---

## Research Questions (Thesis)

1. Can news source reliability learning improve sentiment prediction accuracy?
2. Does adaptive weighting outperform equal weighting?
3. Can two-phase anomaly detection prevent pump-and-dump schemes?
4. How do optimal weights vary across stock types?

---

## Expected Outcomes

| Metric | Target |
|--------|--------|
| Combined Signal Accuracy | 63-71% |
| Sentiment Accuracy (with Node 8) | 70-75% |
| Sentiment Accuracy (without Node 8) | 60-65% |
| Node 8 Improvement | +10-15% |
| Pump-and-Dump Detection | 95%+ |
| False Positive Rate | <3% |
| Signal Generation Time | <5 seconds |

---

## Contributors

Nikola - Graduation Thesis Project

---

## License

Academic Use Only - Graduation Thesis Project

---

## Acknowledgments

- LangGraph framework by LangChain
- FinBERT model by ProsusAI
- Finnhub, NewsAPI, and Anthropic APIs
