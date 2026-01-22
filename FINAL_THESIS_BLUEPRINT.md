# AI Stock Analysis Tool - Final Thesis Blueprint & Roadmap
## Bloomberg Terminal-Lite with Adaptive Multi-Source Intelligence

**Project Deadline:** March 2026  
**Timeline:** 14 Weeks (2 hours/day = ~200 hours total)  
**Academic Level:** Graduation/Thesis Project  
**Last Updated:** January 2025

---

## 🎯 Executive Summary

### What You're Building
A **Bloomberg Terminal-style AI stock analysis dashboard** that:
- Aggregates data from **4 distinct streams** (price, stock news, market news, related companies)
- Performs **technical analysis** with 6+ indicators (RSI, MACD, Bollinger Bands, SMAs, EMAs)
- Conducts **sentiment analysis** using FinBERT (financial domain NLP)
- Generates **Monte Carlo price forecasts** with probability distributions
- Implements **smart adaptive weighting** that learns which signals work best per stock
- Detects **anomalies and manipulation** (pump-and-dump, crashes, keyword alerts)
- Presents everything in a **professional Streamlit dashboard**

### Key Innovation (Thesis Differentiator)
**Multi-Source Adaptive Weighting with News Verification Feedback Loop:**
- System backtests each signal stream independently
- Learns optimal weights based on verified historical accuracy
- Automatically adjusts weights as new data arrives
- Includes novel pump-and-dump detection system

### Expected Outcome
| Metric | Target |
|--------|--------|
| Combined Signal Accuracy | 63-71% |
| Technical Signal Accuracy | 50-60% |
| Sentiment Signal Accuracy | 60-70% |
| Signal Generation Time | < 5 seconds |
| Dashboard Load Time | < 5 seconds |
| Stocks Analyzed for Thesis | 10-15 |

---

## 🛠️ Tech Stack (Final Decisions)

### Core Framework
| Component | Technology | Justification |
|-----------|------------|---------------|
| **UI Framework** | Streamlit | 95% AI accuracy, 5x faster dev, pure Python |
| **Database** | SQLite | Zero setup, built-in Python, sufficient for scale |
| **Language** | Python 3.10 | Best ML/data science ecosystem |

### Libraries
```
# Data & APIs
finnhub-python==2.4.19     # Primary data (60 req/min free)
yfinance==0.2.28           # Backup historical data
newsapi-python==0.2.7      # Additional news sources
pandas==2.1.0              # Data manipulation
numpy==1.24.0              # Numerical computing

# Technical Analysis
pandas-ta==0.3.14b         # 130+ technical indicators

# Machine Learning / NLP
transformers==4.33.0       # Hugging Face (FinBERT)
torch==2.0.1               # PyTorch backend
scikit-learn==1.3.0        # Isolation Forest (anomaly detection)

# Visualization
plotly==5.17.0             # Interactive charts
streamlit==1.28.0          # Dashboard framework

# Statistics
scipy==1.11.2              # Monte Carlo, statistical functions

# Utilities
python-dotenv==1.0.0       # Environment variables
```

---

## 📂 Project Architecture

### Directory Structure
```
stock-analysis-tool/
│
├── data/
│   ├── stock_prices.db              # SQLite database
│   └── cache/                       # Cached API responses
│
├── src/
│   ├── data_fetching/
│   │   ├── __init__.py
│   │   ├── price_fetcher.py         # Finnhub/yfinance price data
│   │   ├── news_fetcher.py          # Stock-specific news
│   │   └── multi_source_news_fetcher.py  # Market + related co news
│   │
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── technical_indicators.py  # RSI, MACD, BB, SMAs
│   │   ├── sentiment_analysis.py    # FinBERT integration
│   │   ├── monte_carlo.py           # Price simulations
│   │   ├── signal_generator.py      # Combined signals
│   │   ├── related_companies_detector.py  # Find connected stocks
│   │   └── anomaly_detectors.py     # All anomaly detection
│   │
│   ├── backtesting/
│   │   ├── __init__.py
│   │   ├── technical_backtest.py    # Test technical accuracy
│   │   ├── sentiment_backtest.py    # Test news accuracy
│   │   ├── multi_stream_backtester.py   # All streams
│   │   ├── weight_calculator.py     # Basic weights
│   │   ├── advanced_weight_calculator.py  # 4-stream weights
│   │   └── news_impact_verifier.py  # Feedback loop verification
│   │
│   ├── database/
│   │   ├── __init__.py
│   │   ├── db_manager.py            # All CRUD operations
│   │   └── schema.sql               # Complete database schema
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── charts.py                # Plotly charts
│   │   ├── weight_visualizer.py     # Weight explanation charts
│   │   └── dashboard_components.py  # Reusable UI elements
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py                # API keys, settings
│       └── helpers.py               # Utility functions
│
├── tests/
│   ├── test_indicators.py
│   ├── test_sentiment.py
│   ├── test_backtesting.py
│   └── test_anomaly_detection.py
│
├── app.py                           # Main Streamlit app
├── requirements.txt                 # Dependencies
├── README.md                        # Documentation
├── .env                             # API keys (not in git)
└── .gitignore
```

### Data Flow Architecture
```
┌─────────────────────────────────────────────────────────────────────┐
│                     DATA ACQUISITION LAYER                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│   │Stock Prices  │  │ Stock News   │  │ Market News  │             │
│   │ (Finnhub)    │  │  (NewsAPI)   │  │  (NewsAPI)   │             │
│   └──────┬───────┘  └──────┬───────┘  └──────┬───────┘             │
│          │                 │                 │                      │
│          └────────────────┬┴─────────────────┘                      │
│                           │                                         │
│                ┌──────────▼──────────┐                              │
│                │ Related Companies   │                              │
│                │ Detector            │                              │
│                └──────────┬──────────┘                              │
│                           │                                         │
│                ┌──────────▼──────────┐                              │
│                │ Related Co. News    │                              │
│                └──────────┬──────────┘                              │
│                           │                                         │
└───────────────────────────┼─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     SQLite DATABASE                                 │
├─────────────────────────────────────────────────────────────────────┤
│  • price_data           • sentiment_daily     • backtest_results    │
│  • stock_news           • weight_history      • prediction_accuracy │
│  • market_news          • weight_components   • anomaly_results     │
│  • related_companies    • related_company_news                      │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     ANALYSIS ENGINE                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌────────────────┐  ┌────────────────┐  ┌───────────────────┐     │
│  │   Technical    │  │  Multi-Source  │  │   Monte Carlo     │     │
│  │   Indicators   │  │   Sentiment    │  │   Simulation      │     │
│  └───────┬────────┘  └───────┬────────┘  └────────┬──────────┘     │
│          │                   │                    │                 │
│          └───────────────────┴────────────────────┘                 │
│                              │                                      │
│                   ┌──────────▼──────────┐                           │
│                   │  Anomaly Detection  │                           │
│                   └──────────┬──────────┘                           │
│                              │                                      │
└──────────────────────────────┼──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│              ADAPTIVE WEIGHT CALCULATION ENGINE                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Input Streams (independently backtested):                          │
│  1. Technical Indicators → Accuracy: ~55%                           │
│  2. Stock-Specific News  → Accuracy: ~65%                           │
│  3. Market News          → Accuracy: ~58%                           │
│  4. Related Companies    → Accuracy: ~60%                           │
│                                                                     │
│  Weight Formula:                                                    │
│  weight_i = accuracy_i / sum(all_accuracies)                        │
│                                                                     │
│  Output: Combined signal with confidence level                      │
│                                                                     │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     STREAMLIT DASHBOARD                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Tab 1: Price & Indicators   Tab 2: Monte Carlo Forecast            │
│  Tab 3: News & Sentiment     Tab 4: Signal Weights                  │
│  Tab 5: Risk Alerts          Tab 6: Export                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📅 14-Week Implementation Timeline

### PHASE 1: Foundation (Week 1-3) ⬜

#### Week 1: Environment Setup & Basic Data
| Day | Task | Deliverable |
|-----|------|-------------|
| 1-2 | Project setup (Conda, Git, folder structure) | Working environment |
| 3-4 | Finnhub API integration, price fetcher | Fetch OHLCV data |
| 5 | SQLite database setup, basic schema | Database operational |
| 6-7 | Basic candlestick chart in Streamlit | First visual working |

**Milestone:** `streamlit run app.py` shows price chart

#### Week 2: Technical Indicators
| Day | Task | Deliverable |
|-----|------|-------------|
| 1-2 | RSI, MACD implementation (pandas-ta) | Indicators calculated |
| 3-4 | Bollinger Bands, SMAs, EMAs | All 6 indicators working |
| 5-6 | Technical signal generation logic | BUY/SELL/HOLD signals |
| 7 | Add indicators to chart overlay | Visual indicator display |

**Milestone:** Technical analysis fully functional

#### Week 3: Multi-Source News Pipeline
| Day | Task | Deliverable |
|-----|------|-------------|
| 1-2 | NewsAPI integration, stock news fetcher | Stock news working |
| 3-4 | Market news fetcher | Market news working |
| 5-6 | Related companies detector (basic) | Finds competitors |
| 7 | Store all news in database | Complete news pipeline |

**Milestone:** 3 news streams collecting data

---

### PHASE 2: Intelligence Layer (Week 4-7) ⬜

#### Week 4: Sentiment Analysis
| Day | Task | Deliverable |
|-----|------|-------------|
| 1-2 | FinBERT setup and testing | Model loaded, working |
| 3-4 | Batch sentiment processing | Analyze multiple articles |
| 5-6 | Sentiment aggregation (daily scores) | Combined sentiment |
| 7 | Store sentiment in database | Historical tracking |

**Milestone:** FinBERT analyzing all news streams

#### Week 5: Backtesting Framework
| Day | Task | Deliverable |
|-----|------|-------------|
| 1-2 | Technical signal backtester | Test technical accuracy |
| 3-4 | Stock news sentiment backtester | Test news accuracy |
| 5-6 | Market news & related co backtester | All 4 streams tested |
| 7 | Store backtest results, accuracy metrics | Database updated |

**Milestone:** All signal streams have measured accuracy

#### Week 6: Adaptive Weight Calculator
| Day | Task | Deliverable |
|-----|------|-------------|
| 1-2 | Weight calculation algorithm | Proportional weights |
| 3-4 | Combined signal generator | Final BUY/SELL/HOLD |
| 5-6 | Weight component details | Detailed breakdown |
| 7 | Weight history tracking | Historical weights stored |

**Milestone:** Adaptive weighting system complete

#### Week 7: Monte Carlo & Forecasting
| Day | Task | Deliverable |
|-----|------|-------------|
| 1-2 | Geometric Brownian Motion implementation | Basic simulation |
| 3-4 | 1000 simulations, percentiles | Confidence intervals |
| 5-6 | Fan chart visualization | Beautiful forecast chart |
| 7 | Probability calculations | Expected price ranges |

**Milestone:** Price forecasting working

---

### PHASE 3: Protection & Visualization (Week 8-10) ⬜

#### Week 8: Anomaly Detection System
| Day | Task | Deliverable |
|-----|------|-------------|
| 1 | Price anomaly detector (z-score) | Crash/spike detection |
| 2 | Volume anomaly detector | Unusual volume alerts |
| 3 | Keyword alert system | Danger word scanning |
| 4 | News surge detector | Breaking news detection |
| 5-6 | **Pump-and-dump detector** (key feature!) | Manipulation detection |
| 7 | Volatility & divergence detectors | Complete protection |

**Milestone:** All 7 anomaly detectors working

#### Week 9: Dashboard Integration
| Day | Task | Deliverable |
|-----|------|-------------|
| 1-2 | Main dashboard layout (6 tabs) | Structure complete |
| 3-4 | Tab 1: Price & Indicators | Interactive charts |
| 5 | Tab 2: Monte Carlo Forecast | Fan charts |
| 6 | Tab 3: News & Sentiment | Headlines + scores |
| 7 | Tab 4: Signal Weights | Weight breakdown |

**Milestone:** Core 4 tabs functional

#### Week 10: Polish & Advanced Features
| Day | Task | Deliverable |
|-----|------|-------------|
| 1-2 | Tab 5: Risk Alerts | Anomaly warnings |
| 3-4 | Weight visualization charts | Interactive explanations |
| 5 | Tab 6: Export (PDF, CSV) | Download functionality |
| 6-7 | Error handling, edge cases | Robust application |

**Milestone:** Production-ready dashboard

---

### PHASE 4: Testing & Documentation (Week 11-12) ⬜

#### Week 11: Testing
| Day | Task | Deliverable |
|-----|------|-------------|
| 1-2 | Unit tests (indicators, sentiment) | 70%+ coverage |
| 3-4 | Integration tests (data pipeline) | End-to-end working |
| 5-6 | Multi-stock testing (10-15 stocks) | Real-world validation |
| 7 | Bug fixes, performance optimization | Stable system |

**Milestone:** Tested, stable application

#### Week 12: Documentation
| Day | Task | Deliverable |
|-----|------|-------------|
| 1-2 | README with setup instructions | User guide |
| 3-4 | Code documentation (docstrings) | Developer guide |
| 5-6 | Methodology documentation | Academic reference |
| 7 | Screenshots, diagrams | Visual documentation |

**Milestone:** Complete documentation

---

### PHASE 5: Thesis Writing (Week 13-14) ⬜

#### Week 13: Thesis Draft
| Day | Task | Deliverable |
|-----|------|-------------|
| 1-2 | Introduction, Problem Statement | 5 pages |
| 3-4 | Literature Review | 10-15 pages |
| 5-6 | Methodology (algorithms, architecture) | 15-20 pages |
| 7 | Implementation details | 15-20 pages |

**Milestone:** First draft complete (~45 pages)

#### Week 14: Thesis Completion & Defense Prep
| Day | Task | Deliverable |
|-----|------|-------------|
| 1-2 | Results & Evaluation | 10-15 pages |
| 3-4 | Discussion, Conclusion | 10-15 pages |
| 5 | Final editing, formatting | Complete thesis |
| 6 | Create presentation slides | 20-minute presentation |
| 7 | Practice defense, demo prep | Defense ready |

**Milestone:** Thesis submitted, defense ready

---

## 🔑 Feature Implementation Checklist

### Core Features (Must-Have)

#### Data Fetching
- [ ] Historical price data (Finnhub)
- [ ] Real-time quotes
- [ ] Stock-specific news
- [ ] Market-wide news
- [ ] Related companies detection
- [ ] Related companies news
- [ ] Database caching

#### Technical Analysis
- [ ] RSI (Relative Strength Index)
- [ ] MACD (Moving Average Convergence Divergence)
- [ ] Bollinger Bands
- [ ] SMA 20, SMA 50
- [ ] EMA 12, EMA 26
- [ ] Volume analysis
- [ ] Technical signal generation

#### Sentiment Analysis
- [ ] FinBERT integration
- [ ] Stock news sentiment scoring
- [ ] Market news sentiment scoring
- [ ] Related companies sentiment
- [ ] Sentiment aggregation (daily)
- [ ] Historical sentiment tracking

#### Monte Carlo Simulation
- [ ] Geometric Brownian Motion
- [ ] 1000+ simulations
- [ ] Confidence intervals (68%, 95%)
- [ ] Fan chart visualization
- [ ] Probability calculations

#### Smart Weighting System
- [ ] Technical backtesting
- [ ] Stock news backtesting
- [ ] Market news backtesting
- [ ] Related companies backtesting
- [ ] Proportional weight calculation
- [ ] Combined signal generation
- [ ] Weight history tracking
- [ ] Weight visualization

#### Anomaly Detection
- [ ] Price anomaly detector (z-score)
- [ ] Volume anomaly detector
- [ ] Keyword alert system
- [ ] News surge detector
- [ ] **Pump-and-dump detector** ⭐ (Key Innovation)
- [ ] Volatility anomaly detector
- [ ] News-price divergence detector

#### Dashboard
- [ ] Multi-tab interface
- [ ] Interactive candlestick charts
- [ ] Indicator overlays
- [ ] Monte Carlo forecast visualization
- [ ] News feed with sentiment
- [ ] Signal display (BUY/SELL/HOLD)
- [ ] Weight explanation panel
- [ ] Risk alerts section
- [ ] Export (PDF/CSV)

### Advanced Features (Nice-to-Have)
- [ ] News impact verification feedback loop
- [ ] ML pattern learning (Random Forest)
- [ ] Multi-stock comparison
- [ ] Portfolio analysis
- [ ] Email alerts
- [ ] Dark mode

---

## 🗄️ Database Schema Summary

### Core Tables
| Table | Purpose |
|-------|---------|
| `price_data` | OHLCV + technical indicators |
| `stock_news` | Stock-specific news + sentiment |
| `market_news` | Market-wide news + sentiment |
| `related_companies` | Company relationships |
| `related_company_news` | News about related stocks |
| `sentiment_daily` | Aggregated daily sentiment |

### Weight System Tables
| Table | Purpose |
|-------|---------|
| `backtest_results` | Individual predictions vs actuals |
| `prediction_accuracy` | Accuracy metrics per signal type |
| `weight_history` | Historical weight calculations |
| `weight_components` | Detailed weight breakdown |
| `related_companies_impact` | Impact of each related company |

### Anomaly Detection Tables
| Table | Purpose |
|-------|---------|
| `anomaly_detection_results` | All anomaly findings |
| `pump_dump_tracking` | Pump-and-dump analysis |
| `keyword_alerts` | Dangerous keyword history |

---

## 🎓 Thesis Structure (50-80 pages)

### 1. Introduction (5 pages)
- Problem statement: Information overload in stock markets
- Solution: AI-powered multi-source aggregation
- Objectives and scope
- Research questions:
  1. Does sentiment analysis add value over technical analysis alone?
  2. How do optimal weights vary across stock types?
  3. Does adaptive weighting improve accuracy vs equal weighting?

### 2. Literature Review (10-15 pages)
- Technical analysis theory and effectiveness studies
- Sentiment analysis in financial markets
- FinBERT and financial NLP
- Monte Carlo methods in finance
- Existing tools (Bloomberg, Yahoo Finance)
- Gaps in current solutions

### 3. Methodology (15-20 pages)
- System architecture (diagrams)
- Data sources and APIs
- Technical indicator algorithms
- FinBERT sentiment analysis approach
- Monte Carlo simulation methodology
- **Adaptive weighting algorithm** (key contribution)
- Anomaly detection methods
- Backtesting methodology

### 4. Implementation (15-20 pages)
- Technology stack justification
- Database design
- Key algorithms (with code snippets)
- Dashboard design principles
- Challenges and solutions

### 5. Results & Evaluation (10-15 pages)
- Case studies (10-15 stocks analyzed)
- Accuracy comparison tables
- Weight distribution analysis
- Anomaly detection effectiveness
- Performance metrics
- User testing feedback

### 6. Discussion (8-10 pages)
- Interpretation of findings
- Comparison with existing tools
- Limitations (API constraints, model assumptions)
- Future improvements

### 7. Conclusion (3-5 pages)
- Summary of achievements
- Contribution to field
- Recommendations for future work

---

## ⚠️ Risk Mitigation

### Technical Risks
| Risk | Mitigation |
|------|------------|
| API rate limits | Implement caching, use multiple sources |
| FinBERT slow | Batch processing, cache results |
| Data gaps | Handle None gracefully, fallback sources |
| Streamlit reruns | Aggressive caching (@st.cache_data) |

### Academic Risks
| Risk | Mitigation |
|------|------------|
| Low accuracy | Focus on improvement over baseline, not absolute |
| Time overrun | Start with MVP, add features incrementally |
| Data quality | Acknowledge limitations, document assumptions |

### Realistic Expectations
- **Won't beat the market consistently**
- **Not suitable for actual trading without risk management**
- **Accuracy of ~60-70% is excellent for this domain**
- **Focus on demonstrating the methodology, not guaranteed profits**

---

## 📊 Success Metrics

### Development Milestones
- [ ] Week 3: Basic price chart + technical indicators
- [ ] Week 5: Sentiment analysis integrated
- [ ] Week 7: Adaptive weighting functional
- [ ] Week 10: Complete polished dashboard
- [ ] Week 12: Documentation complete
- [ ] Week 14: Thesis + defense ready

### Quality Metrics
- [ ] Code coverage: 70%+
- [ ] Dashboard load time: < 5 seconds
- [ ] Signal generation: < 5 seconds
- [ ] Combined signal accuracy: 60%+
- [ ] User satisfaction: 4/5+ from testers

### Academic Metrics
- [ ] Thesis length: 50-80 pages
- [ ] Stocks analyzed: 10-15
- [ ] Data points: 10,000+ prices, 1,000+ news articles
- [ ] Statistical significance: p < 0.05
- [ ] Working demo for defense

---

## 💡 Key Commands Reference

### Setup
```bash
conda create -n stock-tool python=3.10
conda activate stock-tool
pip install -r requirements.txt
```

### Run Application
```bash
streamlit run app.py
```

### Database Access
```bash
sqlite3 data/stock_prices.db
.tables
.schema price_data
```

### Git Workflow
```bash
git add .
git commit -m "Feature: Added pump-and-dump detector"
git push origin main
```

### Testing
```bash
python -m pytest tests/
python -m pytest tests/ -v --cov=src
```

---

## 🎯 Defense Preparation

### Anticipated Questions
1. "Why Streamlit over Dash or React?"
   - 95% AI accuracy, 5x faster development, pure Python
   
2. "How accurate is your Monte Carlo model?"
   - Provides probability distributions, not point predictions
   - 68% of actual prices fall within 68% confidence interval
   
3. "What makes this better than Yahoo Finance?"
   - Multi-source aggregation, adaptive weighting, pump detection
   - Reduces analysis time from 30 min to 2 min per stock
   
4. "How do you handle API rate limits?"
   - SQLite caching, batch requests, smart rate limiting
   
5. "Why these specific indicators?"
   - Most widely used, well-documented, complementary signals

### Demo Checklist
- [ ] Pre-load 3-5 stocks (AAPL, TSLA, NVDA, SPY)
- [ ] Have GME historical data for pump detection demo
- [ ] Prepare offline mode (cached data) as backup
- [ ] Test on presentation laptop beforehand

---

## 🚀 Final Reminders

1. **Start immediately** - Don't overthink, begin coding today
2. **Commit daily** - Git is your backup and progress proof
3. **Test with real stocks** - AAPL, TSLA, NVDA, SPY, GME
4. **Document as you go** - Screenshots, notes, observations
5. **Focus on MVP first** - Features can be added later
6. **The pump detector is your differentiator** - Make it work well
7. **Adaptive weighting is your contribution** - Explain it clearly
8. **Manage expectations** - 60-70% accuracy is excellent
9. **Practice your defense** - Know your code inside-out
10. **You have 200 hours** - That's plenty if you stay consistent!

---

## 📞 Resource Quick Links

| Resource | URL |
|----------|-----|
| Finnhub API | https://finnhub.io/docs/api |
| Streamlit Docs | https://docs.streamlit.io |
| pandas-ta | https://github.com/twopirllc/pandas-ta |
| FinBERT | https://huggingface.co/ProsusAI/finbert |
| Plotly | https://plotly.com/python/ |
| NewsAPI | https://newsapi.org/docs |

---

**You have everything you need. The path is clear. Let's build this! 🚀**

---

*This blueprint synthesizes: PROJECT_REFERENCE.md, implementation_guide.md, bloomberg_lite_roadmap.md, NEWS_VERIFICATION_LEARNING_SYSTEM.md, ENHANCED_MODULAR_BREAKDOWN.md, and ANOMALY_DETECTION_IMPLEMENTATION.md into one actionable document.*
