# AI Stock Analysis Tool - Thesis Overview
## Bloomberg Terminal-Lite with Adaptive Multi-Source Intelligence

**Project Deadline:** March 2026  
**Timeline:** 14 Weeks (2 hours/day = ~200 hours total)  
**Academic Level:** Graduation/Thesis Project  
**Architecture:** LangGraph Multi-Agent System  
**Last Updated:** February 2026

---

## 🎯 What You're Building

### Core Concept
A **Bloomberg Terminal-style AI stock analysis dashboard** powered by a LangGraph multi-agent architecture that provides intelligent, explainable stock analysis for everyday investors.

### The System Aggregates 4 Data Streams:
1. **Price Data** - Historical OHLCV data with technical indicators
2. **Stock-Specific News** - News articles about the target company
3. **Market News** - Broader market and sector news
4. **Related Companies** - News and signals from competitors/suppliers

### The System Performs:
- ✅ **Technical Analysis** with 6+ indicators (RSI, MACD, Bollinger Bands, SMAs, EMAs)
- ✅ **Sentiment Analysis** using FinBERT (financial domain NLP)
- ✅ **Monte Carlo Price Forecasts** with probability distributions
- ✅ **Adaptive Weighting** that learns which signals work best per stock
- ✅ **Two-Phase Anomaly Detection** (content filtering + behavior analysis)
- ✅ **Pump-and-Dump Detection** for investor protection
- ✅ **LLM-Powered Explanations** for beginners and experts
- ✅ **Professional Streamlit Dashboard** with 6 interactive tabs

---

## 💡 Key Innovation (Thesis Differentiator)

### 1. LangGraph Multi-Agent Architecture
**First known application of LangGraph for financial stock analysis**
- 16 independent nodes processing in parallel
- Intelligent state management between agents
- Conditional execution based on risk detection
- Novel approach to financial intelligence systems

### 2. Two-Phase Anomaly Detection System
**Content Filtering + Behavioral Analysis**

**Phase 1 - Early Detection (Node 9A):**
- Filters fake news BEFORE sentiment analysis
- Protects learning system from contaminated data
- Checks: keyword alerts, news surges, source credibility, coordinated posting

**Phase 2 - Late Detection (Node 9B):**
- Detects sophisticated manipulation AFTER historical learning
- Identifies pump-and-dump schemes from behavioral patterns
- Checks: price anomalies, volume spikes, news-price divergence, volatility

### 3. News Impact Learning System (Node 8) 🌟 CORE THESIS INNOVATION
**The system learns from historical news-price correlations to identify reliable sources**

**How It Works:**
1. **Historical Data Collection** - Analyzes 6 months of past news + price movements
2. **Outcome Tracking** - For each historical article, checks if prediction came true
3. **Source Reliability Scoring** - Calculates which news sources are accurate per stock
4. **Confidence Adjustment** - Boosts reliable sources, ignores unreliable ones
5. **Continuous Learning** - Gets smarter as more historical data accumulates

**Example for NVIDIA:**
- Bloomberg.com: 85% accurate historically → Confidence boost 1.2x
- Random blog: 20% accurate historically → Confidence reduction 0.5x
- Today's Bloomberg article gets **higher weight** in final signal
- Today's random blog article gets **ignored** or heavily discounted

**Key Database Tables:**
- `news_outcomes` - Tracks what happened 7 days after each news article
- `source_reliability` - Stores accuracy scores per source per stock
- `news_type_effectiveness` - Tracks if stock/market/related news is more predictive

**Expected Impact:** +10-15% improvement in sentiment signal accuracy

### 4. Multi-Source Adaptive Weighting with Verification Feedback Loop
**The system learns which signals work best for each stock**
- Backtests each of 4 signal streams independently
- Calculates optimal weights based on verified historical accuracy
- Automatically adjusts weights as new data arrives
- Uses Node 8's news reliability scores to adjust news signal weights

Formula: `weight_i = accuracy_i / sum(all_accuracies)`

### 5. LLM-Powered Explainability
**Makes sophisticated finance accessible to beginners**
- Beginner explanations in plain English
- Technical explanations for advanced users
- Explains WHY signals were generated
- Shows which news sources were trusted vs ignored
- Transparent decision-making process

---

## 🎓 Research Questions

1. **Does sentiment analysis add value over technical analysis alone?**
   - Hypothesis: Multi-source sentiment provides 10-15% accuracy improvement

2. **Can news source reliability learning improve prediction accuracy?** 🌟 PRIMARY THESIS QUESTION
   - Hypothesis: Learning which sources are reliable improves sentiment accuracy by 10-15%
   - Sub-question: Do reliability scores vary significantly between stocks?
   - Sub-question: Does source reliability change over time?

3. **How do optimal weights vary across stock types?**
   - Hypothesis: Tech stocks weight news higher, stable stocks weight technical higher

4. **Does adaptive weighting improve accuracy vs equal weighting?**
   - Hypothesis: Adaptive weighting achieves 5-10% better accuracy

5. **Can two-phase anomaly detection prevent pump-and-dump schemes?**
   - Hypothesis: 95%+ detection rate with <3% false positives

---

## 📊 Expected Outcomes

| Metric | Target |
|--------|--------|
| Combined Signal Accuracy | 63-71% |
| Technical Signal Accuracy | 50-60% |
| Sentiment Signal Accuracy (without learning) | 60-65% |
| **Sentiment Signal Accuracy (with Node 8 learning)** | **70-75%** 🌟 |
| **News Learning Improvement** | **+10-15%** 🌟 |
| Source Reliability Accuracy | Bloomberg/Reuters: 80-90%, Blogs: 20-40% |
| Pump-and-Dump Detection Rate | 95%+ |
| False Positive Rate | <3% |
| Signal Generation Time | < 5 seconds |
| Dashboard Load Time | < 5 seconds |
| Stocks Analyzed for Thesis | 10-15 |
| Historical News Events per Stock | 100-200 events |
| Dashboard Load Time | < 5 seconds |
| Stocks Analyzed for Thesis | 10-15 |

---

## 🛠️ Technology Stack

### Core Framework
| Component | Technology | Justification |
|-----------|------------|---------------|
| **Architecture** | LangGraph | Multi-agent orchestration, parallel processing, state management |
| **LLM** | Claude 3.5 Sonnet | Best reasoning for explanations and analysis |
| **UI Framework** | Streamlit | Pure Python, fast development, interactive dashboards |
| **Database** | SQLite | Zero setup, built-in Python, sufficient for scale |
| **Language** | Python 3.10+ | Best ML/AI ecosystem |

### Key Libraries
```python
# LangGraph & LLM
langgraph==0.0.55              # Multi-agent orchestration
langchain==0.1.20              # LLM framework
langchain-anthropic==0.1.11    # Claude integration
anthropic==0.25.0              # Claude API

# Data & APIs
finnhub-python==2.4.19         # Primary price data
yfinance==0.2.28               # Backup historical data
newsapi-python==0.2.7          # News sources
pandas==2.1.0                  # Data manipulation
numpy==1.24.0                  # Numerical computing

# Technical Analysis
pandas-ta==0.3.14b             # 130+ technical indicators

# Machine Learning / NLP
transformers==4.33.0           # Hugging Face (FinBERT)
torch==2.0.1                   # PyTorch backend
scikit-learn==1.3.0            # Isolation Forest (anomaly detection)

# Visualization
plotly==5.17.0                 # Interactive charts
streamlit==1.28.0              # Dashboard framework

# Statistics
scipy==1.11.2                  # Monte Carlo, statistical functions

# Utilities
python-dotenv==1.0.0           # Environment variables
aiohttp==3.9.3                 # Async operations
```

---

## 📂 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INPUT (Ticker)                      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              LAYER 1: Data Acquisition                      │
│  Node 1: Price Data → Node 3: Related Companies →           │
│  Node 2: Multi-Source News                                  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│         LAYER 2: Early Anomaly Detection (Node 9A)          │
│  🛡️ FILTER FAKE NEWS BEFORE ANALYSIS                       │
│  • Keyword alerts  • News surge  • Source credibility       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│         LAYER 3: Parallel Analysis (Nodes 4-7)              │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │ Technical   │ │ Sentiment   │ │ Market      │           │
│  │ Analysis    │ │ (Clean      │ │ Context     │           │
│  │             │ │  News Only) │ │             │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
│  ┌─────────────┐                                            │
│  │ Monte Carlo │                                            │
│  │ Forecast    │                                            │
│  └─────────────┘                                            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│      LAYER 4: News Verification & Learning (Node 8)         │
│  Learn from CLEAN historical patterns                       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│    LAYER 5: Behavioral Anomaly Detection (Node 9B)          │
│  🚨 DETECT SOPHISTICATED MANIPULATION                       │
│  • Pump-and-dump  • Price anomalies  • Volume spikes        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│     LAYER 6: Backtesting & Adaptive Weights (Nodes 10-11)   │
│  Calculate optimal signal weights based on accuracy         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
                 ┌───────┴───────┐
                 │  CRITICAL     │
                 │  RISK?        │
                 └───┬───────┬───┘
                     │       │
              YES ───┘       └─── NO
               │                  │
               ▼                  ▼
        ┌─────────────┐    ┌─────────────┐
        │ HALT &      │    │ Generate    │
        │ Warn User   │    │ Signal      │
        └─────────────┘    └──────┬──────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────┐
│        LAYER 7: LLM Explanations (Nodes 13-14)              │
│  • Beginner explanation  • Technical explanation            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│         LAYER 8: Dashboard Preparation (Node 15)            │
│  Prepare all visualizations and export data                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  STREAMLIT DASHBOARD                        │
│  Tab 1: Price & Technical Indicators                        │
│  Tab 2: Monte Carlo Forecast                                │
│  Tab 3: News & Sentiment Analysis                           │
│  Tab 4: Signal Weights Visualization                        │
│  Tab 5: Risk & Anomaly Alerts                               │
│  Tab 6: Export & Reports                                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Database Schema (SQLite)

### Core Tables

**price_data**
- Historical OHLCV data
- Calculated technical indicators
- Volume analysis

**stock_news**
- Stock-specific news articles
- FinBERT sentiment scores
- News credibility scores

**market_news**
- Market-wide news
- Sector news
- Sentiment scores

**related_companies**
- Related company tickers
- Relationship type (competitor, supplier, customer)
- Correlation scores

**related_company_news**
- News about related companies
- Sentiment scores
- Impact analysis

### Analysis Tables

**sentiment_scores**
- Daily aggregated sentiment
- By source type (stock, market, related)
- Confidence scores

**technical_indicators**
- RSI, MACD, Bollinger Bands
- SMAs, EMAs
- Volume indicators

**monte_carlo_results**
- Simulation paths
- Confidence intervals
- Probability distributions

### Learning Tables (🌟 Node 8 Innovation)

**news_outcomes** 🌟 CRITICAL FOR LEARNING
- Historical news articles with their outcomes
- Price at time of news
- Price 1, 3, 7 days after news
- Predicted direction (from sentiment) vs actual direction
- Was the prediction accurate?
- Enables learning which sources are reliable

**source_reliability** 🌟 CRITICAL FOR LEARNING
- Reliability score per news source per stock
- Total articles analyzed
- Accurate vs inaccurate predictions
- Accuracy rate (0-100%)
- Confidence multiplier for adjusting sentiment
- Example: Bloomberg.com for NVDA = 85% accurate, 1.2x multiplier

**news_type_effectiveness**
- Effectiveness per news type (stock/market/related)
- Accuracy rate per type per stock
- Average price impact
- Used by adaptive weighting to adjust signal weights

**backtest_results**
- Historical accuracy by signal stream
- Sample sizes
- Confidence intervals

**weight_history**
- Adaptive weight evolution over time
- Per stock, per signal type
- Explanation metadata
- Incorporates Node 8 learning adjustments

### Anomaly Detection Tables

**early_anomaly_results**
- Keyword alerts triggered
- News surge detection
- Source credibility issues
- Coordinated posting detection

**behavioral_anomaly_results**
- Pump-and-dump scores
- Price/volume anomalies
- News-price divergence
- Manipulation probability

**anomaly_alerts**
- Combined risk assessments
- User warnings issued
- Trading halts triggered

---

## 🎓 Thesis Structure (50-80 pages)

### 1. Introduction (5 pages)
- **Problem:** Information overload in stock markets
- **Solution:** AI-powered multi-agent analysis with LangGraph
- **Objectives:** Democratize sophisticated analysis
- **Research Questions:** (Listed above)

### 2. Literature Review (10-15 pages)
- Technical analysis effectiveness
- Sentiment analysis in finance
- FinBERT and financial NLP
- Multi-agent systems in finance
- LangGraph and orchestration frameworks
- Anomaly detection and fraud prevention
- Existing tools (Bloomberg, Yahoo Finance)

### 3. Methodology (15-20 pages)
- **LangGraph Architecture Design**
  - Multi-agent orchestration
  - State management
  - Parallel processing
  - Conditional execution
- **News Impact Learning System (Node 8)** 🌟 PRIMARY CONTRIBUTION
  - Historical news-price correlation analysis
  - Source reliability calculation methodology
  - Confidence adjustment algorithms
  - Learning feedback loop design
  - Outcome tracking system
- **Two-Phase Anomaly Detection**
  - Content filtering methodology (Node 9A)
  - Behavioral analysis methodology (Node 9B)
  - How early filtering protects learning
- **Adaptive Weighting Algorithm**
  - Backtesting approach
  - Weight calculation formulas
  - Integration with Node 8 learning
- **Data Sources & APIs**
- **Technical Indicators**
- **FinBERT Sentiment Analysis**
- **Monte Carlo Simulation**
- **LLM Explanation Generation**

### 4. Implementation (15-20 pages)
- **Technology Stack Justification**
  - Why LangGraph?
  - Why Claude 3.5 Sonnet?
  - Why Streamlit?
- **System Architecture**
  - 16-node structure
  - State flow diagrams
  - Database design (with news_outcomes tables)
- **Key Algorithms** (with code snippets)
  - **News source reliability learning** 🌟
  - **Adaptive weighting with learning feedback**
  - Pump-and-dump detection
  - News filtering
- **Dashboard Design**
- **Challenges & Solutions**
  - Building historical dataset from scratch
  - Handling insufficient data for new stocks
  - Balancing learning speed vs accuracy

### 5. Results & Evaluation (10-15 pages)
- **Case Studies** (10-15 stocks)
  - AAPL, TSLA, NVDA, GME, SPY, etc.
  - For each: Source reliability scores, learning improvements
- **News Learning System Effectiveness** 🌟 PRIMARY RESULTS
  - Sentiment accuracy without learning: 60-65%
  - Sentiment accuracy with learning: 70-75%
  - Improvement: +10-15%
  - Source reliability distribution (Bloomberg vs blogs)
  - News type effectiveness comparison
  - Learning curve over time (how quickly system improves)
- **Accuracy Comparisons**
  - Technical vs Sentiment vs Combined
  - Adaptive vs Equal weighting
  - With vs Without Node 8 learning
- **Anomaly Detection Effectiveness**
  - Detection rate
  - False positive rate
  - Case studies (GME pump-and-dump)
- **Performance Metrics**
  - Processing speed
  - Dashboard responsiveness
- **User Testing Feedback**
  - Understanding of source reliability scores
  - Trust in system recommendations

### 6. Discussion (8-10 pages)
- Interpretation of findings
- Comparison with existing tools
- Innovation analysis (LangGraph approach)
- Limitations
  - API rate limits
  - Model assumptions
  - Market efficiency theory
- Future improvements

### 7. Conclusion (3-5 pages)
- Summary of achievements
- Contribution to field
- Recommendations for future work

---

## 💪 Key Differentiators vs Traditional Approaches

| Aspect | Traditional Tools | Your LangGraph System |
|--------|------------------|----------------------|
| **Architecture** | Sequential scripts | Multi-agent orchestration |
| **Processing** | Linear/slow | Parallel/fast |
| **Explainability** | None | LLM-powered explanations |
| **Risk Detection** | Single-phase | Two-phase (early + late) |
| **Adaptability** | Fixed weights | Learning adaptive weights |
| **Modularity** | Monolithic | 16 independent nodes |
| **Extensibility** | Hard to modify | Add nodes easily |
| **Beginner-Friendly** | Technical jargon | Plain English |
| **Fraud Protection** | Basic alerts | Sophisticated detection |

---

## ⚠️ Realistic Expectations

### What This System WILL Do:
✅ Aggregate multi-source financial data efficiently  
✅ Provide intelligent, explainable analysis  
✅ Detect pump-and-dump schemes effectively  
✅ Learn optimal weights per stock  
✅ Reduce analysis time from 30 minutes to 2 minutes  
✅ Make sophisticated analysis accessible to beginners  

### What This System WON'T Do:
❌ Guarantee profits or beat the market consistently  
❌ Replace professional financial advisors  
❌ Work without internet/API access  
❌ Predict black swan events  
❌ Achieve 90%+ accuracy (60-70% is excellent)  

### Academic Focus:
- **Demonstrate methodology improvement** over baseline
- **Show adaptive learning capability** through weights
- **Prove two-phase detection effectiveness** with data
- **Novel architecture application** (LangGraph in finance)
- **NOT about absolute prediction accuracy**

---

## 🎯 Your Thesis Defense Story

> "Traditional stock analysis tools present three major problems: they process 
> information sequentially (slow), produce unexplained technical outputs 
> (inaccessible to beginners), and offer no protection against market manipulation.
>
> My thesis presents a novel solution using LangGraph multi-agent architecture 
> that addresses all three challenges:
>
> **1. Parallel Processing:** 16 independent agents analyze data simultaneously, 
> reducing analysis time from 30 minutes to under 5 seconds.
>
> **2. Explainable AI:** Large language models translate complex financial signals 
> into plain English, making sophisticated analysis accessible to everyday investors.
>
> **3. Two-Phase Anomaly Detection:** Content filtering protects the learning system 
> from fake news, while behavioral analysis detects sophisticated manipulation like 
> pump-and-dump schemes with 95% accuracy.
>
> **4. Adaptive Intelligence:** The system learns optimal signal weights per stock 
> through historical backtesting, achieving 10-15% higher accuracy than equal weighting.
>
> This represents the first known application of LangGraph to financial analysis, 
> demonstrating that multi-agent orchestration can democratize sophisticated 
> investment intelligence while protecting users from manipulation."

---

## 📊 Success Metrics

### Development Milestones
- [ ] Week 3: LangGraph environment + data nodes working
- [ ] Week 5: Sentiment analysis + early anomaly detection integrated
- [ ] Week 7: Adaptive weighting functional
- [ ] Week 10: Complete polished dashboard with all 6 tabs
- [ ] Week 12: Documentation complete + user testing
- [ ] Week 14: Thesis written + defense ready

### Quality Metrics
- [ ] Combined signal accuracy: 60%+
- [ ] Pump-and-dump detection: 95%+
- [ ] False positive rate: <3%
- [ ] Signal generation time: <5 seconds
- [ ] Dashboard load time: <5 seconds
- [ ] User satisfaction: 4/5+ from testers

### Academic Metrics
- [ ] Thesis length: 50-80 pages
- [ ] Stocks analyzed: 10-15
- [ ] Data points: 10,000+ prices, 1,000+ news articles
- [ ] Statistical significance: p < 0.05
- [ ] Working demo for defense

---

## 🚀 Why This Thesis Matters

### Academic Contribution:
- Novel application of LangGraph to financial domain
- First two-phase anomaly detection system for stock analysis
- Demonstrates multi-agent orchestration benefits in finance
- Provides empirical data on adaptive vs fixed weighting

### Practical Impact:
- Makes sophisticated analysis accessible to beginners
- Protects retail investors from manipulation
- Reduces analysis time by 90%
- Open-source architecture for future research

### Industry Relevance:
- Aligns with trend toward AI-powered financial tools
- Addresses real problem (information overload)
- Shows path for responsible AI in finance
- Demonstrates explainable AI principles

---

**You're not just building a stock analysis tool.**  
**You're pioneering multi-agent financial intelligence systems.** 🚀

---

*This overview synthesizes the core thesis concept from FINAL_THESIS_BLUEPRINT.md with the LangGraph architecture innovations from LANGGRAPH_THESIS_ARCHITECTURE.md and UPDATED_ARCHITECTURE_SPLIT_ANOMALY.md*
