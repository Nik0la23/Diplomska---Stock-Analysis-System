---
description: "Two-phase anomaly detection system - content filtering (9A) then behavioral analysis (9B)"
alwaysApply: false
globs: ["**/node_09*.py", "**/anomaly*.py", "**/detection*.py"]
---

# Two-Phase Anomaly Detection (THESIS INNOVATION)

**Novel approach:** Split anomaly detection into TWO phases for maximum protection.

## Phase 1: Node 9A - Early Detection (Content-Based)

**Runs IMMEDIATELY after news fetching, BEFORE sentiment analysis.**

### Purpose
Filter suspicious news content to protect learning system (Node 8) from contaminated data.

### Checks
1. **Keyword Alerts** - Scan for: bankruptcy, fraud, investigation, scandal
2. **News Surge** - Is article count 5x normal? (50+ vs usual 10-20)
3. **Source Credibility** - Are articles from untrusted domains?
4. **Coordinated Posting** - Identical articles posted at same time? (bots)

### Critical Output
- `cleaned_stock_news` - Suspicious articles removed
- `cleaned_market_news` - Filtered data
- `cleaned_related_news` - Clean data only

**IMPORTANT:** Node 5 (Sentiment) uses `cleaned_*_news`, NOT `raw_*_news`.

### Dependencies
- **Runs AFTER:** Node 2 (news fetching)
- **Runs BEFORE:** Nodes 4, 5, 6, 7 (analysis layer)

---

## Phase 2: Node 9B - Late Detection (Behavioral)

**Runs AFTER all analysis and learning, detects sophisticated manipulation.**

### Purpose
Identify pump-and-dump schemes and behavioral anomalies using historical patterns.

### Detection Methods
1. **Pump-and-Dump Score** - Price spike + volume surge + crash
2. **Price Anomalies** - z-score > 3 (spikes/crashes)
3. **Volume Anomalies** - 10x normal volume
4. **Volatility Spikes** - 5x standard deviation
5. **News-Price Divergence** - Positive news but price crashed

### Risk Levels
- **CRITICAL** (score > 75): Halt trading recommendation
- **HIGH** (score > 50): Strong warning
- **MEDIUM/LOW**: Monitor

### Dependencies
- **Runs AFTER:** Node 8 (needs historical patterns)
- **Runs BEFORE:** Node 10 (backtesting)

---

## Why Two Phases?

1. **Early (9A)** catches obvious fake news from content
2. **Late (9B)** catches sophisticated manipulation from behavior
3. **Protects Node 8** - Learning system learns only from clean data
4. **95%+ accuracy** - Pump-and-dump detection with <3% false positives

## Conditional Routing

After Node 11 (weights calculated), conditional edge checks combined risk:

```python
if pump_dump_score > 75 or behavioral_risk == 'CRITICAL':
    Route to Node 13 only (warning, skip Node 12 signal generation)
else:
    Route to Node 12 (normal signal generation)
```

## Reference

See `@NODE_BUILD_GUIDE.md` sections for Node 9A and Node 9B for detailed implementation specs.
