"""
Comprehensive workflow analysis showing Node 9A labeling and parallel node reasoning.
"""
import logging
import sys
from datetime import datetime

# Suppress most logs except our analysis
logging.basicConfig(level=logging.ERROR)

from src.graph.workflow import create_stock_analysis_workflow
from src.graph.state import create_initial_state

print('='*100)
print('COMPREHENSIVE WORKFLOW ANALYSIS: NODE 9A + PARALLEL NODES (4, 5, 6, 7)')
print('='*100)

# Run workflow
workflow = create_stock_analysis_workflow()
initial_state = create_initial_state('AAPL')
final_state = workflow.invoke(initial_state)

# ============================================================================
# PART 1: NODE 9A - CONTENT ANALYSIS
# ============================================================================
print('\n' + '='*100)
print('PART 1: NODE 9A - CONTENT ANALYSIS & LABELING')
print('='*100)

cleaned_stock_news = final_state.get('cleaned_stock_news', [])
print(f'\nTotal articles labeled by Node 9A: {len(cleaned_stock_news)}')

# Show detailed sample
if cleaned_stock_news:
    print('\n' + '-'*100)
    print('SAMPLE ARTICLE WITH FULL NODE 9A LABELING:')
    print('-'*100)
    sample = cleaned_stock_news[0]
    
    print(f"\nORIGINAL ARTICLE DATA:")
    print(f"  Title: {sample.get('headline', sample.get('title', 'N/A'))}")
    print(f"  Summary: {sample.get('summary', 'N/A')[:200]}...")
    print(f"  Source: {sample.get('source', 'N/A')}")
    print(f"  URL: {sample.get('url', 'N/A')[:80]}...")
    
    print(f"\nNODE 9A ADDED SCORES (0-1 scale):")
    print(f"  📊 Sensationalism: {sample.get('sensationalism_score', 0):.3f}")
    print(f"     → Checks for: ALL CAPS, !!!, clickbait keywords")
    print(f"  ⏰ Urgency: {sample.get('urgency_score', 0):.3f}")
    print(f"     → Checks for: BREAKING, URGENT, ACT NOW, time pressure")
    print(f"  ❓ Unverified Claims: {sample.get('unverified_claims_score', 0):.3f}")
    print(f"     → Checks for: allegedly, rumored, sources say, speculation")
    print(f"  ✅ Source Credibility: {sample.get('source_credibility_score', 0):.3f}")
    print(f"     → Based on domain authority (Bloomberg 0.95, Yahoo 0.6, Unknown 0.3)")
    print(f"  🎓 Complexity: {sample.get('complexity_score', 0):.3f}")
    print(f"     → Density of technical financial terms")
    print(f"  ⚠️  COMPOSITE ANOMALY: {sample.get('composite_anomaly_score', 0):.3f}")
    print(f"     → Weighted: 25% sens + 20% urgency + 25% unverified + 30% (1-credibility)")
    
    tags = sample.get('content_tags', {})
    print(f"\nCONTENT TAGS EXTRACTED:")
    print(f"  Topic: {tags.get('topic', 'general')}")
    print(f"  Keywords: {', '.join(tags.get('keywords', [])[:10]) if tags.get('keywords') else 'None'}")
    print(f"  Temporal: {tags.get('temporal', 'current')}")
    print(f"  Entities: {', '.join(tags.get('entities', []))}")
    
    print(f"\n💡 INTERPRETATION:")
    comp_score = sample.get('composite_anomaly_score', 0)
    if comp_score < 0.3:
        print(f"   ✓ CLEAN ARTICLE (< 0.3): Trustworthy source, factual reporting")
    elif comp_score < 0.5:
        print(f"   ⚠️  MODERATE RISK (0.3-0.5): Some concerns, review manually")
    elif comp_score < 0.7:
        print(f"   ⚠️  HIGH RISK (0.5-0.7): Multiple red flags detected")
    else:
        print(f"   🚨 VERY HIGH RISK (> 0.7): Likely misinformation or low-quality source")

# Show distribution
content_summary = final_state.get('content_analysis_summary', {})
dist = content_summary.get('source_credibility_distribution', {})
print(f"\n📈 SOURCE CREDIBILITY DISTRIBUTION:")
print(f"  High (≥ 0.8): {dist.get('high', 0)} articles")
print(f"  Medium (0.5-0.8): {dist.get('medium', 0)} articles")
print(f"  Low (< 0.5): {dist.get('low', 0)} articles")

# ============================================================================
# PART 2: NODE 4 - TECHNICAL ANALYSIS
# ============================================================================
print('\n\n' + '='*100)
print('PART 2: NODE 4 - TECHNICAL ANALYSIS')
print('='*100)

tech_summary = final_state.get('technical_analysis_summary', {})
if tech_summary:
    print(f"\nINPUT DATA:")
    print(f"  Historical price data: {len(final_state.get('price_data', []))} days")
    print(f"  Ticker: AAPL")
    
    print(f"\nTECHNICAL INDICATORS CALCULATED:")
    indicators = tech_summary.get('indicators', {})
    if indicators:
        print(f"  RSI (14-day): {indicators.get('rsi', 'N/A'):.2f}" if indicators.get('rsi') else "  RSI: N/A")
        print(f"    → < 30 = oversold (bullish), > 70 = overbought (bearish)")
        print(f"  MACD: {indicators.get('macd', 'N/A'):.3f}" if indicators.get('macd') else "  MACD: N/A")
        print(f"    → Positive = bullish momentum, Negative = bearish momentum")
        print(f"  MACD Signal: {indicators.get('macd_signal', 'N/A'):.3f}" if indicators.get('macd_signal') else "  MACD Signal: N/A")
        print(f"  MACD Histogram: {indicators.get('macd_hist', 'N/A'):.3f}" if indicators.get('macd_hist') else "  MACD Hist: N/A")
        print(f"  Bollinger Bands:")
        print(f"    Upper: ${indicators.get('bb_upper', 'N/A'):.2f}" if indicators.get('bb_upper') else "    Upper: N/A")
        print(f"    Middle: ${indicators.get('bb_middle', 'N/A'):.2f}" if indicators.get('bb_middle') else "    Middle: N/A")
        print(f"    Lower: ${indicators.get('bb_lower', 'N/A'):.2f}" if indicators.get('bb_lower') else "    Lower: N/A")
        print(f"    → Price near upper = overbought, near lower = oversold")
    
    signals = tech_summary.get('signals', {})
    print(f"\nSIGNALS GENERATED:")
    print(f"  RSI Signal: {signals.get('rsi_signal', 'N/A')}")
    print(f"  MACD Signal: {signals.get('macd_signal', 'N/A')}")
    print(f"  Bollinger Signal: {signals.get('bb_signal', 'N/A')}")
    print(f"  OVERALL: {signals.get('overall', 'N/A')}")
    
    print(f"\n🧮 HOW NODE 4 REACHES ITS CONCLUSION:")
    print(f"  1. Loads historical price data (OHLCV)")
    print(f"  2. Calculates RSI using 14-day window")
    print(f"  3. Calculates MACD (12, 26, 9) and signal line")
    print(f"  4. Calculates Bollinger Bands (20-day, 2 std dev)")
    print(f"  5. Generates individual signals from each indicator")
    print(f"  6. Combines signals: if 2+ indicators agree → strong signal, else → HOLD")
    print(f"  7. Confidence based on indicator agreement and strength")

# ============================================================================
# PART 3: NODE 5 - SENTIMENT ANALYSIS
# ============================================================================
print('\n\n' + '='*100)
print('PART 3: NODE 5 - SENTIMENT ANALYSIS')
print('='*100)

sent_summary = final_state.get('sentiment_analysis_summary', {})
if sent_summary:
    print(f"\nINPUT DATA (from Node 9A cleaned articles):")
    print(f"  Stock news: {sent_summary.get('total_articles', {}).get('stock', 0)} articles")
    print(f"  Market news: {sent_summary.get('total_articles', {}).get('market', 0)} articles")
    print(f"  Related news: {sent_summary.get('total_articles', {}).get('related', 0)} articles")
    
    print(f"\nSENTIMENT SCORES BY NEWS TYPE (-1 to +1):")
    stock_sent = sent_summary.get('stock_sentiment', {})
    market_sent = sent_summary.get('market_sentiment', {})
    related_sent = sent_summary.get('related_sentiment', {})
    
    print(f"  Stock News:")
    print(f"    Average: {stock_sent.get('average', 0):.3f}")
    print(f"    Positive: {stock_sent.get('positive_count', 0)}, Negative: {stock_sent.get('negative_count', 0)}, Neutral: {stock_sent.get('neutral_count', 0)}")
    print(f"    Confidence: {stock_sent.get('confidence', 0):.2f}")
    
    print(f"  Market News:")
    print(f"    Average: {market_sent.get('average', 0):.3f}")
    print(f"    Positive: {market_sent.get('positive_count', 0)}, Negative: {market_sent.get('negative_count', 0)}, Neutral: {market_sent.get('neutral_count', 0)}")
    
    print(f"  Related News:")
    print(f"    Average: {related_sent.get('average', 0):.3f}")
    print(f"    Confidence: {related_sent.get('confidence', 0):.2f}")
    
    print(f"\nCOMBINED SENTIMENT:")
    print(f"  Score: {sent_summary.get('combined_sentiment', 0):.3f}")
    print(f"  Signal: {sent_summary.get('signal', 'N/A')}")
    print(f"  Confidence: {sent_summary.get('confidence', 0):.2f}")
    
    print(f"\n🧮 HOW NODE 5 REACHES ITS CONCLUSION:")
    print(f"  1. Takes cleaned articles from Node 9A")
    print(f"  2. For each article:")
    print(f"     - If Alpha Vantage sentiment exists → use it")
    print(f"     - Else → analyze with FinBERT (ProsusAI/finbert model)")
    print(f"  3. Aggregate by news type:")
    print(f"     - Calculate average sentiment")
    print(f"     - Count positive/negative/neutral")
    print(f"     - Calculate confidence (based on article count and score distribution)")
    print(f"  4. Combine with weights: 50% stock + 25% market + 25% related")
    print(f"  5. Generate signal:")
    print(f"     - > +0.2 and confidence > 0.5 → BUY")
    print(f"     - < -0.2 and confidence > 0.5 → SELL")
    print(f"     - Else → HOLD")

# ============================================================================
# PART 4: NODE 6 - MARKET CONTEXT
# ============================================================================
print('\n\n' + '='*100)
print('PART 4: NODE 6 - MARKET CONTEXT ANALYSIS')
print('='*100)

market_context = final_state.get('market_context_summary', {})
if market_context:
    print(f"\nINPUT DATA:")
    print(f"  Ticker: AAPL")
    print(f"  Historical price data")
    print(f"  Market index: SPY (S&P 500)")
    print(f"  Related companies from Node 3")
    
    print(f"\nSECTOR ANALYSIS:")
    sector_info = market_context.get('sector', {})
    print(f"  Stock Sector: {sector_info.get('sector_name', 'N/A')}")
    print(f"  Industry: {sector_info.get('industry_name', 'N/A')}")
    print(f"  Sector ETF: {sector_info.get('sector_etf', 'N/A')}")
    perf = sector_info.get('sector_performance', 0)
    print(f"  Sector Performance: {perf:.2f}%" if perf else "  Sector Performance: N/A")
    print(f"  Signal: {sector_info.get('sector_signal', 'N/A')}")
    
    print(f"\nMARKET TREND ANALYSIS:")
    market_trend = market_context.get('market_trend', {})
    mperf = market_trend.get('market_performance', 0)
    print(f"  SPY Performance: {mperf:.2f}%" if mperf else "  SPY Performance: N/A")
    vol = market_trend.get('volatility', 0)
    print(f"  Volatility: {vol:.2f}%" if vol else "  Volatility: N/A")
    print(f"  Trend: {market_trend.get('trend', 'N/A')}")
    print(f"  Signal: {market_trend.get('signal', 'N/A')}")
    
    print(f"\nRELATED COMPANIES ANALYSIS:")
    related = market_context.get('related_companies', {})
    rperf = related.get('average_performance', 0)
    print(f"  Average Performance: {rperf:.2f}%" if rperf else "  Average Performance: N/A")
    print(f"  Signal: {related.get('signal', 'N/A')}")
    
    print(f"\nCORRELATION WITH MARKET (SPY):")
    correlation = market_context.get('correlation', {})
    corr = correlation.get('correlation_coefficient', 0)
    print(f"  Pearson Correlation: {corr:.3f}" if corr else "  Correlation: N/A")
    print(f"    → 1.0 = perfect positive, 0 = no correlation, -1.0 = perfect negative")
    beta = correlation.get('beta', 0)
    print(f"  Beta: {beta:.3f}" if beta else "  Beta: N/A")
    print(f"    → 1.0 = moves with market, > 1 = more volatile, < 1 = less volatile")
    
    print(f"\nCONTEXT SIGNAL:")
    print(f"  Overall: {market_context.get('context_signal', 'N/A')}")
    print(f"  Confidence: {market_context.get('confidence', 0):.2f}")
    
    print(f"\n🧮 HOW NODE 6 REACHES ITS CONCLUSION:")
    print(f"  1. Detect stock's sector using yfinance")
    print(f"  2. Fetch sector ETF performance (e.g., XLK for Technology)")
    print(f"  3. Analyze overall market trend:")
    print(f"     - Fetch SPY (S&P 500) data")
    print(f"     - Calculate 1-day, 5-day, 20-day returns")
    print(f"     - Calculate volatility")
    print(f"  4. Analyze related companies from Node 3:")
    print(f"     - Fetch 1-day performance")
    print(f"     - Calculate average")
    print(f"  5. Calculate correlation and beta with SPY (30-day window)")
    print(f"  6. Generate signals:")
    print(f"     - Sector: > 1% positive, < -1% negative")
    print(f"     - Market: uptrend/downtrend/sideways based on returns")
    print(f"     - Related: > 0.5% positive, < -0.5% negative")
    print(f"  7. Combine factors with weights:")
    print(f"     - Sector: 25%, Market: 30%, Related: 20%, Correlation: 25%")
    print(f"  8. Final signal: > 0 → POSITIVE, < 0 → NEGATIVE, else → NEUTRAL")

# ============================================================================
# PART 5: NODE 7 - FUNDAMENTAL ANALYSIS
# ============================================================================
print('\n\n' + '='*100)
print('PART 5: NODE 7 - FUNDAMENTAL ANALYSIS')
print('='*100)

fund_summary = final_state.get('fundamental_analysis_summary', {})
if fund_summary:
    print(f"\nINPUT DATA:")
    print(f"  Ticker: AAPL")
    print(f"  Source: Finnhub API")
    
    print(f"\nVALUATION METRICS:")
    metrics = fund_summary.get('metrics', {})
    valuation = metrics.get('valuation', {})
    print(f"  P/E Ratio: {valuation.get('pe_ratio', 'N/A')}")
    print(f"    → Price-to-Earnings: lower = cheaper, higher = expensive/growth")
    print(f"  P/B Ratio: {valuation.get('pb_ratio', 'N/A')}")
    print(f"    → Price-to-Book: < 1 = undervalued, > 3 = overvalued")
    mcap = valuation.get('market_cap', 0)
    print(f"  Market Cap: ${mcap/1e9:.2f}B" if mcap else "  Market Cap: N/A")
    
    print(f"\nFINANCIAL HEALTH:")
    health = metrics.get('financial_health', {})
    print(f"  Current Ratio: {health.get('current_ratio', 'N/A')}")
    print(f"    → > 1 = can pay short-term debt, < 1 = liquidity concerns")
    print(f"  Debt-to-Equity: {health.get('debt_to_equity', 'N/A')}")
    print(f"    → < 1 = less debt than equity (good), > 2 = high leverage (risky)")
    print(f"  ROE: {health.get('roe', 'N/A')}")
    print(f"    → Return on Equity: > 15% = strong, < 10% = weak")
    
    print(f"\nGROWTH METRICS:")
    growth = metrics.get('growth', {})
    print(f"  Revenue Growth: {growth.get('revenue_growth', 'N/A')}")
    print(f"  Earnings Growth: {growth.get('earnings_growth', 'N/A')}")
    
    print(f"\nFUNDAMENTAL SIGNAL:")
    print(f"  Overall: {fund_summary.get('signal', 'N/A')}")
    print(f"  Confidence: {fund_summary.get('confidence', 0):.2f}")
    
    print(f"\n🧮 HOW NODE 7 REACHES ITS CONCLUSION:")
    print(f"  1. Fetch financial metrics from Finnhub")
    print(f"  2. Calculate valuation score:")
    print(f"     - P/E < 15 = undervalued, 15-25 = fair, > 25 = overvalued")
    print(f"     - P/B < 1 = undervalued, 1-3 = fair, > 3 = overvalued")
    print(f"  3. Calculate health score:")
    print(f"     - Current ratio > 1.5 = good, 1-1.5 = fair, < 1 = poor")
    print(f"     - Debt-to-Equity < 1 = good, 1-2 = fair, > 2 = poor")
    print(f"     - ROE > 15% = good, 10-15% = fair, < 10% = poor")
    print(f"  4. Calculate growth score from revenue/earnings growth")
    print(f"  5. Combine scores with weights: 40% valuation + 30% health + 30% growth")
    print(f"  6. Generate signal: > 0.6 → BUY, < 0.4 → SELL, else → HOLD")

print('\n' + '='*100)
print('ANALYSIS COMPLETE')
print('='*100)
