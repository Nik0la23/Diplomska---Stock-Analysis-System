"""
Test Hybrid API Implementation
================================
Tests the optimized hybrid approach:
- Node 1: yfinance (primary) + Polygon (backup)
- Node 2: Alpha Vantage (primary) + Finnhub (supplement)
- Node 3: Finnhub Peers (only source)
"""

import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.graph.state import create_initial_state
from src.langgraph_nodes.node_01_data_fetching import fetch_price_data_node
from src.langgraph_nodes.node_03_related_companies import detect_related_companies_node
from src.langgraph_nodes.node_02_news_fetching import fetch_all_news_node
from src.utils.logger import setup_application_logging

# Setup logging
setup_application_logging()

print("=" * 100)
print(" " * 25 + "HYBRID API IMPLEMENTATION TEST")
print("=" * 100)
print()
print("Testing optimized free-tier API setup:")
print("  📊 Node 1: yfinance (primary) → Polygon.io (backup)")
print("  📰 Node 2: Alpha Vantage (primary) → Finnhub (supplement)")
print("  🔗 Node 3: Finnhub Peers API")
print()
print("=" * 100)

# Test ticker
ticker = 'NVDA'
print(f"\n🎯 Test Subject: {ticker}")
print(f"📅 Target: 180 days price data, 10+ days news data\n")

# ============================================================================
# TEST 1: NODE 1 - PRICE DATA (yfinance primary)
# ============================================================================

print("=" * 100)
print("TEST 1: NODE 1 - PRICE DATA FETCHING")
print("=" * 100)

state = create_initial_state(ticker)

print(f"\n[1/3] Running Node 1: Price Data (yfinance primary)...")
start_time = datetime.now()
state = fetch_price_data_node(state)
elapsed = (datetime.now() - start_time).total_seconds()

if state.get('raw_price_data') is not None:
    df = state['raw_price_data']
    print(f"\n✅ SUCCESS - Node 1 completed in {elapsed:.2f}s")
    print(f"   📊 Price data: {len(df)} trading days")
    print(f"   📅 Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   💰 Latest close: ${df.iloc[-1]['close']:.2f}")
    
    # Check data quality
    span_days = (df['date'].max() - df['date'].min()).days
    print(f"   📏 Span: {span_days} calendar days")
    
    if span_days >= 180:
        print(f"   ✅ Meets 180-day requirement")
    else:
        print(f"   ⚠️  Only {span_days} days (target: 180)")
else:
    print(f"\n❌ FAILED - No price data")
    exit(1)

# ============================================================================
# TEST 2: NODE 3 - RELATED COMPANIES (Finnhub Peers)
# ============================================================================

print("\n" + "=" * 100)
print("TEST 2: NODE 3 - RELATED COMPANIES DETECTION")
print("=" * 100)

print(f"\n[2/3] Running Node 3: Related Companies (Finnhub Peers)...")
start_time = datetime.now()
state = detect_related_companies_node(state)
elapsed = (datetime.now() - start_time).total_seconds()

if state.get('related_companies'):
    print(f"\n✅ SUCCESS - Node 3 completed in {elapsed:.2f}s")
    print(f"   🔗 Related companies: {len(state['related_companies'])}")
    print(f"   📋 Peers: {', '.join(state['related_companies'])}")
else:
    print(f"\n⚠️  WARNING - No related companies found")

# ============================================================================
# TEST 3: NODE 2 - NEWS DATA (Alpha Vantage + Finnhub)
# ============================================================================

print("\n" + "=" * 100)
print("TEST 3: NODE 2 - NEWS DATA FETCHING (HYBRID)")
print("=" * 100)

print(f"\n[3/3] Running Node 2: News Fetching (Alpha Vantage primary + Finnhub supplement)...")
start_time = datetime.now()
state = fetch_all_news_node(state)
elapsed = (datetime.now() - start_time).total_seconds()

stock_news = state.get('stock_news', [])
market_news = state.get('market_news', [])
total_news = len(stock_news) + len(market_news)

print(f"\n✅ Node 2 completed in {elapsed:.2f}s")
print(f"   📰 Total news articles: {total_news}")
print(f"   📊 Stock news (Alpha Vantage): {len(stock_news)}")
print(f"   📈 Market news (Finnhub): {len(market_news)}")

# Analyze news coverage
if stock_news:
    dates = [datetime.fromtimestamp(a['datetime']) for a in stock_news if 'datetime' in a]
    if dates:
        oldest = min(dates)
        newest = max(dates)
        span = (newest - oldest).days
        
        print(f"\n📅 STOCK NEWS COVERAGE (Alpha Vantage):")
        print(f"   Oldest: {oldest.strftime('%Y-%m-%d %H:%M')}")
        print(f"   Newest: {newest.strftime('%Y-%m-%d %H:%M')}")
        print(f"   Span: {span} days")
        
        if span >= 10:
            print(f"   ✅ Good coverage ({span} days)")
        else:
            print(f"   ⚠️  Limited coverage ({span} days)")
        
        # Check for sentiment data
        with_sentiment = sum(1 for a in stock_news if 'overall_sentiment_score' in a)
        print(f"\n💡 SENTIMENT ANALYSIS:")
        print(f"   Articles with sentiment: {with_sentiment}/{len(stock_news)} ({with_sentiment/len(stock_news)*100:.1f}%)")
        
        if with_sentiment > 0:
            print(f"   ✅ Alpha Vantage provides built-in sentiment!")
            
            # Sample sentiment
            sample = next((a for a in stock_news if 'overall_sentiment_score' in a), None)
            if sample:
                print(f"\n   📰 Sample Article:")
                print(f"      Title: {sample.get('headline', '')[:60]}...")
                print(f"      Sentiment: {sample.get('overall_sentiment_label', 'N/A')} ({sample.get('overall_sentiment_score', 0.0):.3f})")
                ticker_sent = sample.get('ticker_sentiment_score')
                if ticker_sent is not None and isinstance(ticker_sent, (int, float)):
                    print(f"      {ticker}-specific sentiment: {ticker_sent:.3f}")

if market_news:
    dates = [datetime.fromtimestamp(a['datetime']) for a in market_news if 'datetime' in a]
    if dates:
        oldest = min(dates)
        newest = max(dates)
        span = (newest - oldest).days
        
        print(f"\n📅 MARKET NEWS COVERAGE (Finnhub):")
        print(f"   Oldest: {oldest.strftime('%Y-%m-%d %H:%M')}")
        print(f"   Newest: {newest.strftime('%Y-%m-%d %H:%M')}")
        print(f"   Span: {span} days")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 100)
print(" " * 30 + "FINAL SUMMARY")
print("=" * 100)

# Calculate total execution time
total_time = sum(state['node_execution_times'].values())

print(f"\n📊 EXECUTION PERFORMANCE:")
print(f"   Node 1 (Price):  {state['node_execution_times'].get('node_1', 0):.2f}s")
print(f"   Node 3 (Peers):  {state['node_execution_times'].get('node_3', 0):.2f}s")
print(f"   Node 2 (News):   {state['node_execution_times'].get('node_2', 0):.2f}s")
print(f"   ────────────────────────")
print(f"   Total: {total_time:.2f}s")

print(f"\n📈 DATA COVERAGE:")
print(f"   Price data: {len(state.get('raw_price_data', []))} trading days")
print(f"   News articles: {total_news} total")
print(f"   Related companies: {len(state.get('related_companies', []))}")

print(f"\n✅ ADVANTAGES OF HYBRID APPROACH:")
print(f"   🔑 Fewer API keys needed (yfinance needs NO key)")
print(f"   ⚡ Fast performance ({total_time:.1f}s total)")
print(f"   💡 Built-in sentiment from Alpha Vantage")
print(f"   📊 Best free-tier coverage available")
print(f"   💰 100% FREE (within rate limits)")

if total_news >= 50 and len(state.get('raw_price_data', [])) >= 100:
    print(f"\n🎯 VERDICT: ✅ IMPLEMENTATION SUCCESSFUL")
    print(f"   System ready for thesis development!")
else:
    print(f"\n⚠️  VERDICT: NEEDS ATTENTION")
    print(f"   Check API keys and rate limits")

print(f"\n" + "=" * 100)
print(f"✅ Test complete!\n")
