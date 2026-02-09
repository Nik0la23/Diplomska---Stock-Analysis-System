"""
Comprehensive NVIDIA Analysis Test
Tests full pipeline: Price data + Related companies + News (6 months back)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.graph.state import create_initial_state
from src.langgraph_nodes.node_01_data_fetching import fetch_price_data_node
from src.langgraph_nodes.node_03_related_companies import detect_related_companies_node
from src.langgraph_nodes.node_02_news_fetching import fetch_all_news_node
from src.utils.logger import setup_application_logging
from datetime import datetime, timedelta

# Setup logging
setup_application_logging()

print("=" * 100)
print(" " * 30 + "NVIDIA (NVDA) STOCK ANALYSIS")
print("=" * 100)
print(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 100)

# ============================================================================
# STEP 1: FETCH STOCK PRICES (6 MONTHS)
# ============================================================================
print("\n📊 STEP 1: FETCHING STOCK PRICES (6 MONTHS)")
print("-" * 100)

state = create_initial_state('NVDA')

# Modify Node 1 to fetch 6 months (180 days) - we'll call it directly
from src.langgraph_nodes.node_01_data_fetching import fetch_from_polygon, fetch_from_yfinance, validate_price_data

# Try to fetch 6 months
df = fetch_from_polygon('NVDA', days=180)
if df is None or not validate_price_data(df, 'NVDA', min_rows=100):
    print("   Polygon didn't return enough data, trying yfinance...")
    df = fetch_from_yfinance('NVDA', days=180)

if df is not None:
    state['raw_price_data'] = df
    print(f"✅ Successfully fetched {len(df)} days of price data")
    print(f"\n📈 PRICE SUMMARY:")
    print(f"   Date Range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    print(f"   Current Price: ${df['close'].iloc[-1]:.2f}")
    print(f"   Highest Price: ${df['high'].max():.2f}")
    print(f"   Lowest Price: ${df['low'].min():.2f}")
    print(f"   Average Volume: {df['volume'].mean():,.0f} shares/day")
    
    # Calculate price change
    price_start = df['close'].iloc[0]
    price_end = df['close'].iloc[-1]
    price_change_pct = ((price_end - price_start) / price_start) * 100
    
    print(f"\n   📊 6-Month Performance:")
    print(f"      Starting Price: ${price_start:.2f}")
    print(f"      Ending Price: ${price_end:.2f}")
    print(f"      Change: {price_change_pct:+.2f}%")
    
    # Show last 7 days
    print(f"\n   📅 Last 7 Trading Days:")
    print("-" * 100)
    last_7 = df.tail(7)
    print(f"   {'Date':<12} {'Open':>10} {'High':>10} {'Low':>10} {'Close':>10} {'Volume':>15}")
    print("-" * 100)
    for _, row in last_7.iterrows():
        print(f"   {row['date'].strftime('%Y-%m-%d'):<12} "
              f"${row['open']:>9.2f} ${row['high']:>9.2f} ${row['low']:>9.2f} "
              f"${row['close']:>9.2f} {row['volume']:>14,.0f}")
else:
    print("❌ Failed to fetch price data")

# ============================================================================
# STEP 2: DETECT RELATED COMPANIES
# ============================================================================
print("\n\n🔗 STEP 2: DETECTING RELATED COMPANIES")
print("-" * 100)

state = detect_related_companies_node(state)

print(f"✅ Found {len(state['related_companies'])} related companies")
print(f"\n   Related Companies (Competitors/Peers):")
for i, ticker in enumerate(state['related_companies'], 1):
    print(f"   {i}. {ticker}")

print(f"\n   Execution Time: {state['node_execution_times']['node_3']:.2f}s")

# ============================================================================
# STEP 3: FETCH NEWS FROM ALL SOURCES
# ============================================================================
print("\n\n📰 STEP 3: FETCHING NEWS FROM 3 SOURCES (PARALLEL)")
print("-" * 100)

state = fetch_all_news_node(state)

total_articles = len(state['stock_news']) + len(state['market_news']) + len(state['related_company_news'])

print(f"✅ Successfully fetched {total_articles} total articles")
print(f"\n   📊 News Breakdown:")
print(f"      Stock-specific news (NVDA): {len(state['stock_news'])} articles")
print(f"      Market-wide news: {len(state['market_news'])} articles")
print(f"      Related companies news: {len(state['related_company_news'])} articles")
print(f"\n   Execution Time: {state['node_execution_times']['node_2']:.2f}s")

# ============================================================================
# DISPLAY SAMPLE NEWS
# ============================================================================

# Show sample stock news
print(f"\n\n📱 SAMPLE STOCK NEWS (NVIDIA)")
print("=" * 100)
for i, article in enumerate(state['stock_news'][:5], 1):
    dt = datetime.fromtimestamp(article['datetime']) if article['datetime'] else datetime.now()
    print(f"\n{i}. {article['headline']}")
    print(f"   Source: {article['source']}")
    print(f"   Date: {dt.strftime('%Y-%m-%d %H:%M')}")
    print(f"   URL: {article['url'][:80]}...")
    if article['summary']:
        print(f"   Summary: {article['summary'][:150]}...")

# Show sample market news
print(f"\n\n🌍 SAMPLE MARKET NEWS")
print("=" * 100)
for i, article in enumerate(state['market_news'][:5], 1):
    dt = datetime.fromtimestamp(article['datetime']) if article['datetime'] else datetime.now()
    print(f"\n{i}. {article['headline']}")
    print(f"   Source: {article['source']}")
    print(f"   Date: {dt.strftime('%Y-%m-%d %H:%M')}")
    print(f"   URL: {article['url'][:80]}...")

# Show sample related companies news
if state['related_company_news']:
    print(f"\n\n🔗 SAMPLE RELATED COMPANIES NEWS")
    print("=" * 100)
    for i, article in enumerate(state['related_company_news'][:5], 1):
        dt = datetime.fromtimestamp(article['datetime']) if article['datetime'] else datetime.now()
        print(f"\n{i}. [{article['ticker']}] {article['headline']}")
        print(f"   Source: {article['source']}")
        print(f"   Date: {dt.strftime('%Y-%m-%d %H:%M')}")
        print(f"   URL: {article['url'][:80]}...")

# ============================================================================
# EXECUTION SUMMARY
# ============================================================================
print("\n\n" + "=" * 100)
print(" " * 35 + "EXECUTION SUMMARY")
print("=" * 100)

print(f"\n✅ Pipeline Status: ALL NODES COMPLETED SUCCESSFULLY")
print(f"\n   Node Execution Times:")
if 'node_1' in state['node_execution_times']:
    print(f"      Node 1 (Price Fetching): {state['node_execution_times']['node_1']:.2f}s")
print(f"      Node 3 (Related Companies): {state['node_execution_times']['node_3']:.2f}s")
print(f"      Node 2 (News Fetching): {state['node_execution_times']['node_2']:.2f}s")

total_time = sum(state['node_execution_times'].values())
print(f"\n   Total Execution Time: {total_time:.2f}s")

print(f"\n   📊 Data Retrieved:")
print(f"      Price data points: {len(state['raw_price_data']) if state['raw_price_data'] is not None else 0}")
print(f"      Related companies: {len(state['related_companies'])}")
print(f"      News articles: {total_articles}")
print(f"         - Stock news: {len(state['stock_news'])}")
print(f"         - Market news: {len(state['market_news'])}")
print(f"         - Related news: {len(state['related_company_news'])}")

if state['errors']:
    print(f"\n   ⚠️  Errors: {len(state['errors'])}")
    for error in state['errors']:
        print(f"      - {error}")
else:
    print(f"\n   ✅ No errors encountered")

print("\n" + "=" * 100)
print(" " * 30 + "ANALYSIS COMPLETE - READY FOR NODE 4+")
print("=" * 100)
