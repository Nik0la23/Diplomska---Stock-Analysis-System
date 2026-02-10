"""
YFinance Comprehensive Capability Testing
==========================================
Professional evaluation of yfinance library to determine if it can
replace paid APIs for stock prices, news, and related companies.

Tests:
1. Historical price data (180 days)
2. Stock-specific news
3. Company information & peers
4. Data quality and stability
5. Performance benchmarks
"""

import sys
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yfinance as yf
import pandas as pd

TEST_TICKER = 'NVDA'
ALTERNATIVE_TICKERS = ['AAPL', 'TSLA', 'MSFT']  # Test stability across multiple stocks

print("=" * 100)
print(" " * 25 + "YFINANCE COMPREHENSIVE CAPABILITY TEST")
print("=" * 100)
print(f"\n🎯 Primary Test Subject: {TEST_TICKER}")
print(f"📅 Target Period: 180 days (6 months)")
print(f"💡 Advantage: NO API KEY REQUIRED - Completely FREE!")
print("\n" + "=" * 100)


# ============================================================================
# TEST 1: HISTORICAL PRICE DATA (180 DAYS)
# ============================================================================

def test_yfinance_price_data():
    """Test yfinance for 180 days of historical OHLCV data"""
    
    print("\n" + "=" * 100)
    print("TEST 1: YFINANCE - HISTORICAL PRICE DATA (OHLCV)")
    print("=" * 100)
    
    try:
        print(f"\n📊 Testing 180-day price data retrieval...")
        
        # Create ticker object
        ticker = yf.Ticker(TEST_TICKER)
        
        # Calculate 180-day period
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)
        
        print(f"📅 Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Fetch historical data
        print(f"🔄 Fetching data...")
        start_time = time.time()
        df = ticker.history(period='6mo')  # 6 months
        elapsed = time.time() - start_time
        
        print(f"⏱️  Response time: {elapsed:.2f}s")
        
        if df.empty:
            print(f"❌ No data returned")
            return None
        
        print(f"\n✅ SUCCESS - Received {len(df)} trading days")
        
        # Analyze data structure
        print(f"\n📋 DATA STRUCTURE:")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Index type: {type(df.index).__name__}")
        
        # Date coverage
        first_date = df.index[0]
        last_date = df.index[-1]
        span = (last_date - first_date).days
        
        print(f"\n📅 DATE COVERAGE:")
        print(f"   First: {first_date.strftime('%Y-%m-%d')}")
        print(f"   Last:  {last_date.strftime('%Y-%m-%d')}")
        print(f"   Span:  {span} days")
        print(f"   Trading days: {len(df)}")
        
        if span >= 180:
            print(f"   ✅ COVERS 6 MONTHS (180+ days)")
        else:
            print(f"   ⚠️  Only {span} days (requested 180)")
        
        # Sample data
        print(f"\n💰 SAMPLE DATA (Latest - {last_date.strftime('%Y-%m-%d')}):")
        latest = df.iloc[-1]
        print(f"   Open:   ${latest['Open']:.2f}")
        print(f"   High:   ${latest['High']:.2f}")
        print(f"   Low:    ${latest['Low']:.2f}")
        print(f"   Close:  ${latest['Close']:.2f}")
        print(f"   Volume: {int(latest['Volume']):,}")
        
        # Data quality
        print(f"\n🔍 DATA QUALITY:")
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        total_missing = missing_counts.sum()
        
        if total_missing == 0:
            print(f"   ✅ No missing values")
        else:
            print(f"   ⚠️  Missing values found:")
            for col, count in missing_counts.items():
                if count > 0:
                    print(f"      {col}: {count} ({count/len(df)*100:.1f}%)")
        
        # Check for zero/null prices
        zero_prices = (df['Close'] == 0).sum()
        if zero_prices > 0:
            print(f"   ⚠️  {zero_prices} days with zero close price")
        else:
            print(f"   ✅ No zero prices")
        
        # Check for duplicates
        duplicates = df.index.duplicated().sum()
        if duplicates > 0:
            print(f"   ⚠️  {duplicates} duplicate dates")
        else:
            print(f"   ✅ No duplicate dates")
        
        # Volatility check (extreme price changes might indicate data issues)
        df['pct_change'] = df['Close'].pct_change()
        extreme_moves = (df['pct_change'].abs() > 0.2).sum()  # >20% daily moves
        print(f"   Extreme moves (>20%): {extreme_moves}")
        
        return {
            'api': 'yfinance',
            'data_type': 'OHLCV',
            'status': 'SUCCESS',
            'records': len(df),
            'response_time': elapsed,
            'date_range_days': span,
            'missing_values': total_missing,
            'data_quality': 'EXCELLENT' if total_missing == 0 and zero_prices == 0 else 'ACCEPTABLE'
        }
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# TEST 2: NEWS DATA
# ============================================================================

def test_yfinance_news():
    """Test yfinance for news articles"""
    
    print("\n" + "=" * 100)
    print("TEST 2: YFINANCE - NEWS DATA")
    print("=" * 100)
    
    try:
        print(f"\n📰 Testing news data retrieval...")
        
        ticker = yf.Ticker(TEST_TICKER)
        
        print(f"🔄 Fetching news...")
        start_time = time.time()
        news = ticker.news
        elapsed = time.time() - start_time
        
        print(f"⏱️  Response time: {elapsed:.2f}s")
        
        if not news or len(news) == 0:
            print(f"❌ No news returned")
            return None
        
        print(f"\n✅ SUCCESS - Received {len(news)} news articles")
        
        # Analyze data structure
        if news:
            sample = news[0]
            print(f"\n📋 DATA STRUCTURE:")
            print(f"   Fields: {list(sample.keys())}")
            
            # Analyze dates
            dates = []
            for article in news:
                if 'providerPublishTime' in article:
                    dt = datetime.fromtimestamp(article['providerPublishTime'])
                    dates.append(dt)
            
            if dates:
                oldest = min(dates)
                newest = max(dates)
                span_days = (newest - oldest).days
                
                print(f"\n📅 DATE COVERAGE:")
                print(f"   Oldest: {oldest.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"   Newest: {newest.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"   Span:   {span_days} days")
                print(f"   Total articles: {len(news)}")
                
                if span_days >= 180:
                    print(f"   ✅ COVERS 6 MONTHS (180+ days)")
                elif span_days >= 30:
                    print(f"   ⚠️  PARTIAL COVERAGE ({span_days} days)")
                else:
                    print(f"   ❌ LIMITED COVERAGE ({span_days} days)")
            
            # Sample article
            print(f"\n📰 SAMPLE ARTICLE:")
            sample = news[0]
            print(f"   Title: {sample.get('title', 'N/A')[:80]}...")
            print(f"   Publisher: {sample.get('publisher', 'N/A')}")
            print(f"   Link: {sample.get('link', 'N/A')[:60]}...")
            if 'providerPublishTime' in sample:
                pub_time = datetime.fromtimestamp(sample['providerPublishTime'])
                print(f"   Published: {pub_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Check for thumbnails/images
            with_thumbnail = sum(1 for a in news if a.get('thumbnail'))
            print(f"\n🖼️  MEDIA:")
            print(f"   Articles with thumbnail: {with_thumbnail}/{len(news)} ({with_thumbnail/len(news)*100:.1f}%)")
            
            # Data quality
            print(f"\n🔍 DATA QUALITY:")
            complete = sum(1 for a in news if all(k in a for k in ['title', 'link', 'publisher', 'providerPublishTime']))
            print(f"   Complete articles: {complete}/{len(news)} ({complete/len(news)*100:.1f}%)")
            
            return {
                'api': 'yfinance',
                'data_type': 'News',
                'status': 'SUCCESS',
                'records': len(news),
                'response_time': elapsed,
                'date_range_days': span_days if dates else 0,
                'has_sentiment': False  # yfinance doesn't provide sentiment
            }
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# TEST 3: COMPANY INFO & PEERS/COMPETITORS
# ============================================================================

def test_yfinance_company_info():
    """Test yfinance for company information and related companies"""
    
    print("\n" + "=" * 100)
    print("TEST 3: YFINANCE - COMPANY INFO & RELATED COMPANIES")
    print("=" * 100)
    
    try:
        print(f"\n🏢 Testing company information retrieval...")
        
        ticker = yf.Ticker(TEST_TICKER)
        
        # Get company info
        print(f"🔄 Fetching company info...")
        start_time = time.time()
        info = ticker.info
        elapsed = time.time() - start_time
        
        print(f"⏱️  Response time: {elapsed:.2f}s")
        
        if not info:
            print(f"❌ No company info returned")
            return None
        
        print(f"\n✅ SUCCESS - Retrieved company information")
        
        # Display key info
        print(f"\n🏢 COMPANY INFORMATION:")
        print(f"   Name: {info.get('longName', 'N/A')}")
        print(f"   Sector: {info.get('sector', 'N/A')}")
        print(f"   Industry: {info.get('industry', 'N/A')}")
        print(f"   Website: {info.get('website', 'N/A')}")
        print(f"   Market Cap: ${info.get('marketCap', 0):,}")
        print(f"   Employees: {info.get('fullTimeEmployees', 'N/A'):,}" if info.get('fullTimeEmployees') else "   Employees: N/A")
        
        # Check for recommendations (similar companies)
        print(f"\n🔍 Checking for related companies/recommendations...")
        
        has_recommendations = False
        recommendations = None
        
        # Try to get recommendations
        try:
            recommendations = ticker.recommendations
            if recommendations is not None and not recommendations.empty:
                has_recommendations = True
                print(f"   ✅ Found {len(recommendations)} analyst recommendations")
        except:
            print(f"   ⚠️  Recommendations not available")
        
        # Check info dict for related fields
        related_fields = ['companyOfficers', 'recommendationKey', 'recommendationMean']
        found_fields = [f for f in related_fields if f in info]
        
        print(f"\n📊 AVAILABLE DATA FIELDS:")
        print(f"   Total fields: {len(info)}")
        print(f"   Sample fields: {list(info.keys())[:10]}")
        
        # Check specifically for sector/industry peers
        sector = info.get('sector')
        industry = info.get('industry')
        
        if sector and industry:
            print(f"\n🎯 PEER IDENTIFICATION:")
            print(f"   Can identify peers by:")
            print(f"   - Sector: {sector}")
            print(f"   - Industry: {industry}")
            print(f"   ⚠️  But yfinance doesn't provide direct peer list")
            print(f"   💡 Would need to query multiple tickers and filter by sector/industry")
        
        return {
            'api': 'yfinance',
            'data_type': 'Company Info',
            'status': 'SUCCESS',
            'has_direct_peers': False,
            'has_sector_industry': sector is not None and industry is not None,
            'response_time': elapsed,
            'info_fields': len(info)
        }
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# TEST 4: STABILITY & RELIABILITY TEST (Multiple Stocks)
# ============================================================================

def test_yfinance_stability():
    """Test yfinance stability across multiple stocks"""
    
    print("\n" + "=" * 100)
    print("TEST 4: YFINANCE - STABILITY & RELIABILITY")
    print("=" * 100)
    
    print(f"\n🔬 Testing stability across {len(ALTERNATIVE_TICKERS)} different stocks...")
    
    results = []
    
    for test_ticker in ALTERNATIVE_TICKERS:
        print(f"\n📊 Testing {test_ticker}...")
        
        try:
            ticker = yf.Ticker(test_ticker)
            
            start_time = time.time()
            df = ticker.history(period='6mo')
            elapsed = time.time() - start_time
            
            if df.empty:
                print(f"   ❌ Failed - no data")
                results.append({'ticker': test_ticker, 'success': False})
                continue
            
            span = (df.index[-1] - df.index[0]).days
            print(f"   ✅ Success - {len(df)} days ({span} calendar days) in {elapsed:.2f}s")
            
            results.append({
                'ticker': test_ticker,
                'success': True,
                'records': len(df),
                'span_days': span,
                'response_time': elapsed
            })
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            results.append({'ticker': test_ticker, 'success': False})
        
        time.sleep(0.5)  # Be nice to the API
    
    # Summary
    successful = sum(1 for r in results if r.get('success'))
    
    print(f"\n📊 STABILITY SUMMARY:")
    print(f"   Tested: {len(ALTERNATIVE_TICKERS)} stocks")
    print(f"   Successful: {successful}/{len(ALTERNATIVE_TICKERS)} ({successful/len(ALTERNATIVE_TICKERS)*100:.1f}%)")
    
    if successful == len(ALTERNATIVE_TICKERS):
        print(f"   ✅ HIGHLY RELIABLE - All tests passed")
    elif successful >= len(ALTERNATIVE_TICKERS) * 0.8:
        print(f"   ⚠️  MOSTLY RELIABLE - Some failures")
    else:
        print(f"   ❌ UNRELIABLE - Many failures")
    
    if successful > 0:
        avg_time = sum(r['response_time'] for r in results if r.get('success')) / successful
        print(f"   Average response time: {avg_time:.2f}s")
    
    return {
        'tested': len(ALTERNATIVE_TICKERS),
        'successful': successful,
        'reliability': successful / len(ALTERNATIVE_TICKERS) * 100
    }


# ============================================================================
# COMPARISON WITH OTHER APIs
# ============================================================================

def generate_comparison():
    """Compare yfinance with Polygon, Finnhub, and Alpha Vantage"""
    
    print("\n" + "=" * 100)
    print(" " * 25 + "YFINANCE vs OTHER APIs - COMPARISON")
    print("=" * 100)
    
    comparison_table = """
┌─────────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
│ FEATURE         │ YFINANCE     │ POLYGON.IO   │ FINNHUB      │ ALPHA VANT.  │
├─────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ API Key Needed  │ ❌ NO        │ ✅ YES       │ ✅ YES       │ ✅ YES       │
├─────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ Price Data      │ ✅ 180 days  │ ✅ 180 days  │ ❌ Limited   │ ⚠️ Limited   │
├─────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ News Data       │ ✅ YES       │ ❌ NO        │ ✅ 3 days    │ ✅ 10 days   │
├─────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ News Date Range │ ? (testing)  │ N/A          │ 3 days       │ 10 days      │
├─────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ Peers/Related   │ ❌ NO        │ ❌ NO        │ ✅ YES       │ ❌ NO        │
├─────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ Sentiment       │ ❌ NO        │ ❌ NO        │ ❌ NO        │ ✅ YES       │
├─────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ Rate Limits     │ 🟢 Generous  │ 5/min        │ 60/min       │ 25/day       │
├─────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ Speed           │ ? (testing)  │ 0.5s         │ 0.5s         │ 4.2s         │
├─────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ Reliability     │ ? (testing)  │ ✅ High      │ ✅ High      │ ⚠️ Med       │
├─────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ Data Quality    │ ? (testing)  │ ✅ Excellent │ ✅ Good      │ ✅ Good      │
├─────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ Cost            │ 🟢 FREE      │ 🟢 FREE      │ 🟢 FREE      │ 🟢 FREE      │
└─────────────────┴──────────────┴──────────────┴──────────────┴──────────────┘
"""
    
    print(comparison_table)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    results = []
    
    print(f"\n🚀 Starting comprehensive yfinance testing...")
    time.sleep(1)
    
    # Test 1: Price data
    result1 = test_yfinance_price_data()
    results.append(result1)
    time.sleep(1)
    
    # Test 2: News data
    result2 = test_yfinance_news()
    results.append(result2)
    time.sleep(1)
    
    # Test 3: Company info
    result3 = test_yfinance_company_info()
    results.append(result3)
    time.sleep(1)
    
    # Test 4: Stability
    result4 = test_yfinance_stability()
    results.append(result4)
    
    # Generate comparison
    generate_comparison()
    
    # Final recommendation
    print(f"\n" + "=" * 100)
    print(" " * 30 + "FINAL ASSESSMENT")
    print("=" * 100)
    
    successful_tests = sum(1 for r in results if r and r.get('status') == 'SUCCESS')
    
    print(f"\n📊 YFINANCE TEST RESULTS:")
    print(f"   Tests passed: {successful_tests}/{len([r for r in results if r is not None])}")
    
    if result1:
        print(f"\n✅ PRICE DATA:")
        print(f"   Coverage: {result1['date_range_days']} days")
        print(f"   Quality: {result1['data_quality']}")
        print(f"   Speed: {result1['response_time']:.2f}s")
    
    if result2:
        print(f"\n📰 NEWS DATA:")
        print(f"   Coverage: {result2['date_range_days']} days")
        print(f"   Articles: {result2['records']}")
        print(f"   Speed: {result2['response_time']:.2f}s")
    
    if result3:
        print(f"\n🏢 COMPANY INFO:")
        print(f"   Direct peers: {'❌ Not available' if not result3.get('has_direct_peers') else '✅ Available'}")
        print(f"   Sector/Industry: {'✅ Available' if result3.get('has_sector_industry') else '❌ Not available'}")
    
    if result4:
        print(f"\n🔬 RELIABILITY:")
        print(f"   Success rate: {result4['reliability']:.1f}%")
    
    print(f"\n" + "=" * 100)
    print(" " * 35 + "RECOMMENDATION")
    print("=" * 100)
    
    print(f"""
🎯 YFINANCE SUITABILITY FOR YOUR PROJECT:

PROS:
✅ NO API KEY REQUIRED - Simplest setup
✅ 180-day historical price data
✅ News data available
✅ No rate limit worries
✅ Already in your requirements.txt

CONS:
❌ No direct peers/related companies API
❌ No sentiment analysis
❌ News date range may be limited
❌ Unofficial API (could break)

RECOMMENDATION:
""")
    
    if result1 and result1['date_range_days'] >= 180 and result2 and result2['date_range_days'] >= 30:
        print("🌟 YFINANCE AS PRIMARY SOURCE")
        print("   Use yfinance for both price AND news (if news covers 30+ days)")
        print("   Supplement with Finnhub for peers only")
    elif result1 and result1['date_range_days'] >= 180:
        print("⚡ HYBRID APPROACH RECOMMENDED")
        print("   - yfinance: Price data (180 days)")
        print("   - Alpha Vantage: News + Sentiment (10 days)")
        print("   - Finnhub: Peers + Market context")
    else:
        print("⚠️  KEEP CURRENT MULTI-API APPROACH")
        print("   yfinance has limitations, stick with Polygon + Alpha Vantage + Finnhub")
    
    print(f"\n" + "=" * 100)
    print(f"✅ Testing complete!\n")
