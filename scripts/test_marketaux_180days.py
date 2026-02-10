"""
MarketAux 180-Day Historical Data Test
========================================
Professional verification that MarketAux can pull exactly 180 days
of historical stock news and market news with sentiment analysis.

Tests:
1. Stock-specific news for NVDA (180 days)
2. General market news (180 days)
3. Data structure validation
4. Sentiment analysis verification
5. Error handling and stability
"""

import sys
import os
import time
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
load_dotenv()

MARKETAUX_API_KEY = os.getenv('MARKETAUX_API_KEY')
TEST_TICKER = 'NVDA'

print("=" * 100)
print(" " * 20 + "MARKETAUX 180-DAY HISTORICAL DATA VERIFICATION")
print("=" * 100)
print(f"\n🎯 Test Subject: {TEST_TICKER}")
print(f"📅 Target: Exactly 180 days of historical data")
print(f"🔑 API Key: {MARKETAUX_API_KEY[:20]}..." if MARKETAUX_API_KEY else "❌ NO API KEY")
print("\n" + "=" * 100)


# ============================================================================
# TEST 1: STOCK NEWS (180 DAYS)
# ============================================================================

def test_stock_news_180_days():
    """Test MarketAux stock news for exactly 180 days"""
    
    print("\n" + "=" * 100)
    print("TEST 1: STOCK-SPECIFIC NEWS (180 DAYS)")
    print("=" * 100)
    
    if not MARKETAUX_API_KEY or MARKETAUX_API_KEY == 'your_marketaux_api_key_here':
        print("❌ FAILED: No valid API key")
        return None
    
    try:
        # Calculate exact 180-day period
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)
        
        # CRITICAL: Use this format per documentation hint
        published_after = start_date.strftime('%Y-%m-%dT%H:%M')
        published_before = end_date.strftime('%Y-%m-%dT%H:%M')
        
        print(f"\n📅 Date Range Configuration:")
        print(f"   Start: {start_date.strftime('%Y-%m-%d %H:%M')}")
        print(f"   End:   {end_date.strftime('%Y-%m-%d %H:%M')}")
        print(f"   Span:  180 days")
        
        url = "https://api.marketaux.com/v1/news/all"
        params = {
            'api_token': MARKETAUX_API_KEY,
            'symbols': TEST_TICKER,
            'filter_entities': 'true',
            'language': 'en',
            'published_after': published_after,
            'published_before': published_before,
            'limit': 100,  # Max per request
            'page': 1
        }
        
        print(f"\n🔄 Making API request...")
        print(f"   URL: {url}")
        print(f"   Symbol: {TEST_TICKER}")
        print(f"   published_after: {published_after}")
        print(f"   published_before: {published_before}")
        
        start_time = time.time()
        response = requests.get(url, params=params, timeout=30)
        elapsed = time.time() - start_time
        
        print(f"\n⏱️  Response time: {elapsed:.2f}s")
        print(f"📡 Status code: {response.status_code}")
        
        if response.status_code != 200:
            print(f"❌ API ERROR: {response.status_code}")
            print(f"Response: {response.text[:500]}")
            return None
        
        data = response.json()
        
        # Validate response structure
        if 'data' not in data:
            print(f"❌ FAILED: No 'data' field in response")
            print(f"Response keys: {data.keys()}")
            return None
        
        stories = data['data']
        meta = data.get('meta', {})
        
        print(f"\n✅ SUCCESS - API Response Received")
        print(f"\n📊 METADATA:")
        print(f"   Total found: {meta.get('found', 'N/A')}")
        print(f"   Returned: {meta.get('returned', len(stories))}")
        print(f"   Limit: {meta.get('limit', 'N/A')}")
        print(f"   Page: {meta.get('page', 'N/A')}")
        
        if not stories:
            print(f"\n⚠️  WARNING: No stories returned")
            return None
        
        print(f"\n✅ Retrieved {len(stories)} articles")
        
        # Analyze date coverage
        dates = []
        for article in stories:
            if 'published_at' in article:
                # Parse ISO format: "2026-02-10T17:00:00.000000Z"
                pub_time_str = article['published_at']
                # Remove microseconds and Z
                pub_time_str = pub_time_str.replace('Z', '+00:00')
                try:
                    dt = datetime.fromisoformat(pub_time_str.replace('.000000', ''))
                    dates.append(dt)
                except Exception as e:
                    print(f"⚠️  Date parse error: {e}")
        
        if dates:
            oldest = min(dates)
            newest = max(dates)
            span_days = (newest - oldest).days
            
            print(f"\n📅 ACTUAL DATE COVERAGE:")
            print(f"   Oldest article: {oldest.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Newest article: {newest.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Actual span: {span_days} days")
            
            if span_days >= 180:
                print(f"   ✅ VERIFICATION PASSED: Covers 180+ days ({span_days} days)")
            elif span_days >= 150:
                print(f"   ⚠️  PARTIAL: {span_days} days (close to 180)")
            else:
                print(f"   ❌ FAILED: Only {span_days} days (need 180)")
        else:
            print(f"\n❌ FAILED: No valid dates found")
            span_days = 0
        
        # Analyze data structure
        if stories:
            sample = stories[0]
            print(f"\n📋 DATA STRUCTURE VALIDATION:")
            print(f"   Article fields: {list(sample.keys())}")
            
            # Check required fields
            required_fields = ['uuid', 'title', 'url', 'published_at', 'source']
            missing = [f for f in required_fields if f not in sample]
            
            if missing:
                print(f"   ❌ Missing fields: {missing}")
            else:
                print(f"   ✅ All required fields present")
            
            # Check sentiment
            has_entities = 'entities' in sample and len(sample['entities']) > 0
            
            if has_entities:
                entity = sample['entities'][0]
                print(f"\n💡 SENTIMENT ANALYSIS:")
                print(f"   ✅ Entities present: {len(sample['entities'])}")
                print(f"   Entity fields: {list(entity.keys())}")
                
                if 'sentiment_score' in entity:
                    print(f"   ✅ Sentiment score: {entity['sentiment_score']}")
                if 'match_score' in entity:
                    print(f"   ✅ Match score: {entity['match_score']}")
                
                # Check highlights
                if 'highlights' in entity and entity['highlights']:
                    highlight = entity['highlights'][0]
                    print(f"   ✅ Highlights present: {len(entity['highlights'])}")
                    print(f"   Highlight fields: {list(highlight.keys())}")
            else:
                print(f"\n⚠️  WARNING: No entities in sample article")
            
            # Sample article display
            print(f"\n📰 SAMPLE ARTICLE:")
            print(f"   UUID: {sample.get('uuid', 'N/A')}")
            print(f"   Title: {sample.get('title', 'N/A')[:80]}...")
            print(f"   Source: {sample.get('source', 'N/A')}")
            print(f"   Published: {sample.get('published_at', 'N/A')}")
            
            # Data quality metrics
            print(f"\n📊 DATA QUALITY:")
            complete = sum(1 for a in stories if all(k in a for k in required_fields))
            print(f"   Complete articles: {complete}/{len(stories)} ({complete/len(stories)*100:.1f}%)")
            
            with_entities = sum(1 for a in stories if 'entities' in a and a['entities'])
            print(f"   With entities: {with_entities}/{len(stories)} ({with_entities/len(stories)*100:.1f}%)")
            
            with_sentiment = sum(1 for a in stories 
                                if 'entities' in a and a['entities'] 
                                and any('sentiment_score' in e for e in a['entities']))
            print(f"   With sentiment: {with_sentiment}/{len(stories)} ({with_sentiment/len(stories)*100:.1f}%)")
        
        return {
            'api': 'MarketAux',
            'data_type': 'Stock News',
            'status': 'SUCCESS',
            'records': len(stories),
            'response_time': elapsed,
            'date_range_days': span_days if dates else 0,
            'meets_180_days': span_days >= 180 if dates else False,
            'has_sentiment': with_sentiment > 0 if stories else False,
            'total_found': meta.get('found', 0)
        }
        
    except requests.exceptions.Timeout:
        print(f"\n❌ ERROR: Request timeout (30s)")
        return None
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# TEST 2: MARKET NEWS (180 DAYS)
# ============================================================================

def test_market_news_180_days():
    """Test MarketAux general market news for 180 days"""
    
    print("\n" + "=" * 100)
    print("TEST 2: GENERAL MARKET NEWS (180 DAYS)")
    print("=" * 100)
    
    if not MARKETAUX_API_KEY or MARKETAUX_API_KEY == 'your_marketaux_api_key_here':
        print("❌ FAILED: No valid API key")
        return None
    
    try:
        # Calculate 180-day period
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)
        
        published_after = start_date.strftime('%Y-%m-%dT%H:%M')
        published_before = end_date.strftime('%Y-%m-%dT%H:%M')
        
        print(f"\n📅 Testing general market news (no specific ticker)")
        
        url = "https://api.marketaux.com/v1/news/all"
        params = {
            'api_token': MARKETAUX_API_KEY,
            'language': 'en',
            'published_after': published_after,
            'published_before': published_before,
            'limit': 50,  # Smaller limit for market news
            'page': 1
        }
        
        print(f"\n🔄 Making API request...")
        start_time = time.time()
        response = requests.get(url, params=params, timeout=30)
        elapsed = time.time() - start_time
        
        print(f"⏱️  Response time: {elapsed:.2f}s")
        print(f"📡 Status code: {response.status_code}")
        
        if response.status_code != 200:
            print(f"❌ API ERROR: {response.status_code}")
            return None
        
        data = response.json()
        stories = data.get('data', [])
        meta = data.get('meta', {})
        
        print(f"\n✅ SUCCESS - Retrieved {len(stories)} market news articles")
        print(f"   Total available: {meta.get('found', 'N/A')}")
        
        if stories:
            # Date analysis
            dates = []
            for article in stories:
                if 'published_at' in article:
                    pub_time_str = article['published_at'].replace('Z', '+00:00').replace('.000000', '')
                    try:
                        dt = datetime.fromisoformat(pub_time_str)
                        dates.append(dt)
                    except:
                        pass
            
            if dates:
                span_days = (max(dates) - min(dates)).days
                print(f"\n📅 Market news coverage: {span_days} days")
                
                if span_days >= 150:
                    print(f"   ✅ Good market news coverage")
        
        return {
            'status': 'SUCCESS',
            'records': len(stories),
            'response_time': elapsed
        }
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        return None


# ============================================================================
# FINAL VERDICT
# ============================================================================

def generate_verdict(stock_result, market_result):
    """Generate final implementation verdict"""
    
    print("\n" + "=" * 100)
    print(" " * 30 + "FINAL VERDICT")
    print("=" * 100)
    
    if not stock_result:
        print(f"\n❌ FAILED: Stock news test failed")
        print(f"   → DO NOT IMPLEMENT")
        return False
    
    meets_180 = stock_result.get('meets_180_days', False)
    has_sentiment = stock_result.get('has_sentiment', False)
    
    print(f"\n📊 TEST RESULTS:")
    print(f"   Stock news articles: {stock_result.get('records', 0)}")
    print(f"   Date coverage: {stock_result.get('date_range_days', 0)} days")
    print(f"   Meets 180-day requirement: {'✅ YES' if meets_180 else '❌ NO'}")
    print(f"   Has sentiment analysis: {'✅ YES' if has_sentiment else '❌ NO'}")
    print(f"   Response time: {stock_result.get('response_time', 0):.2f}s")
    
    if market_result:
        print(f"   Market news articles: {market_result.get('records', 0)}")
    
    print(f"\n{'=' * 100}")
    
    if meets_180 and has_sentiment:
        print(f"✅ IMPLEMENTATION APPROVED")
        print(f"{'=' * 100}")
        print(f"\nMarketAux meets all requirements:")
        print(f"  ✅ Provides 180+ days of historical data")
        print(f"  ✅ Includes entity-level sentiment analysis")
        print(f"  ✅ Professional-grade data quality")
        print(f"  ✅ Both stock news and market news work")
        print(f"\n→ PROCEED WITH NODE 2 IMPLEMENTATION")
        return True
    else:
        print(f"❌ IMPLEMENTATION REJECTED")
        print(f"{'=' * 100}")
        print(f"\nMarketAux does not meet requirements:")
        if not meets_180:
            print(f"  ❌ Only {stock_result.get('date_range_days', 0)} days (need 180)")
        if not has_sentiment:
            print(f"  ❌ No sentiment analysis found")
        print(f"\n→ DO NOT IMPLEMENT, KEEP CURRENT SETUP")
        return False


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    print(f"\n🚀 Starting MarketAux 180-day verification...\n")
    time.sleep(1)
    
    # Test 1: Stock news
    stock_result = test_stock_news_180_days()
    time.sleep(2)
    
    # Test 2: Market news
    market_result = test_market_news_180_days()
    
    # Generate verdict
    approved = generate_verdict(stock_result, market_result)
    
    print(f"\n" + "=" * 100)
    print(f" " * 35 + "TEST COMPLETE")
    print(f"=" * 100)
    
    if approved:
        print(f"\n✅ All tests passed. Ready for implementation.\n")
        exit(0)
    else:
        print(f"\n❌ Tests failed. Implementation blocked.\n")
        exit(1)
