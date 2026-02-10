"""
Careful Re-Test of MarketAux and TickerTick
============================================
Double-checking implementation against documentation
to ensure no errors in testing methodology.
"""

import sys
import os
import requests
import time
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dotenv import load_dotenv
load_dotenv()

MARKETAUX_API_KEY = os.getenv('MARKETAUX_API_KEY')
TEST_TICKER = 'NVDA'

print("=" * 100)
print(" " * 25 + "CAREFUL API RE-TESTING")
print("=" * 100)


# ============================================================================
# RE-TEST 1: MARKETAUX WITH DIFFERENT APPROACHES
# ============================================================================

def retest_marketaux():
    """Re-test MarketAux with different parameter combinations"""
    
    print("\n" + "=" * 100)
    print("RE-TEST 1: MARKETAUX - TRYING DIFFERENT APPROACHES")
    print("=" * 100)
    
    if not MARKETAUX_API_KEY:
        print("❌ No API key")
        return None
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    
    # Try different date formats per documentation
    test_cases = [
        {
            'name': 'ISO format with time',
            'published_after': start_date.strftime('%Y-%m-%dT%H:%M:%S'),
            'published_before': end_date.strftime('%Y-%m-%dT%H:%M:%S')
        },
        {
            'name': 'Date only format',
            'published_after': start_date.strftime('%Y-%m-%d'),
            'published_before': end_date.strftime('%Y-%m-%d')
        },
        {
            'name': 'Using published_on with recent date',
            'published_on': (end_date - timedelta(days=7)).strftime('%Y-%m-%d')
        },
        {
            'name': 'No date filter (get latest)',
            # No date parameters
        }
    ]
    
    for test in test_cases:
        print(f"\n{'─' * 100}")
        print(f"Testing: {test['name']}")
        print(f"{'─' * 100}")
        
        url = "https://api.marketaux.com/v1/news/all"
        params = {
            'api_token': MARKETAUX_API_KEY,
            'symbols': TEST_TICKER,
            'filter_entities': 'true',
            'language': 'en',
        }
        
        # Add date parameters if present
        for key in ['published_after', 'published_before', 'published_on']:
            if key in test:
                params[key] = test[key]
                print(f"   {key}: {test[key]}")
        
        # Try without explicit limit first
        try:
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                meta = data.get('meta', {})
                stories = data.get('data', [])
                
                print(f"\n   Status: 200 ✅")
                print(f"   Found: {meta.get('found', 'N/A')}")
                print(f"   Returned: {len(stories)}")
                print(f"   Default limit: {meta.get('limit', 'N/A')}")
                
                if stories:
                    # Check date spread
                    dates = []
                    for s in stories:
                        if 'published_at' in s:
                            try:
                                dt_str = s['published_at'].replace('Z', '+00:00').replace('.000000', '')
                                dt = datetime.fromisoformat(dt_str)
                                dates.append(dt)
                            except:
                                pass
                    
                    if dates:
                        span = (max(dates) - min(dates)).days
                        print(f"   Date span: {span} days")
                        print(f"   Oldest: {min(dates).strftime('%Y-%m-%d')}")
                        print(f"   Newest: {max(dates).strftime('%Y-%m-%d')}")
            else:
                print(f"\n   Status: {response.status_code} ❌")
                print(f"   Error: {response.text[:200]}")
                
        except Exception as e:
            print(f"\n   Error: {str(e)}")
        
        time.sleep(1)  # Be nice to API


# ============================================================================
# RE-TEST 2: TICKERTICK WITH PAGINATION
# ============================================================================

def retest_tickertick_with_pagination():
    """
    Re-test TickerTick with proper pagination to get historical data.
    Documentation says use 'last' parameter for pagination.
    """
    
    print("\n" + "=" * 100)
    print("RE-TEST 2: TICKERTICK - WITH PAGINATION FOR HISTORICAL DATA")
    print("=" * 100)
    
    print(f"\n📰 TickerTick pagination approach:")
    print(f"   1. Get first 200 articles")
    print(f"   2. Use 'last' parameter with oldest article ID")
    print(f"   3. Keep paginating until we have 180 days of coverage")
    
    url = "https://api.tickertick.com/feed"
    all_stories = []
    total_requests = 0
    max_requests = 10  # Limit to 10 requests for testing (10/min rate limit)
    
    try:
        # First request
        params = {
            'q': f'tt:{TEST_TICKER.lower()}',
            'n': 200  # Max per request
        }
        
        print(f"\n🔄 Request 1: Getting first 200 articles...")
        response = requests.get(url, params=params, timeout=15)
        total_requests += 1
        
        if response.status_code != 200:
            print(f"❌ Error: {response.status_code}")
            return None
        
        data = response.json()
        stories = data.get('stories', [])
        
        if not stories:
            print(f"❌ No stories returned")
            return None
        
        all_stories.extend(stories)
        print(f"   ✅ Got {len(stories)} articles")
        
        # Get dates from first batch
        dates = []
        for s in stories:
            if 'time' in s and s['time'] > 0:
                dt = datetime.fromtimestamp(s['time'] / 1000)
                dates.append(dt)
        
        if dates:
            span = (max(dates) - min(dates)).days
            print(f"   Date span: {span} days")
            print(f"   Oldest: {min(dates).strftime('%Y-%m-%d %H:%M')}")
            print(f"   Newest: {max(dates).strftime('%Y-%m-%d %H:%M')}")
        
        # Paginate if needed
        last_id = data.get('last_id')
        
        while last_id and total_requests < max_requests:
            time.sleep(6)  # Rate limit: 10/min = 6 seconds between requests
            
            params['last'] = last_id
            
            print(f"\n🔄 Request {total_requests + 1}: Paginating (last={last_id})...")
            response = requests.get(url, params=params, timeout=15)
            total_requests += 1
            
            if response.status_code != 200:
                print(f"   ⚠️  Status: {response.status_code}, stopping")
                break
            
            data = response.json()
            stories = data.get('stories', [])
            
            if not stories:
                print(f"   ℹ️  No more stories, stopping")
                break
            
            all_stories.extend(stories)
            print(f"   ✅ Got {len(stories)} more articles (total: {len(all_stories)})")
            
            # Check date span
            new_dates = []
            for s in stories:
                if 'time' in s and s['time'] > 0:
                    dt = datetime.fromtimestamp(s['time'] / 1000)
                    new_dates.append(dt)
            
            if new_dates:
                dates.extend(new_dates)
                total_span = (max(dates) - min(dates)).days
                print(f"   Total date span: {total_span} days")
                
                # Stop if we have 180+ days
                if total_span >= 180:
                    print(f"   ✅ Reached 180+ days, stopping pagination")
                    break
            
            last_id = data.get('last_id')
            
            if not last_id:
                print(f"   ℹ️  No more pages, stopping")
                break
        
        # Final analysis
        print(f"\n{'=' * 100}")
        print(f"TICKERTICK PAGINATION RESULTS:")
        print(f"{'=' * 100}")
        print(f"   Total requests: {total_requests}")
        print(f"   Total articles: {len(all_stories)}")
        
        if dates:
            total_span = (max(dates) - min(dates)).days
            print(f"   Total date span: {total_span} days")
            print(f"   Oldest article: {min(dates).strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Newest article: {max(dates).strftime('%Y-%m-%d %H:%M:%S')}")
            
            if total_span >= 180:
                print(f"\n   ✅ SUCCESS: Covers 180+ days with pagination!")
                print(f"   📊 Estimated requests for 180 days: {total_requests}")
                print(f"   ⏱️  Time needed (at 10/min): ~{total_requests * 6} seconds")
                return {
                    'status': 'SUCCESS',
                    'total_articles': len(all_stories),
                    'date_span_days': total_span,
                    'requests_needed': total_requests,
                    'meets_180_days': True
                }
            else:
                print(f"\n   ⚠️  Only {total_span} days (need 180)")
                print(f"   💡 May need more pagination requests")
                return {
                    'status': 'PARTIAL',
                    'total_articles': len(all_stories),
                    'date_span_days': total_span,
                    'requests_needed': total_requests,
                    'meets_180_days': False
                }
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    print(f"\n🔍 Re-testing both APIs carefully...\n")
    
    # Re-test MarketAux
    retest_marketaux()
    
    print(f"\n{'=' * 100}\n")
    time.sleep(2)
    
    # Re-test TickerTick with pagination
    tickertick_result = retest_tickertick_with_pagination()
    
    print(f"\n{'=' * 100}")
    print(f" " * 35 + "FINAL VERDICT")
    print(f"{'=' * 100}")
    
    if tickertick_result and tickertick_result.get('meets_180_days'):
        print(f"\n✅ TICKERTICK CAN PROVIDE 180 DAYS WITH PAGINATION!")
        print(f"\n   Implementation approach:")
        print(f"   • Use pagination with 'last' parameter")
        print(f"   • ~{tickertick_result['requests_needed']} requests needed")
        print(f"   • Takes ~{tickertick_result['requests_needed'] * 6} seconds")
        print(f"   • NO API key required")
        print(f"   • NO daily limits")
        print(f"\n   💡 RECOMMEND: Implement TickerTick with pagination")
    else:
        print(f"\n⚠️  Neither API meets 180-day requirement easily")
        print(f"\n   Options:")
        print(f"   1. Keep current setup (Alpha Vantage 10 days)")
        print(f"   2. Implement TickerTick pagination (may get partial coverage)")
        print(f"   3. Upgrade to paid API tier")
    
    print(f"\n{'=' * 100}\n")
