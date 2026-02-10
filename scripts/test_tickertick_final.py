"""
TickerTick Final Test - Respecting Rate Limits
===============================================
Testing TickerTick with proper rate limiting to see
actual historical coverage without hitting 429 errors.
"""

import sys
import os
import requests
import time
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

TEST_TICKER = 'NVDA'

print("=" * 100)
print(" " * 20 + "TICKERTICK HISTORICAL COVERAGE TEST (RATE-LIMITED)")
print("=" * 100)

url = "https://api.tickertick.com/feed"
all_stories = []
dates = []

# Test with proper rate limiting
MAX_REQUESTS = 10  # Test with 10 requests
SLEEP_BETWEEN = 7  # 7 seconds = safe for 10/min limit

params = {
    'q': f'tt:{TEST_TICKER.lower()}',
    'n': 200
}

print(f"\n🎯 Testing {TEST_TICKER}")
print(f"📊 Max requests: {MAX_REQUESTS}")
print(f"⏱️  Rate limit: 10/min (7s between requests)")

for i in range(MAX_REQUESTS):
    print(f"\n{'─' * 80}")
    print(f"Request {i+1}/{MAX_REQUESTS}")
    
    try:
        response = requests.get(url, params=params, timeout=15)
        
        if response.status_code != 200:
            print(f"   ❌ Status: {response.status_code}")
            if response.status_code == 429:
                print(f"   Rate limited - stopping")
            break
        
        data = response.json()
        stories = data.get('stories', [])
        
        if not stories:
            print(f"   ℹ️  No more stories - end of feed")
            break
        
        all_stories.extend(stories)
        
        # Extract dates
        new_dates = []
        for s in stories:
            if 'time' in s and s['time'] > 0:
                dt = datetime.fromtimestamp(s['time'] / 1000)
                new_dates.append(dt)
        
        dates.extend(new_dates)
        
        # Calculate current coverage
        if dates:
            span = (max(dates) - min(dates)).days
            oldest = min(dates)
            newest = max(dates)
            
            print(f"   ✅ Articles: +{len(stories)} (total: {len(all_stories)})")
            print(f"   📅 Current span: {span} days")
            print(f"   📆 Range: {oldest.strftime('%Y-%m-%d')} to {newest.strftime('%Y-%m-%d')}")
            
            if span >= 180:
                print(f"\n   🎉 REACHED 180 DAYS!")
                break
        
        # Get last_id for next page
        last_id = data.get('last_id')
        if not last_id:
            print(f"   ℹ️  No last_id - end of feed")
            break
        
        params['last'] = last_id
        
        # Respect rate limit
        if i < MAX_REQUESTS - 1:
            print(f"   ⏳ Waiting {SLEEP_BETWEEN}s (rate limit)...")
            time.sleep(SLEEP_BETWEEN)
    
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        break

# Final summary
print(f"\n{'=' * 100}")
print(f" " * 30 + "FINAL RESULTS")
print(f"{'=' * 100}")

if dates:
    span = (max(dates) - min(dates)).days
    
    print(f"\n📊 TICKERTICK COVERAGE:")
    print(f"   Total requests: {i+1}")
    print(f"   Total articles: {len(all_stories)}")
    print(f"   Date coverage: {span} days")
    print(f"   Oldest: {min(dates).strftime('%Y-%m-%d %H:%M')}")
    print(f"   Newest: {max(dates).strftime('%Y-%m-%d %H:%M')}")
    print(f"   Time taken: ~{(i+1) * SLEEP_BETWEEN} seconds")
    
    if span >= 180:
        print(f"\n✅ SUCCESS: TickerTick can provide 180+ days!")
        print(f"   Implementation requirements:")
        print(f"   • Use pagination with 'last' parameter")
        print(f"   • Respect 10/min rate limit")
        print(f"   • Estimated: {i+1} requests needed")
    else:
        print(f"\n⚠️  LIMITED: Only {span} days after {i+1} requests")
        # Extrapolate
        if i+1 > 0 and span > 0:
            requests_needed = int((180 / span) * (i+1))
            time_needed = requests_needed * 6 / 60  # minutes
            print(f"   Extrapolated for 180 days:")
            print(f"   • Estimated requests: ~{requests_needed}")
            print(f"   • Estimated time: ~{time_needed:.1f} minutes")
            
            if requests_needed > 50:
                print(f"   ❌ Too many requests needed - impractical")
            else:
                print(f"   ⚠️  Might be workable with caching")

print(f"\n{'=' * 100}\n")
