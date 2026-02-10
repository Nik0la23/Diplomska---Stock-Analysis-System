"""
MarketAux Free Tier Limit Investigation
=========================================
Testing to understand actual free tier limitations
"""

import sys
import os
import requests
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dotenv import load_dotenv
load_dotenv()

MARKETAUX_API_KEY = os.getenv('MARKETAUX_API_KEY')

print("=" * 80)
print(" " * 20 + "MARKETAUX FREE TIER INVESTIGATION")
print("=" * 80)

# Test different limit values
end_date = datetime.now()
start_date = end_date - timedelta(days=180)

published_after = start_date.strftime('%Y-%m-%dT%H:%M')
published_before = end_date.strftime('%Y-%m-%dT%H:%M')

url = "https://api.marketaux.com/v1/news/all"

test_configs = [
    {'limit': None, 'desc': 'No limit specified'},
    {'limit': 3, 'desc': 'Limit = 3'},
    {'limit': 10, 'desc': 'Limit = 10'},
    {'limit': 50, 'desc': 'Limit = 50'},
    {'limit': 100, 'desc': 'Limit = 100'},
]

for config in test_configs:
    params = {
        'api_token': MARKETAUX_API_KEY,
        'symbols': 'NVDA',
        'filter_entities': 'true',
        'language': 'en',
        'published_after': published_after,
        'published_before': published_before,
    }
    
    if config['limit'] is not None:
        params['limit'] = config['limit']
    
    print(f"\n{'=' * 80}")
    print(f"TEST: {config['desc']}")
    print(f"{'=' * 80}")
    
    try:
        response = requests.get(url, params=params, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            meta = data.get('meta', {})
            stories = data.get('data', [])
            
            print(f"✅ Status: {response.status_code}")
            print(f"   Total found: {meta.get('found', 'N/A')}")
            print(f"   Returned: {meta.get('returned', len(stories))}")
            print(f"   Limit (in response): {meta.get('limit', 'N/A')}")
            print(f"   Actual articles: {len(stories)}")
        else:
            print(f"❌ Status: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")

print(f"\n{'=' * 80}")
print("CONCLUSION:")
print(f"{'=' * 80}")
print("If all tests return only 3 articles, the free tier is limited to 3 articles/request")
print("This would make MarketAux unsuitable for 180-day historical data")
print("=" * 80)
