"""
NYT Archive API Comprehensive Test
====================================
Testing New York Times Archive API for 180-day historical coverage.

API Structure:
- Endpoint: https://api.nytimes.com/svc/archive/v1/{year}/{month}.json
- Returns: ALL articles from that month (can be 20MB)
- For 6 months: 6 requests needed
- Rate limit: 5/min (12s delay recommended)
- Daily limit: 500 requests/day

Tests:
1. Fetch last 6 months of NYT articles
2. Filter for business/finance news
3. Search for stock ticker mentions
4. Analyze data structure and quality
5. Determine suitability for thesis project
"""

import sys
import os
import requests
import time
from datetime import datetime, timedelta
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dotenv import load_dotenv
load_dotenv()

NYT_API_KEY = os.getenv('NYT_API_KEY')
TEST_TICKER = 'NVDA'
COMPANY_NAME = 'Nvidia'

print("=" * 100)
print(" " * 20 + "NEW YORK TIMES ARCHIVE API - 180 DAY TEST")
print("=" * 100)
print(f"\n🎯 Test Subject: {TEST_TICKER} ({COMPANY_NAME})")
print(f"📅 Target: 6 months (180 days) of archive data")
print(f"🔑 API Key: {NYT_API_KEY[:20]}..." if NYT_API_KEY else "❌ NO API KEY")
print("\n" + "=" * 100)


# ============================================================================
# TEST: FETCH 6 MONTHS OF NYT ARCHIVE
# ============================================================================

def test_nyt_archive_6_months():
    """Fetch 6 months of NYT archive and analyze for stock news"""
    
    if not NYT_API_KEY or NYT_API_KEY == 'your_nyt_api_key_here':
        print("❌ FAILED: No valid API key")
        return None
    
    # Calculate last 6 months
    current_date = datetime.now()
    months_to_fetch = []
    
    for i in range(6):
        month_date = current_date - timedelta(days=30 * i)
        months_to_fetch.append({
            'year': month_date.year,
            'month': month_date.month,
            'label': month_date.strftime('%B %Y')
        })
    
    months_to_fetch.reverse()  # Oldest first
    
    print(f"\n📅 MONTHS TO FETCH:")
    for m in months_to_fetch:
        print(f"   • {m['label']}")
    
    print(f"\n🔄 Starting archive retrieval (12s delay between requests)...")
    
    all_articles = []
    business_articles = []
    stock_mentioned = []
    
    for idx, month_info in enumerate(months_to_fetch):
        print(f"\n{'─' * 100}")
        print(f"Request {idx+1}/6: {month_info['label']}")
        print(f"{'─' * 100}")
        
        url = f"https://api.nytimes.com/svc/archive/v1/{month_info['year']}/{month_info['month']}.json"
        params = {'api-key': NYT_API_KEY}
        
        try:
            start_time = time.time()
            response = requests.get(url, params=params, timeout=60)
            elapsed = time.time() - start_time
            
            print(f"   Status: {response.status_code}")
            print(f"   Response time: {elapsed:.2f}s")
            print(f"   Response size: {len(response.content) / 1024 / 1024:.2f} MB")
            
            if response.status_code != 200:
                print(f"   ❌ Error: {response.text[:200]}")
                continue
            
            data = response.json()
            
            if 'response' not in data or 'docs' not in data['response']:
                print(f"   ❌ Unexpected response structure")
                continue
            
            docs = data['response']['docs']
            all_articles.extend(docs)
            
            print(f"   ✅ Retrieved {len(docs)} articles")
            
            # Filter for business/finance articles
            month_business = []
            month_stock_mentioned = []
            
            for doc in docs:
                # Check section
                section = doc.get('section_name', '').lower()
                news_desk = doc.get('news_desk', '').lower()
                
                # Business/finance sections
                if any(keyword in section or keyword in news_desk 
                       for keyword in ['business', 'finance', 'economy', 'market', 'stock', 'technology']):
                    month_business.append(doc)
                
                # Check if ticker or company mentioned
                headline = doc.get('headline', {}).get('main', '').lower()
                abstract = doc.get('abstract', '').lower()
                lead = doc.get('lead_paragraph', '').lower()
                
                full_text = f"{headline} {abstract} {lead}"
                
                if TEST_TICKER.lower() in full_text or COMPANY_NAME.lower() in full_text:
                    month_stock_mentioned.append(doc)
            
            business_articles.extend(month_business)
            stock_mentioned.extend(month_stock_mentioned)
            
            print(f"   📊 Business articles: {len(month_business)}")
            print(f"   🎯 Mentions {TEST_TICKER}/{COMPANY_NAME}: {len(month_stock_mentioned)}")
            
            # Rate limit protection (12s recommended)
            if idx < len(months_to_fetch) - 1:
                print(f"   ⏳ Waiting 12s (rate limit)...")
                time.sleep(12)
        
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            continue
    
    # Analysis
    print(f"\n{'=' * 100}")
    print(f" " * 30 + "ANALYSIS RESULTS")
    print(f"{'=' * 100}")
    
    print(f"\n📊 TOTAL COVERAGE:")
    print(f"   Total articles: {len(all_articles):,}")
    print(f"   Business/Finance: {len(business_articles):,}")
    print(f"   Mentions {TEST_TICKER}/{COMPANY_NAME}: {len(stock_mentioned)}")
    
    if all_articles:
        # Date coverage
        dates = []
        for doc in all_articles:
            if 'pub_date' in doc:
                try:
                    dt = datetime.fromisoformat(doc['pub_date'].replace('Z', '+00:00'))
                    dates.append(dt)
                except:
                    pass
        
        if dates:
            span = (max(dates) - min(dates)).days
            print(f"\n📅 DATE COVERAGE:")
            print(f"   Oldest: {min(dates).strftime('%Y-%m-%d')}")
            print(f"   Newest: {max(dates).strftime('%Y-%m-%d')}")
            print(f"   Span: {span} days")
            
            if span >= 180:
                print(f"   ✅ COVERS 180+ DAYS!")
        
        # Analyze sections
        if business_articles:
            sections = Counter()
            desks = Counter()
            
            for doc in business_articles:
                sections[doc.get('section_name', 'Unknown')] += 1
                desks[doc.get('news_desk', 'Unknown')] += 1
            
            print(f"\n📑 BUSINESS ARTICLE SECTIONS:")
            for section, count in sections.most_common(10):
                print(f"   {section}: {count}")
            
            print(f"\n🗂️  NEWS DESKS:")
            for desk, count in desks.most_common(10):
                print(f"   {desk}: {count}")
        
        # Sample article structure
        if stock_mentioned:
            print(f"\n📰 SAMPLE ARTICLE MENTIONING {TEST_TICKER}:")
            sample = stock_mentioned[0]
            print(f"   Headline: {sample.get('headline', {}).get('main', 'N/A')[:80]}...")
            print(f"   Section: {sample.get('section_name', 'N/A')}")
            print(f"   Date: {sample.get('pub_date', 'N/A')[:10]}")
            print(f"   URL: {sample.get('web_url', 'N/A')[:60]}...")
            
            print(f"\n   Available fields: {list(sample.keys())}")
            
            # Check for useful fields
            has_abstract = 'abstract' in sample
            has_lead = 'lead_paragraph' in sample
            has_keywords = 'keywords' in sample
            has_byline = 'byline' in sample
            
            print(f"\n   📋 DATA FIELDS:")
            print(f"      Abstract: {'✅' if has_abstract else '❌'}")
            print(f"      Lead paragraph: {'✅' if has_lead else '❌'}")
            print(f"      Keywords: {'✅' if has_keywords else '❌'}")
            print(f"      Byline: {'✅' if has_byline else '❌'}")
            print(f"      Sentiment: ❌ (not provided)")
    
    # Final verdict
    print(f"\n{'=' * 100}")
    print(f" " * 35 + "VERDICT")
    print(f"{'=' * 100}")
    
    if not all_articles:
        print(f"\n❌ FAILED: No articles retrieved")
        return None
    
    meets_coverage = len(dates) > 0 and (max(dates) - min(dates)).days >= 180
    has_stock_news = len(stock_mentioned) > 0
    has_business = len(business_articles) > 100
    
    print(f"\n📊 REQUIREMENTS CHECK:")
    print(f"   180-day coverage: {'✅ YES' if meets_coverage else '❌ NO'}")
    print(f"   Stock mentions found: {'✅ YES' if has_stock_news else '❌ NO'} ({len(stock_mentioned)} articles)")
    print(f"   Business news volume: {'✅ YES' if has_business else '❌ NO'} ({len(business_articles)} articles)")
    
    if meets_coverage and has_business:
        print(f"\n✅ NYT ARCHIVE APPROVED FOR IMPLEMENTATION")
        print(f"\n   Implementation approach:")
        print(f"   • Fetch 6 months = 6 API requests")
        print(f"   • Filter for Business/Finance sections")
        print(f"   • Search article text for ticker mentions")
        print(f"   • Extract {len(stock_mentioned)} stock-relevant articles per 6 months")
        print(f"   • Cache aggressively (500 requests/day is generous)")
        print(f"   • One-time fetch takes ~72 seconds (6 requests × 12s)")
        
        return {
            'api': 'NYT Archive',
            'status': 'SUCCESS',
            'total_articles': len(all_articles),
            'business_articles': len(business_articles),
            'stock_mentions': len(stock_mentioned),
            'date_span_days': (max(dates) - min(dates)).days if dates else 0,
            'meets_180_days': meets_coverage,
            'requests_for_6_months': 6,
            'estimated_time': '72 seconds'
        }
    else:
        print(f"\n⚠️  NYT ARCHIVE HAS LIMITATIONS:")
        if not meets_coverage:
            print(f"   • Insufficient date coverage")
        if not has_business:
            print(f"   • Limited business news volume")
        if not has_stock_news:
            print(f"   • No {TEST_TICKER} mentions found")
        
        return None


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    print(f"\n🚀 Starting NYT Archive API test...")
    print(f"⏱️  This will take ~72 seconds (6 months × 12s delay)\n")
    time.sleep(2)
    
    result = test_nyt_archive_6_months()
    
    print(f"\n{'=' * 100}")
    print(f" " * 35 + "TEST COMPLETE")
    print(f"{'=' * 100}\n")
    
    if result and result['meets_180_days']:
        print(f"✅ NYT Archive API verified for 180-day coverage!")
        print(f"\n📊 Summary:")
        print(f"   • Total articles (6 months): {result['total_articles']:,}")
        print(f"   • Business articles: {result['business_articles']:,}")
        print(f"   • {TEST_TICKER} mentions: {result['stock_mentions']}")
        print(f"   • Setup time: {result['estimated_time']}")
        print(f"\n🚀 Ready for Node 2 implementation!\n")
        exit(0)
    else:
        print(f"❌ NYT Archive API not suitable for this project.\n")
        exit(1)
