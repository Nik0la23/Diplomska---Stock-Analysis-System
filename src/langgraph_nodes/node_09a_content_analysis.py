"""
Node 9A: Content Analysis & Feature Extraction
Phase 1 of two-phase anomaly detection - scores and tags news content.

This node extracts quantifiable signals from news articles WITHOUT making
filtering decisions. It produces numerical scores and metadata that will be
used by Node 9B for final validation.

Runs AFTER: Node 2 (news fetching)
Runs BEFORE: Node 5 (sentiment analysis), Node 4 (technical analysis)
Can run in PARALLEL with: Nothing (must complete before analysis layer)
"""

import re
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from src.utils.domain_authority import (
    SENSATIONALISM_KEYWORDS,
    URGENCY_PHRASES,
    HEDGING_PHRASES,
    FINANCIAL_KEYWORDS,
    TECHNICAL_FINANCIAL_TERMS,
    get_domain_credibility
)

logger = logging.getLogger(__name__)


# ============================================================================
# HELPER FUNCTION 1: Sensationalism Scoring
# ============================================================================

def calculate_sensationalism_score(text: str) -> float:
    """
    Calculate sensationalism score based on clickbait patterns.
    
    Detects:
    - Excessive punctuation (!!!, ???)
    - ALL CAPS words
    - Emotional/hyperbolic language
    - Sensationalist keywords
    
    Args:
        text: Article text (title + summary combined)
        
    Returns:
        Score from 0.0 (not sensational) to 1.0 (highly sensational)
        
    Example:
        >>> calculate_sensationalism_score("SHOCKING NEWS!!!")
        0.85
        >>> calculate_sensationalism_score("Company reports earnings")
        0.15
    """
    if not text:
        return 0.0
    
    score = 0.0
    text_upper = text.upper()
    
    # Check 1: Excessive punctuation (weight: 0.3)
    exclamation_count = text.count('!!')
    question_count = text.count('??')
    if exclamation_count >= 2 or question_count >= 2:
        score += 0.3
    elif exclamation_count >= 1 or question_count >= 1:
        score += 0.15
    
    # Check 2: ALL CAPS words (weight: 0.3)
    words = text.split()
    if len(words) > 0:
        caps_words = [w for w in words if len(w) > 2 and w.isupper() and w.isalpha()]
        caps_ratio = len(caps_words) / len(words)
        
        if caps_ratio > 0.3:  # More than 30% caps
            score += 0.3
        elif caps_ratio > 0.15:  # More than 15% caps
            score += 0.2
        elif caps_ratio > 0.05:  # More than 5% caps
            score += 0.1
    
    # Check 3: Sensationalist keywords (weight: 0.4)
    keyword_matches = sum(1 for keyword in SENSATIONALISM_KEYWORDS if keyword in text_upper)
    if keyword_matches >= 3:
        score += 0.4
    elif keyword_matches >= 2:
        score += 0.3
    elif keyword_matches >= 1:
        score += 0.15
    
    # Normalize to 0-1 range (max possible score is 1.0)
    return min(score, 1.0)


# ============================================================================
# HELPER FUNCTION 2: Urgency Scoring
# ============================================================================

def calculate_urgency_score(text: str) -> float:
    """
    Calculate urgency score based on time-pressure phrases.
    
    Detects:
    - Urgency keywords (BREAKING, URGENT, ACT NOW)
    - Time-limited language
    - Call-to-action pressure
    
    Args:
        text: Article text (title + summary combined)
        
    Returns:
        Score from 0.0 (no urgency) to 1.0 (extreme urgency)
        
    Example:
        >>> calculate_urgency_score("BREAKING: URGENT NEWS - ACT NOW!")
        0.95
        >>> calculate_urgency_score("Company announces quarterly results")
        0.0
    """
    if not text:
        return 0.0
    
    text_upper = text.upper()
    
    # Count urgency phrase matches
    matches = []
    for phrase in URGENCY_PHRASES:
        if phrase in text_upper:
            matches.append(phrase)
    
    match_count = len(matches)
    
    # Score based on number of urgency indicators
    if match_count == 0:
        return 0.0
    elif match_count == 1:
        return 0.3
    elif match_count == 2:
        return 0.6
    elif match_count >= 3:
        return 0.9
    
    return 0.0


# ============================================================================
# HELPER FUNCTION 3: Unverified Claims Detection
# ============================================================================

def calculate_unverified_claims_score(text: str) -> float:
    """
    Calculate unverified claims score based on hedging language.
    
    Detects:
    - Hedging phrases (allegedly, rumored, sources say)
    - Speculation vs. fact patterns
    - Missing source attribution
    
    Args:
        text: Article text (title + summary combined)
        
    Returns:
        Score from 0.0 (verified/factual) to 1.0 (highly speculative)
        
    Example:
        >>> calculate_unverified_claims_score("Allegedly, sources say...")
        0.7
        >>> calculate_unverified_claims_score("Company officially announced...")
        0.1
    """
    if not text:
        return 0.0
    
    text_lower = text.lower()
    
    # Count hedging phrase matches
    matches = []
    for phrase in HEDGING_PHRASES:
        if phrase in text_lower:
            matches.append(phrase)
    
    match_count = len(matches)
    
    # Score based on number of hedging indicators
    if match_count == 0:
        return 0.1  # Even factual articles have some uncertainty
    elif match_count == 1:
        return 0.4
    elif match_count == 2:
        return 0.7
    elif match_count >= 3:
        return 0.95
    
    return 0.1


# ============================================================================
# HELPER FUNCTION 4: Source Credibility Assessment
# ============================================================================

def assess_source_credibility(source_url: str, source_name: str) -> float:
    """
    Assess source credibility based on domain authority.
    
    Uses domain authority mapping from domain_authority.py:
    - Tier 1 (0.9-1.0): Bloomberg, Reuters, WSJ
    - Tier 2 (0.7-0.8): CNBC, Forbes, MarketWatch
    - Tier 3 (0.5-0.6): Yahoo Finance, Seeking Alpha
    - Tier 4 (0.1-0.4): Unknown domains
    
    Args:
        source_url: Article URL
        source_name: Source name (fallback)
        
    Returns:
        Score from 0.0 (untrusted) to 1.0 (highly trusted)
        
    Example:
        >>> assess_source_credibility("https://bloomberg.com/news", "Bloomberg")
        0.95
        >>> assess_source_credibility("https://unknown-blog.com", "Unknown")
        0.30
    """
    return get_domain_credibility(source_url, source_name)


# ============================================================================
# HELPER FUNCTION 5: Complexity Analysis
# ============================================================================

def calculate_complexity_score(text: str) -> float:
    """
    Calculate text complexity score based on technical jargon density.
    
    Measures:
    - Technical financial terms density
    - Average word length
    - Sentence complexity
    
    Args:
        text: Article text (title + summary combined)
        
    Returns:
        Score from 0.0 (simple) to 1.0 (highly technical)
        
    Example:
        >>> calculate_complexity_score("EBITDA, derivatives, and arbitrage...")
        0.8
        >>> calculate_complexity_score("Stock price went up today")
        0.2
    """
    if not text:
        return 0.0
    
    text_lower = text.lower()
    words = text.split()
    
    if len(words) == 0:
        return 0.0
    
    score = 0.0
    
    # Check 1: Technical financial terms (weight: 0.6)
    technical_matches = sum(1 for term in TECHNICAL_FINANCIAL_TERMS if term in text_lower)
    tech_density = technical_matches / max(len(words) / 10, 1)  # Per 10 words
    
    if tech_density > 1.5:
        score += 0.6
    elif tech_density > 1.0:
        score += 0.5
    elif tech_density > 0.5:
        score += 0.3
    elif tech_density > 0:
        score += 0.15
    
    # Check 2: Average word length (weight: 0.2)
    avg_word_length = sum(len(w) for w in words) / len(words)
    if avg_word_length > 7:
        score += 0.2
    elif avg_word_length > 6:
        score += 0.1
    
    # Check 3: Long sentences (weight: 0.2)
    sentences = text.split('.')
    if len(sentences) > 0:
        avg_sentence_length = len(words) / len(sentences)
        if avg_sentence_length > 25:
            score += 0.2
        elif avg_sentence_length > 20:
            score += 0.1
    
    return min(score, 1.0)


# ============================================================================
# HELPER FUNCTION 6: Entity & Keyword Recognition
# ============================================================================

def extract_content_tags(text: str, ticker: str) -> Dict[str, List[str]]:
    """
    Extract content tags including keywords, entities, and topics.
    
    Extracts:
    - Financial keywords (fraud, investigation, merger, etc.)
    - Named entities (companies, executives)
    - Temporal markers (past, current, future)
    - Topic classification
    
    Args:
        text: Article text (title + summary combined)
        ticker: Stock ticker for context
        
    Returns:
        Dictionary with categorized tags
        
    Example:
        >>> extract_content_tags("SEC investigates fraud at XYZ Corp", "XYZ")
        {'keywords': ['SEC', 'investigation', 'fraud'], 'topic': 'regulatory_action'}
    """
    text_lower = text.lower()
    text_upper = text.upper()
    
    tags = {
        'keywords': [],
        'topic': 'general',
        'temporal': 'current',
        'entities': []
    }
    
    # Extract financial keywords
    for keyword in FINANCIAL_KEYWORDS:
        if keyword.lower() in text_lower:
            tags['keywords'].append(keyword)
    
    # Topic classification based on keywords
    if any(word in text_lower for word in ['fraud', 'investigation', 'sec', 'lawsuit', 'probe']):
        tags['topic'] = 'regulatory_action'
    elif any(word in text_lower for word in ['earnings', 'revenue', 'profit', 'results', 'guidance']):
        tags['topic'] = 'earnings_report'
    elif any(word in text_lower for word in ['merger', 'acquisition', 'takeover', 'buyout']):
        tags['topic'] = 'merger_acquisition'
    elif any(word in text_lower for word in ['bankruptcy', 'insolvency', 'default']):
        tags['topic'] = 'financial_distress'
    elif any(word in text_lower for word in ['ipo', 'offering', 'debut']):
        tags['topic'] = 'ipo_offering'
    
    # Temporal classification
    if any(word in text_lower for word in ['will', 'plans', 'expects', 'forecasts', 'outlook']):
        tags['temporal'] = 'future'
    elif any(word in text_lower for word in ['reported', 'announced', 'filed', 'disclosed']):
        tags['temporal'] = 'past'
    
    # Add ticker as entity
    if ticker.upper() in text_upper:
        tags['entities'].append(ticker.upper())
    
    # Extract company names (basic - looks for common patterns)
    # This is simplified; a full implementation would use NER
    corp_patterns = [' Corp', ' Inc', ' Ltd', ' LLC', ' Company']
    for pattern in corp_patterns:
        if pattern in text:
            # Very basic extraction - just note that companies mentioned
            if 'companies_mentioned' not in tags:
                tags['companies_mentioned'] = True
            break
    
    return tags


# ============================================================================
# HELPER FUNCTION 7: Composite Score Calculation
# ============================================================================

def calculate_composite_anomaly_score(scores: Dict[str, float]) -> float:
    """
    Calculate composite anomaly score from individual scores.
    
    Weighted combination:
    - Sensationalism: 25%
    - Urgency: 20%
    - Unverified Claims: 25%
    - Source Credibility: 30% (inverted - low credibility = high anomaly)
    
    Args:
        scores: Dictionary with individual scores
        
    Returns:
        Composite score from 0.0 (clean) to 1.0 (highly anomalous)
        
    Example:
        >>> scores = {
        ...     'sensationalism': 0.8,
        ...     'urgency': 0.9,
        ...     'unverified': 0.7,
        ...     'credibility': 0.2
        ... }
        >>> calculate_composite_anomaly_score(scores)
        0.79
    """
    sensationalism = scores.get('sensationalism', 0.0)
    urgency = scores.get('urgency', 0.0)
    unverified = scores.get('unverified', 0.0)
    credibility = scores.get('credibility', 0.5)
    
    # Invert credibility (low credibility = high anomaly)
    inverted_credibility = 1.0 - credibility
    
    # Weighted combination
    composite = (
        sensationalism * 0.25 +
        urgency * 0.20 +
        unverified * 0.25 +
        inverted_credibility * 0.30
    )
    
    return min(composite, 1.0)


# ============================================================================
# HELPER FUNCTION 8: Process Single Article
# ============================================================================

def process_article(
    article: Dict[str, Any],
    ticker: str
) -> Dict[str, Any]:
    """
    Process a single article and add content scores.
    
    Args:
        article: Article dictionary
        ticker: Stock ticker for context
        
    Returns:
        Article with embedded content scores
    """
    # Combine title and summary for analysis
    title = article.get('headline', article.get('title', ''))
    summary = article.get('summary', article.get('description', ''))
    text = f"{title} {summary}"
    
    # Extract source information
    source_url = article.get('url', '')
    source_name = article.get('source', '')
    
    # Calculate all scores
    sensationalism = calculate_sensationalism_score(text)
    urgency = calculate_urgency_score(text)
    unverified = calculate_unverified_claims_score(text)
    credibility = assess_source_credibility(source_url, source_name)
    complexity = calculate_complexity_score(text)
    
    # Calculate composite score
    scores_dict = {
        'sensationalism': sensationalism,
        'urgency': urgency,
        'unverified': unverified,
        'credibility': credibility
    }
    composite = calculate_composite_anomaly_score(scores_dict)
    
    # Extract content tags
    tags = extract_content_tags(text, ticker)
    
    # Create enriched article (copy original + add scores)
    enriched_article = article.copy()
    enriched_article.update({
        'sensationalism_score': round(sensationalism, 3),
        'urgency_score': round(urgency, 3),
        'unverified_claims_score': round(unverified, 3),
        'source_credibility_score': round(credibility, 3),
        'complexity_score': round(complexity, 3),
        'composite_anomaly_score': round(composite, 3),
        'content_tags': tags
    })
    
    return enriched_article


# ============================================================================
# MAIN NODE FUNCTION
# ============================================================================

def content_analysis_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 9A: Content Analysis & Feature Extraction
    
    Execution flow:
    1. Extract news from state (stock, market, related)
    2. For each article: calculate scores and extract tags
    3. Embed scores directly into article dictionaries
    4. Generate content analysis summary
    5. Update state with cleaned news lists
    
    Args:
        state: LangGraph state containing news data
        
    Returns:
        Updated state with cleaned news and content analysis summary
    """
    start_time = datetime.now()
    ticker = state['ticker']
    
    try:
        logger.info(f"Node 9A: Starting content analysis for {ticker}")
        
        # ====================================================================
        # STEP 1: Extract raw news from state
        # ====================================================================
        stock_news = state.get('stock_news', [])
        market_news = state.get('market_news', [])
        related_news = state.get('related_company_news', [])
        
        total_input_articles = len(stock_news) + len(market_news) + len(related_news)
        logger.info(f"Node 9A: Processing {total_input_articles} articles")
        logger.info(f"  - Stock news: {len(stock_news)}")
        logger.info(f"  - Market news: {len(market_news)}")
        logger.info(f"  - Related news: {len(related_news)}")
        
        # ====================================================================
        # STEP 2: Process each news category
        # ====================================================================
        cleaned_stock_news = []
        cleaned_market_news = []
        cleaned_related_news = []
        
        # Process stock news
        for article in stock_news:
            try:
                enriched = process_article(article, ticker)
                cleaned_stock_news.append(enriched)
            except Exception as e:
                logger.warning(f"Node 9A: Failed to process stock article: {str(e)}")
                # Include article without scores (better than losing it)
                cleaned_stock_news.append(article)
        
        # Process market news
        for article in market_news:
            try:
                enriched = process_article(article, ticker)
                cleaned_market_news.append(enriched)
            except Exception as e:
                logger.warning(f"Node 9A: Failed to process market article: {str(e)}")
                cleaned_market_news.append(article)
        
        # Process related company news
        for article in related_news:
            try:
                enriched = process_article(article, ticker)
                cleaned_related_news.append(enriched)
            except Exception as e:
                logger.warning(f"Node 9A: Failed to process related article: {str(e)}")
                cleaned_related_news.append(article)
        
        # ====================================================================
        # STEP 3: Generate content analysis summary
        # ====================================================================
        all_articles = cleaned_stock_news + cleaned_market_news + cleaned_related_news
        
        # Calculate aggregate statistics
        total_processed = len(all_articles)
        
        if total_processed > 0:
            avg_sensationalism = sum(a.get('sensationalism_score', 0) for a in all_articles) / total_processed
            avg_urgency = sum(a.get('urgency_score', 0) for a in all_articles) / total_processed
            avg_unverified = sum(a.get('unverified_claims_score', 0) for a in all_articles) / total_processed
            avg_credibility = sum(a.get('source_credibility_score', 0.5) for a in all_articles) / total_processed
            avg_composite = sum(a.get('composite_anomaly_score', 0) for a in all_articles) / total_processed
            
            # Count high-risk articles (composite > 0.7)
            high_risk_count = sum(1 for a in all_articles if a.get('composite_anomaly_score', 0) > 0.7)
            
            # Collect all keywords
            all_keywords = []
            for article in all_articles:
                tags = article.get('content_tags', {})
                all_keywords.extend(tags.get('keywords', []))
            
            # Count keyword frequencies
            keyword_freq = {}
            for kw in all_keywords:
                keyword_freq[kw] = keyword_freq.get(kw, 0) + 1
            
            # Top 10 keywords
            top_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Source credibility distribution
            credibility_dist = {
                'high': sum(1 for a in all_articles if a.get('source_credibility_score', 0) >= 0.8),
                'medium': sum(1 for a in all_articles if 0.5 <= a.get('source_credibility_score', 0) < 0.8),
                'low': sum(1 for a in all_articles if a.get('source_credibility_score', 0) < 0.5)
            }
            
            summary = {
                'total_articles_processed': total_processed,
                'articles_by_type': {
                    'stock': len(cleaned_stock_news),
                    'market': len(cleaned_market_news),
                    'related': len(cleaned_related_news)
                },
                'average_scores': {
                    'sensationalism': round(avg_sensationalism, 3),
                    'urgency': round(avg_urgency, 3),
                    'unverified_claims': round(avg_unverified, 3),
                    'source_credibility': round(avg_credibility, 3),
                    'composite_anomaly': round(avg_composite, 3)
                },
                'high_risk_articles': high_risk_count,
                'high_risk_percentage': round(high_risk_count / total_processed * 100, 1),
                'top_keywords': top_keywords,
                'source_credibility_distribution': credibility_dist
            }
        else:
            # No articles to process
            summary = {
                'total_articles_processed': 0,
                'articles_by_type': {
                    'stock': 0,
                    'market': 0,
                    'related': 0
                },
                'average_scores': {
                    'sensationalism': 0.0,
                    'urgency': 0.0,
                    'unverified_claims': 0.0,
                    'source_credibility': 0.0,
                    'composite_anomaly': 0.0
                },
                'high_risk_articles': 0,
                'high_risk_percentage': 0.0,
                'top_keywords': [],
                'source_credibility_distribution': {'high': 0, 'medium': 0, 'low': 0}
            }
        
        # ====================================================================
        # STEP 4: Update state
        # ====================================================================
        state['cleaned_stock_news'] = cleaned_stock_news
        state['cleaned_market_news'] = cleaned_market_news
        state['cleaned_related_company_news'] = cleaned_related_news
        state['content_analysis_summary'] = summary
        
        elapsed = (datetime.now() - start_time).total_seconds()
        state['node_execution_times']['node_9a'] = elapsed
        
        # ====================================================================
        # STEP 5: Log results
        # ====================================================================
        logger.info(f"Node 9A: Completed in {elapsed:.2f}s")
        logger.info(f"Node 9A: Processed {total_processed} articles")
        logger.info(f"Node 9A: Average composite anomaly score: {summary['average_scores']['composite_anomaly']:.3f}")
        logger.info(f"Node 9A: High-risk articles: {summary['high_risk_articles']} ({summary['high_risk_percentage']}%)")
        logger.info(f"Node 9A: Source distribution - High: {summary['source_credibility_distribution']['high']}, "
                   f"Medium: {summary['source_credibility_distribution']['medium']}, "
                   f"Low: {summary['source_credibility_distribution']['low']}")
        
        return state
        
    except Exception as e:
        logger.error(f"Node 9A: Critical error for {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Update state with error
        state['errors'].append(f"Node 9A: {str(e)}")
        
        # Provide empty cleaned lists (graceful degradation)
        state['cleaned_stock_news'] = state.get('stock_news', [])
        state['cleaned_market_news'] = state.get('market_news', [])
        state['cleaned_related_company_news'] = state.get('related_company_news', [])
        state['content_analysis_summary'] = {
            'total_articles_processed': 0,
            'error': str(e)
        }
        
        elapsed = (datetime.now() - start_time).total_seconds()
        state['node_execution_times']['node_9a'] = elapsed
        
        return state
