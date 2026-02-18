"""
LangGraph Workflow Builder

Defines the complete stock analysis workflow with all nodes and edges.

Current Flow (Nodes 1-8, 9A):
1. Node 1: Price Data Fetching
2. Node 3: Related Companies Detection
3. Node 2: Multi-Source News Fetching
4. Node 9A: Content Analysis & Feature Extraction
5. PARALLEL: Nodes 4, 5, 6, 7 (Technical, Sentiment, Market Context, Monte Carlo)
6. Node 8: News Verification & Learning (thesis innovation)

Future additions:
- Nodes 9B-15: Remaining pipeline
"""

from langgraph.graph import StateGraph, END
from typing import Literal
import logging

from src.graph.state import StockAnalysisState, create_initial_state
from src.database.db_manager import get_news_outcomes_pending
from src.langgraph_nodes.node_01_data_fetching import fetch_price_data_node
from src.langgraph_nodes.node_03_related_companies import detect_related_companies_node
from src.langgraph_nodes.node_02_news_fetching import fetch_all_news_node
from src.langgraph_nodes.node_09a_content_analysis import content_analysis_node
from src.langgraph_nodes.node_04_technical_analysis import technical_analysis_node
from src.langgraph_nodes.node_05_sentiment_analysis import sentiment_analysis_node
from src.langgraph_nodes.node_06_market_context import market_context_node
from src.langgraph_nodes.node_07_monte_carlo import monte_carlo_forecasting_node
from src.langgraph_nodes.node_08_news_verification import news_verification_node
from src.langgraph_nodes.node_09b_behavioral_anomaly import (
    behavioral_anomaly_detection_node,
)

logger = logging.getLogger(__name__)


# ============================================================================
# CONDITIONAL EDGES (Future)
# ============================================================================

def should_continue_after_parallel(state: StockAnalysisState) -> Literal["continue", "end"]:
    """
    Conditional edge: Determine if workflow should continue after parallel nodes.
    
    Args:
        state: Current workflow state
        
    Returns:
        'continue' to proceed to background outcomes then Node 8, 'end' to finish
    """
    if state.get('errors') and len(state['errors']) > 0:
        critical_keywords = ['no price data', 'failed to fetch']
        has_critical = any(
            any(keyword in error.lower() for keyword in critical_keywords)
            for error in state['errors']
        )
        if has_critical:
            logger.warning("Critical errors detected, ending workflow")
            return "end"
    logger.info("Parallel nodes complete, routing to background outcomes then Node 8")
    return "continue"


def run_background_outcomes_node(state: StockAnalysisState) -> StockAnalysisState:
    """
    Run the news-outcomes evaluator before Node 8 when needed.
    
    Nodes 1–2 have already written price_data and news_articles to the DB.
    This node queries: articles in news_articles that are 7+ days old and have
    no row in news_outcomes. If any exist, runs the background script to fill
    news_outcomes so Node 8 can learn from them. Pass-through: returns state unchanged.
    """
    ticker = state.get("ticker")
    if not ticker:
        return state
    pending = get_news_outcomes_pending(ticker=ticker, limit=500)
    if not pending:
        logger.debug(f"No pending news outcomes for {ticker}, skipping background script")
        return state
    try:
        from scripts.update_news_outcomes import run_evaluation
        bg = run_evaluation(ticker=ticker, limit=500, verbose=False)
        logger.info(
            f"Background outcomes for {ticker}: evaluated={bg['evaluated']}, "
            f"skipped={bg['skipped']}, accuracy={bg['accuracy_pct']:.1f}%"
        )
    except Exception as e:
        logger.warning(f"Background outcomes script failed (non-fatal): {e}")
    return state


# ============================================================================
# WORKFLOW BUILDER
# ============================================================================

def create_stock_analysis_workflow() -> StateGraph:
    """
    Build the complete LangGraph workflow for stock analysis.
    
    Current Workflow:
    ```
    START
      ↓
    Node 1: Price Data Fetching
      ↓
    Node 3: Related Companies Detection
      ↓
    Node 2: Multi-Source News Fetching
      ↓
    Node 9A: Content Analysis
      ↓
    ┌─────────┬─────────┬─────────┐
    ↓         ↓         ↓         ↓
    Node 4    Node 5    Node 6    Node 7
    Tech      Sent      Market    Monte Carlo
    └─────────┴─────────┴─────────┘
      ↓
    Node 8: News Verification & Learning
      ↓
    END
    ```
    
    Returns:
        Compiled StateGraph ready for execution
        
    Example:
        >>> workflow = create_stock_analysis_workflow()
        >>> result = workflow.invoke({'ticker': 'AAPL'})
        >>> print(result['technical_signal'])
        'BUY'
    """
    logger.info("Building stock analysis workflow...")
    
    # Initialize StateGraph with our state type
    workflow = StateGraph(StockAnalysisState)
    
    # ========================================================================
    # ADD NODES
    # ========================================================================
    
    # Phase 1: Data Acquisition
    workflow.add_node("fetch_price", fetch_price_data_node)
    workflow.add_node("related_companies", detect_related_companies_node)
    workflow.add_node("fetch_news", fetch_all_news_node)
    workflow.add_node("content_analysis", content_analysis_node)
    
    # Phase 2: Parallel Analysis (4 nodes execute simultaneously)
    workflow.add_node("technical_analysis", technical_analysis_node)
    workflow.add_node("sentiment_analysis", sentiment_analysis_node)
    workflow.add_node("market_context", market_context_node)
    workflow.add_node("monte_carlo", monte_carlo_forecasting_node)
    
    # Phase 3: Background outcomes (before Node 8) then Learning + 9B
    workflow.add_node("run_background_outcomes", run_background_outcomes_node)
    workflow.add_node("news_verification", news_verification_node)
    workflow.add_node("behavioral_anomaly_detection", behavioral_anomaly_detection_node)
    
    # ========================================================================
    # DEFINE EDGES (Sequential Flow)
    # ========================================================================
    
    # Set entry point
    workflow.set_entry_point("fetch_price")
    
    # Sequential pipeline
    workflow.add_edge("fetch_price", "related_companies")
    workflow.add_edge("related_companies", "fetch_news")
    workflow.add_edge("fetch_news", "content_analysis")
    
    # ========================================================================
    # PARALLEL EXECUTION (Nodes 4, 5, 6, 7)
    # ========================================================================
    
    # From Node 9A, split to ALL 4 analysis nodes
    # These execute in PARALLEL (LangGraph handles this automatically)
    workflow.add_edge("content_analysis", "technical_analysis")
    workflow.add_edge("content_analysis", "sentiment_analysis")
    workflow.add_edge("content_analysis", "market_context")
    workflow.add_edge("content_analysis", "monte_carlo")
    
    # ========================================================================
    # CONVERGENCE: parallel → background outcomes → Node 8
    # ========================================================================
    
    # All 4 parallel nodes converge to background-outcomes node, then Node 8
    workflow.add_conditional_edges(
        "technical_analysis",
        should_continue_after_parallel,
        {"continue": "run_background_outcomes", "end": END}
    )
    workflow.add_conditional_edges(
        "sentiment_analysis",
        should_continue_after_parallel,
        {"continue": "run_background_outcomes", "end": END}
    )
    workflow.add_conditional_edges(
        "market_context",
        should_continue_after_parallel,
        {"continue": "run_background_outcomes", "end": END}
    )
    workflow.add_conditional_edges(
        "monte_carlo",
        should_continue_after_parallel,
        {"continue": "run_background_outcomes", "end": END}
    )
    
    # Background outcomes runs first (fills news_outcomes when needed), then Node 8, then 9B
    workflow.add_edge("run_background_outcomes", "news_verification")
    workflow.add_edge("news_verification", "behavioral_anomaly_detection")
    workflow.add_edge("behavioral_anomaly_detection", END)
    
    logger.info("Workflow built successfully (parallel → background outcomes → Node 8)")
    
    # Compile and return
    return workflow.compile()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def run_stock_analysis(ticker: str) -> StockAnalysisState:
    """
    Convenience function to run complete stock analysis for a ticker.
    
    This is a high-level interface that:
    1. Creates initial state
    2. Builds workflow
    3. Executes workflow (includes background outcomes before Node 8 when needed)
    4. Returns final state
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'NVDA')
        
    Returns:
        Final state with all analysis results
        
    Example:
        >>> result = run_stock_analysis('AAPL')
        >>> print(f"Signal: {result['technical_signal']}")
        >>> print(f"Forecast: ${result['forecasted_price']:.2f}")
    """
    logger.info(f"Starting stock analysis for {ticker}")
    
    # Create initial state
    initial_state = create_initial_state(ticker)
    
    # Build workflow
    workflow = create_stock_analysis_workflow()
    
    # Execute workflow (background outcomes run inside graph before Node 8)
    final_state = workflow.invoke(initial_state)
    
    logger.info(f"Stock analysis complete for {ticker}")
    
    return final_state


def print_analysis_summary(state: StockAnalysisState):
    """
    Print a human-readable summary of analysis results.
    
    Args:
        state: Final state from workflow
    """
    ticker = state['ticker']
    
    print(f"\n{'='*60}")
    print(f"Stock Analysis Summary: {ticker}")
    print(f"{'='*60}")
    
    # Price Data
    if state.get('raw_price_data') is not None:
        print(f"\n📊 Price Data: {len(state['raw_price_data'])} days")
    
    # Related Companies
    if state.get('related_companies'):
        print(f"🔗 Related Companies: {', '.join(state['related_companies'][:5])}")
    
    # News Data
    stock_news_count = len(state.get('cleaned_stock_news', []))
    market_news_count = len(state.get('cleaned_market_news', []))
    print(f"📰 News Articles: {stock_news_count} stock, {market_news_count} market")
    
    # Technical Analysis
    if state.get('technical_signal'):
        print(f"\n📈 Technical Analysis:")
        print(f"   Signal: {state['technical_signal']}")
        print(f"   Confidence: {state['technical_confidence']*100:.1f}%")
        
        if state.get('technical_indicators'):
            indicators = state['technical_indicators']
            if 'rsi' in indicators:
                print(f"   RSI: {indicators['rsi']:.2f}")
            if 'macd' in indicators:
                print(f"   MACD: {indicators['macd']['macd']:.2f}")
    
    # Sentiment Analysis
    if state.get('sentiment_signal'):
        print(f"\n💭 Sentiment Analysis:")
        print(f"   Signal: {state['sentiment_signal']}")
        print(f"   Confidence: {state['sentiment_confidence']*100:.1f}%")
        print(f"   Combined Sentiment: {state['aggregated_sentiment']:.3f}")
    
    # Market Context
    if state.get('market_context'):
        mc_data = state['market_context']
        print(f"\n🌍 Market Context:")
        print(f"   Sector: {mc_data.get('sector', 'N/A')} ({mc_data.get('sector_trend', 'N/A')})")
        print(f"   Market: {mc_data.get('market_trend', 'N/A')}")
        print(f"   Signal: {mc_data.get('context_signal', 'N/A')}")
        print(f"   Correlation: {mc_data.get('market_correlation', 0):.2f}")
    
    # Monte Carlo Forecast
    if state.get('monte_carlo_results'):
        mc = state['monte_carlo_results']
        print(f"\n🎲 Monte Carlo Forecast (7 days):")
        print(f"   Current: ${mc['current_price']:.2f}")
        print(f"   Expected: ${mc['mean_forecast']:.2f}")
        print(f"   Expected Return: {mc['expected_return']:.2f}%")
        print(f"   Probability Up: {mc['probability_up']*100:.1f}%")
        print(f"   95% CI: [${mc['confidence_95']['lower']:.2f}, ${mc['confidence_95']['upper']:.2f}]")
    
    # Node 8: News Verification & Learning
    if state.get('news_impact_verification'):
        niv = state['news_impact_verification']
        print(f"\n📊 Node 8: News Verification & Learning:")
        print(f"   Learning adjustment: {niv.get('learning_adjustment', 1.0):.3f}")
        print(f"   News accuracy score: {niv.get('news_accuracy_score', 0):.1f}%")
        print(f"   Historical correlation: {niv.get('historical_correlation', 0):.3f}")
        print(f"   Sample size: {niv.get('sample_size', 0)} events")
        if niv.get('insufficient_data'):
            print(f"   (Insufficient historical data — neutral defaults)")
    
    # Errors
    if state.get('errors'):
        print(f"\n⚠️  Errors: {len(state['errors'])}")
        for error in state['errors']:
            print(f"   - {error}")
    
    # Execution Time
    if state.get('node_execution_times'):
        total_time = sum(state['node_execution_times'].values())
        print(f"\n⏱️  Execution Time: {total_time:.2f}s")
        print(f"   Node breakdown:")
        for node, time in state['node_execution_times'].items():
            print(f"   - {node}: {time:.2f}s")
    
    print(f"\n{'='*60}\n")


# ============================================================================
# MAIN EXECUTION (for testing)
# ============================================================================

if __name__ == "__main__":
    # Example usage
    import sys
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Get ticker from command line or use default
    ticker = sys.argv[1] if len(sys.argv) > 1 else 'AAPL'
    
    print(f"\n🚀 Running stock analysis for {ticker}...")
    
    # Run analysis
    result = run_stock_analysis(ticker)
    
    # Print summary
    print_analysis_summary(result)
    
    # Save results (optional)
    # import json
    # with open(f'{ticker}_analysis.json', 'w') as f:
    #     json.dump(result, f, indent=2, default=str)
