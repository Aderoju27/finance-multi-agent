"""
Analyst Agent - Fetches data and retrieves knowledge for the Advisor.

This agent has access to:
1. Internet (Alpha Vantage API for market data)
2. Knowledge store (RAG-based retrieval)

The Analyst performs the heavy lifting: fetching market data,
computing portfolio metrics, and retrieving relevant financial knowledge.
"""

import os
import time
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from alpha_vantage.timeseries import TimeSeries

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_chroma import Chroma

# Load environment variables
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# ============================================================
# MARKET DATA MODULE (Internet Access)
# ============================================================

# Cache and rate limiting for Alpha Vantage
_av_hist_cache: Dict[str, Tuple[datetime, pd.Series]] = {}
_av_hist_lock = threading.Lock()
_last_av_call = 0.0
AV_MIN_INTERVAL = 1.5  # seconds between API calls
AV_CACHE_TTL = 3600  # 1 hour cache


def fetch_prices(symbol: str, horizon_days: int = 252) -> pd.Series:
    """
    Fetch historical daily close prices from Alpha Vantage.
    
    Features:
    - Caching to avoid redundant API calls
    - Rate limiting to respect API limits
    - Exponential backoff on failures
    """
    global _last_av_call
    
    cache_key = f"{symbol}_{horizon_days}"
    now = datetime.utcnow()
    
    # Check cache first
    with _av_hist_lock:
        cached = _av_hist_cache.get(cache_key)
        if cached and now < cached[0]:
            return cached[1].copy()
    
    if not ALPHA_VANTAGE_API_KEY:
        raise ValueError("ALPHA_VANTAGE_API_KEY not set")
    
    # Rate limiting
    elapsed = time.time() - _last_av_call
    if elapsed < AV_MIN_INTERVAL:
        time.sleep(AV_MIN_INTERVAL - elapsed)
    
    # Retry with exponential backoff
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format="pandas")
            data, _ = ts.get_daily(symbol=symbol, outputsize="compact")
            _last_av_call = time.time()
            
            close_prices = data["4. close"].sort_index().tail(horizon_days)
            close_prices = close_prices.astype(float)
            
            # Cache the result
            expires = now + timedelta(seconds=AV_CACHE_TTL)
            with _av_hist_lock:
                _av_hist_cache[cache_key] = (expires, close_prices.copy())
            
            return close_prices
            
        except Exception as e:
            error_msg = str(e).lower()
            if "rate" in error_msg or "limit" in error_msg:
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                raise
    
    raise ValueError(f"Failed to fetch prices for {symbol} after {max_retries} retries")


def compute_portfolio_metrics(
    portfolio: List[Dict[str, Any]],
    horizon_days: int = 252,
    benchmark: str = "SPY"
) -> Dict[str, Any]:
    """
    Compute portfolio risk metrics using real market data.
    
    Returns:
    - volatility (annualized)
    - returns
    - max drawdown
    - beta vs benchmark
    - concentration analysis
    """
    symbols = [h["symbol"] for h in portfolio]
    quantities = np.array([h.get("quantity", 1) for h in portfolio], dtype=float)
    
    # Fetch prices for all symbols
    try:
        price_df = pd.DataFrame({
            s: fetch_prices(s, horizon_days) for s in symbols
        }).dropna()
    except Exception as e:
        return {
            "error": f"Failed to fetch market data: {e}",
            "status": "error"
        }
    
    if price_df.empty or len(price_df) < 20:
        return {
            "error": "Insufficient price data",
            "status": "error"
        }
    
    # Calculate daily returns
    returns = price_df.pct_change().dropna()
    
    # Portfolio weights (based on latest prices)
    latest_prices = price_df.iloc[-1].values
    position_values = quantities * latest_prices
    total_value = position_values.sum()
    weights = position_values / total_value
    
    # Portfolio returns
    portfolio_returns = (returns * weights).sum(axis=1)
    
    # Annualized volatility
    portfolio_vol = portfolio_returns.std() * np.sqrt(252)
    
    # Total return
    cumulative_return = (1 + portfolio_returns).prod() - 1
    
    # Max drawdown
    cumulative = (1 + portfolio_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Beta vs benchmark
    beta = None
    try:
        bench_prices = fetch_prices(benchmark.upper(), horizon_days)
        bench_returns = bench_prices.pct_change().dropna()
        
        aligned = pd.DataFrame({
            "portfolio": portfolio_returns,
            "benchmark": bench_returns
        }).dropna()
        
        if len(aligned) > 20:
            cov = aligned.cov().iloc[0, 1]
            var_bench = aligned["benchmark"].var()
            beta = cov / var_bench if var_bench > 0 else None
    except:
        pass
    
    # Concentration analysis
    risk_flags = []
    if weights.max() > 0.4:
        top_holding = symbols[weights.argmax()]
        risk_flags.append(f"High concentration in {top_holding} ({weights.max()*100:.1f}%)")
    
    if portfolio_vol > 0.25:
        risk_flags.append(f"High portfolio volatility ({portfolio_vol*100:.1f}%)")
    
    # Per-position analysis
    position_analysis = []
    for i, symbol in enumerate(symbols):
        pos_vol = returns[symbol].std() * np.sqrt(252)
        pos_return = (price_df[symbol].iloc[-1] / price_df[symbol].iloc[0]) - 1
        position_analysis.append({
            "symbol": symbol,
            "weight": float(weights[i]),
            "volatility": float(pos_vol),
            "return": float(pos_return),
            "value": float(position_values[i])
        })
    
    return {
        "status": "success",
        "total_value": float(total_value),
        "volatility": float(portfolio_vol),
        "total_return": float(cumulative_return),
        "max_drawdown": float(max_drawdown),
        "beta": float(beta) if beta else None,
        "benchmark": benchmark,
        "positions": position_analysis,
        "risk_flags": risk_flags,
        "analysis_period_days": len(portfolio_returns)
    }


# ============================================================
# KNOWLEDGE STORE MODULE (RAG)
# ============================================================

# Simple in-memory knowledge base for demonstration
KNOWLEDGE_BASE = [
    {
        "topic": "asset_allocation",
        "content": "Asset allocation is the strategy of dividing investments among different asset categories like stocks, bonds, and cash. A common rule of thumb is the '100 minus age' rule for stock allocation - a 30-year-old might hold 70% stocks and 30% bonds. However, this should be adjusted based on risk tolerance and investment goals."
    },
    {
        "topic": "risk_tolerance",
        "content": "Risk tolerance refers to an investor's ability and willingness to lose some or all of their investment in exchange for greater potential returns. Conservative investors prefer stable, lower-risk investments. Moderate investors balance growth and stability. Aggressive investors accept higher volatility for potentially higher returns."
    },
    {
        "topic": "diversification",
        "content": "Diversification is a risk management strategy that mixes a variety of investments within a portfolio. The rationale is that a diversified portfolio will, on average, yield higher returns and pose lower risk than any individual investment. Concentration in a single stock (over 20-30% of portfolio) is generally considered high risk."
    },
    {
        "topic": "rebalancing",
        "content": "Portfolio rebalancing involves periodically buying or selling assets to maintain your target asset allocation. As investments grow at different rates, portfolios can drift from their target. Most advisors recommend rebalancing annually or when allocations drift more than 5% from targets."
    },
    {
        "topic": "volatility",
        "content": "Volatility measures how much an investment's price fluctuates over time. Higher volatility means greater price swings and typically higher risk. Annual volatility under 15% is considered low, 15-25% is moderate, and above 25% is high. Tech stocks and growth stocks tend to have higher volatility than value stocks or bonds."
    },
    {
        "topic": "beta",
        "content": "Beta measures an investment's volatility relative to the market (usually S&P 500). A beta of 1 means the investment moves with the market. Beta above 1 indicates higher volatility than the market. Beta below 1 indicates lower volatility. Conservative portfolios typically have beta below 1."
    },
    {
        "topic": "retirement_planning",
        "content": "Retirement planning involves determining retirement income goals and the actions needed to achieve them. Key factors include time horizon, desired retirement lifestyle, and risk tolerance. Generally, younger investors can take more risk, while those closer to retirement should shift toward more conservative allocations."
    },
    {
        "topic": "market_conditions",
        "content": "Market conditions refer to the current state of the stock market, including trends, volatility, and economic indicators. In bull markets, stock prices rise and investors are optimistic. In bear markets, prices fall at least 20% from recent highs. Understanding market conditions helps in making informed investment decisions."
    },
]


def retrieve_knowledge(query: str, top_k: int = 3) -> List[Dict[str, str]]:
    """
    Retrieve relevant knowledge based on the query.
    
    Uses simple keyword matching for demonstration.
    In production, this would use vector embeddings and semantic search.
    """
    query_lower = query.lower()
    
    # Score each knowledge item by keyword overlap
    scored = []
    for item in KNOWLEDGE_BASE:
        score = 0
        content_lower = item["content"].lower()
        topic_lower = item["topic"].lower()
        
        # Check for topic keywords
        topic_words = topic_lower.replace("_", " ").split()
        for word in topic_words:
            if word in query_lower:
                score += 3
        
        # Check for content keywords
        query_words = query_lower.split()
        for word in query_words:
            if len(word) > 3 and word in content_lower:
                score += 1
        
        if score > 0:
            scored.append((score, item))
    
    # Sort by score and return top_k
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:top_k]]


# ============================================================
# ANALYST AGENT
# ============================================================

class AnalystAgent:
    """
    Analyst Agent with access to internet (market data) and knowledge store.
    
    The Analyst:
    - Fetches real-time market data via Alpha Vantage API
    - Computes portfolio metrics (volatility, drawdown, beta)
    - Retrieves relevant financial knowledge
    - Synthesizes findings into a report for the Advisor
    """
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model, temperature=0.3)
    
    def analyze(
        self,
        query: str,
        client_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform analysis for the Advisor.
        
        Args:
            query: The question or task from the Advisor
            client_profile: Client's profile (age, risk tolerance, portfolio)
            
        Returns:
            Analysis results including market data, metrics, and knowledge
        """
        results = {
            "query": query,
            "market_data": None,
            "knowledge": [],
            "synthesis": None
        }
        
        # 1. Compute portfolio metrics (internet access)
        portfolio = client_profile.get("portfolio", [])
        if portfolio:
            metrics = compute_portfolio_metrics(portfolio)
            results["market_data"] = metrics
        
        # 2. Retrieve relevant knowledge
        knowledge_items = retrieve_knowledge(query)
        results["knowledge"] = [item["content"] for item in knowledge_items]
        
        # 3. Synthesize findings with LLM
        results["synthesis"] = self._synthesize(query, client_profile, results)
        
        return results
    
    def _synthesize(
        self,
        query: str,
        client_profile: Dict[str, Any],
        analysis_results: Dict[str, Any]
    ) -> str:
        """Use LLM to synthesize analysis into coherent findings."""
        
        # Build context
        market_summary = "No market data available."
        if analysis_results["market_data"] and analysis_results["market_data"].get("status") == "success":
            md = analysis_results["market_data"]
            market_summary = f"""
Portfolio Analysis:
- Total Value: ${md['total_value']:,.2f}
- Annual Volatility: {md['volatility']*100:.1f}%
- Total Return: {md['total_return']*100:.1f}%
- Max Drawdown: {md['max_drawdown']*100:.1f}%
- Beta vs {md['benchmark']}: {md['beta']:.2f if md['beta'] else 'N/A'}
- Risk Flags: {', '.join(md['risk_flags']) if md['risk_flags'] else 'None'}

Position Details:
""" + "\n".join([
                f"- {p['symbol']}: {p['weight']*100:.1f}% weight, {p['volatility']*100:.1f}% volatility"
                for p in md['positions']
            ])
        
        knowledge_summary = "\n".join([
            f"- {k}" for k in analysis_results["knowledge"]
        ]) if analysis_results["knowledge"] else "No relevant knowledge found."
        
        prompt = f"""You are a financial analyst. Synthesize the following information into a concise report for the Advisor.

Client Profile:
- Age: {client_profile.get('age')}
- Risk Tolerance: {client_profile.get('risk_tolerance')}
- Investment Goals: {client_profile.get('investment_goals', 'General wealth building')}

Query: {query}

Market Data Analysis:
{market_summary}

Relevant Knowledge:
{knowledge_summary}

Provide a concise synthesis (2-3 paragraphs) that:
1. Summarizes the key findings from the data
2. Relates them to the client's profile and goals
3. Highlights any concerns or opportunities

Be factual and reference specific numbers from the analysis."""

        messages = [
            SystemMessage(content="You are a financial analyst providing objective analysis."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        return response.content.strip()


if __name__ == "__main__":
    # Quick test
    from dotenv import load_dotenv
    load_dotenv()
    
    analyst = AnalystAgent()
    
    test_profile = {
        "age": 35,
        "risk_tolerance": "moderate",
        "portfolio": [
            {"symbol": "AAPL", "quantity": 50},
            {"symbol": "MSFT", "quantity": 30},
        ],
        "investment_goals": "Long-term wealth building"
    }
    
    results = analyst.analyze(
        "Is this portfolio too concentrated?",
        test_profile
    )
    
    print("=== Analyst Results ===")
    print(f"Synthesis:\n{results['synthesis']}")
