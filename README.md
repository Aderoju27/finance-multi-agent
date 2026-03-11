# Multi-Agent Financial Advisory System

A multi-agent system where agents collaborate to provide financial investment guidance to a client.

## Overview

This system demonstrates multi-agent collaboration in a financial advisory context. Three agents work together to achieve a common goal: providing tailored financial guidance to a simulated client.

### Agents

| Agent | Role | Capabilities |
|-------|------|--------------|
| **Client Agent** | Simulated client with profile | Has age, risk tolerance, portfolio; asks questions; indicates satisfaction |
| **Advisor Agent** | Orchestrator | Sole agent that talks to client; delegates research to Analyst; formulates responses |
| **Analyst Agent** | Research & Analysis | Internet access (Alpha Vantage API); Knowledge store; Portfolio metrics computation |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     CLIENT AGENT                             │
│  Profile: age, risk_tolerance, portfolio, goals              │
│  Actions: ask questions, respond, indicate satisfaction      │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          │ "Is my portfolio too concentrated?"
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                     ADVISOR AGENT                            │
│  - Receives client questions                                 │
│  - Delegates research to Analyst                            │
│  - Synthesizes response for client                          │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          │ "Analyze this portfolio for concentration risk"
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                     ANALYST AGENT                            │
│                                                              │
│  ┌─────────────────┐    ┌─────────────────────────────────┐ │
│  │ Internet Access │    │ Knowledge Store                 │ │
│  │ (Alpha Vantage) │    │ (Financial concepts)            │ │
│  │                 │    │                                 │ │
│  │ • Stock prices  │    │ • Asset allocation              │ │
│  │ • Volatility    │    │ • Risk tolerance                │ │
│  │ • Beta          │    │ • Diversification               │ │
│  │ • Returns       │    │ • Rebalancing                   │ │
│  └─────────────────┘    └─────────────────────────────────┘ │
│                                                              │
│  Output: Synthesis of market data + relevant knowledge       │
└─────────────────────────────────────────────────────────────┘
```

## Conversation Flow

```
1. Client generates initial question based on profile
   └─→ "I'm 35 with moderate risk tolerance. Is my portfolio too concentrated?"

2. Advisor receives question and delegates to Analyst
   └─→ "Analyze portfolio for concentration risk"

3. Analyst performs research:
   ├─→ Fetches market data (Alpha Vantage API)
   ├─→ Computes portfolio metrics (volatility, beta, concentration)
   ├─→ Retrieves relevant knowledge (diversification, risk)
   └─→ Synthesizes findings

4. Advisor formulates client-friendly response
   └─→ "Your portfolio is 62% in AAPL, which exceeds typical concentration..."

5. Client responds (follow-up or satisfaction)
   └─→ "What should I do about that?" OR "Thank you, that answers my question"

6. Loop continues until client is satisfied
   └─→ "✅ CONVERSATION CONCLUDED: Resolution reached"
```

## Quick Start

```bash
# Clone repository
git clone https://github.com/Aderoju27/jpmorgan-multi-agent.git
cd jpmorgan-multi-agent

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your OPENAI_API_KEY and ALPHA_VANTAGE_API_KEY

# Run the system (CLI)
python src/main.py

# Or run with Streamlit UI
streamlit run app.py
```

## Usage Options

### Command Line Interface

```bash
# Use preset profile
python src/main.py --profile 0  # Sarah Chen (moderate risk)
python src/main.py --profile 1  # Robert Martinez (conservative)
python src/main.py --profile 2  # Alex Thompson (aggressive)

# Generate random client profile
python src/main.py --random

# Adjust conversation length
python src/main.py --max-turns 10
```

### Streamlit Web UI

```bash
streamlit run app.py
```

Features:
- 🎲 Random or preset client profile selection
- 📊 Portfolio visualization with interactive charts
- 💬 Real-time conversation display with agent labels
- ⚙️ Configurable settings (max turns)
- 📈 Live conversation stats

### Command Line Options

```bash
# Use different client profiles (0=moderate, 1=conservative, 2=aggressive)
python src/main.py --profile 1

# Set maximum conversation turns
python src/main.py --max-turns 3

# Quiet mode (summary only)
python src/main.py --quiet
```

## Sample Output

```
============================================================
MULTI-AGENT FINANCIAL ADVISORY CONVERSATION
============================================================

Client Profile:
  Name: Sarah Chen
  Age: 35
  Risk Tolerance: moderate
  Portfolio: 50 shares of AAPL, 30 shares of MSFT, 100 shares of VTI
  Goals: Save for retirement and build long-term wealth

------------------------------------------------------------

🧑 Client: Given my moderate risk tolerance and current portfolio allocation, 
   should I be concerned about my concentration in technology stocks?

   [Advisor consulting with Analyst...]

👔 Advisor: Looking at your portfolio, you currently have about 45% of your 
   holdings in tech stocks (AAPL and MSFT), which is on the higher end for 
   a moderate risk investor. Your portfolio volatility is around 22%, which 
   is moderate but could spike during tech sector downturns.

   Given your age of 35 and long-term goals, you have time to weather 
   volatility, but for a moderate risk tolerance, I'd suggest considering 
   some diversification. Your VTI position provides broad market exposure, 
   which is good. You might consider adding some bond exposure or 
   international stocks to reduce sector concentration.

🧑 Client: That makes sense. What specific changes would you recommend to 
   better align my portfolio with my moderate risk tolerance?

   [Advisor consulting with Analyst...]

👔 Advisor: Based on your profile, I'd suggest rebalancing to roughly:
   - 50-60% US stocks (keep VTI, reduce individual tech)
   - 20-25% international (consider VXUS)
   - 15-20% bonds (consider BND)
   
   This would reduce your portfolio volatility to around 15% while still 
   providing growth potential for your retirement goals.

🧑 Client: Thank you, that's very helpful advice. I appreciate the specific 
   recommendations.

------------------------------------------------------------
✅ CONVERSATION CONCLUDED: Resolution reached
------------------------------------------------------------

============================================================
CONVERSATION SUMMARY
============================================================
Client: Sarah Chen
Turns: 2
Resolved: Yes
Duration: 45.3 seconds
============================================================
```

## Sample Client Profiles

| Profile | Name | Age | Risk Tolerance | Portfolio |
|---------|------|-----|----------------|-----------|
| 0 | Sarah Chen | 35 | Moderate | AAPL, MSFT, VTI |
| 1 | Robert Martinez | 58 | Conservative | BND, VYM, AAPL |
| 2 | Alex Thompson | 28 | Aggressive | NVDA, TSLA, ARKK |

## Technical Details

### Internet Access (Alpha Vantage API)

The Analyst Agent fetches real market data:
- Daily stock prices
- Historical returns
- Used for volatility, beta, and drawdown calculations

Features:
- Rate limiting (1.5s between calls)
- Response caching (1 hour TTL)
- Exponential backoff on failures

### Knowledge Store

Simple keyword-based retrieval covering:
- Asset allocation strategies
- Risk tolerance definitions
- Diversification principles
- Rebalancing guidelines
- Market conditions
- Retirement planning

### Portfolio Metrics Computed

| Metric | Description |
|--------|-------------|
| Volatility | Annualized standard deviation of returns |
| Max Drawdown | Largest peak-to-trough decline |
| Beta | Correlation with SPY benchmark |
| Concentration | Weight per position |
| Total Return | Cumulative return over analysis period |

## Project Structure

```
jpmorgan-multi-agent/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
└── src/
    ├── main.py              # Entry point
    └── agents/
        ├── client_agent.py  # Client with profile
        ├── advisor_agent.py # Orchestrator
        └── analyst_agent.py # Market data + knowledge
```

## Production Considerations

For a production deployment, I would add:

- **Idempotency** - Prevent duplicate processing of requests
- **Exponential backoff** - Handle API failures gracefully
- **Audit logging** - Track all interactions for compliance
- **Rate limiting** - Prevent abuse
- **Database persistence** - Store conversation history

These patterns are implemented in my production-grade [Finnie AI](https://github.com/Aderoju27/finnie-ai) project.

## Author

Tunde Aderoju

## License

MIT
