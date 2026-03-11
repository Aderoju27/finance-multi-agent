"""
Client Agent - Simulated client with profile attributes.

The client has a profile (age, risk tolerance, assets/investments) and
can ask questions to the Advisor, react to responses, and indicate
when they are satisfied (conversation resolution).
"""

import json
import random
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


class ClientProfile(BaseModel):
    """Client profile with financial attributes."""
    name: str = Field(description="Client's name")
    age: int = Field(description="Client's age")
    risk_tolerance: str = Field(description="conservative, moderate, or aggressive")
    portfolio: List[Dict[str, Any]] = Field(description="List of holdings with symbol and quantity")
    investment_goals: Optional[str] = Field(default=None, description="Client's investment goals")
    
    def total_portfolio_description(self) -> str:
        """Generate a text description of the portfolio."""
        if not self.portfolio:
            return "No current investments"
        
        holdings = []
        for h in self.portfolio:
            symbol = h.get("symbol", "Unknown")
            quantity = h.get("quantity", 0)
            holdings.append(f"{quantity} shares of {symbol}")
        return ", ".join(holdings)


class ClientAgent:
    """
    Simulated client that interacts with the Advisor.
    
    The client:
    - Has a profile (age, risk tolerance, portfolio)
    - Generates questions based on their profile
    - Responds to advisor's answers (follow-up or satisfaction)
    - Indicates when the conversation should conclude
    """
    
    def __init__(self, profile: ClientProfile, model: str = "gpt-4o-mini"):
        self.profile = profile
        self.conversation_history: List[Dict[str, str]] = []
        self.is_satisfied = False
        self.llm = ChatOpenAI(model=model, temperature=0.7)
    
    def _get_system_prompt(self) -> str:
        """System prompt for the client persona."""
        return f"""You are role-playing as a financial advisory client with the following profile:

Name: {self.profile.name}
Age: {self.profile.age}
Risk Tolerance: {self.profile.risk_tolerance}
Current Portfolio: {self.profile.total_portfolio_description()}
Investment Goals: {self.profile.investment_goals or 'General wealth building'}

Stay in character as this client. You are seeking financial guidance from an advisor.
Be natural and conversational. Ask questions that someone with your profile would ask.
Express concerns appropriate to your risk tolerance level.

IMPORTANT BEHAVIOR:
- You are a thorough client who likes to understand things fully before making decisions
- Don't be satisfied with the first answer - ask follow-up questions to dig deeper
- Ask about specific recommendations, risks, implementation steps, or alternatives
- After 2-3 exchanges with good detailed answers, you can express satisfaction

Only express FINAL satisfaction (ending the conversation) when you have:
1. Received specific, actionable advice
2. Understood the risks and tradeoffs
3. Have a clear next step or plan

When truly satisfied, say something definitive like "That answers all my questions, thank you!"
"""
    
    def generate_initial_question(self) -> str:
        """Generate the opening question based on the client's profile."""
        prompt = f"""Based on your profile, ask the financial advisor ONE specific question 
about your investments or financial situation. 

Consider your age ({self.profile.age}), risk tolerance ({self.profile.risk_tolerance}), 
and current holdings ({self.profile.total_portfolio_description()}).

Ask something relevant to your situation - it could be about:
- Whether your portfolio allocation is appropriate
- If you should rebalance
- Concerns about concentration risk
- Questions about market conditions affecting your holdings
- Retirement planning considerations

Respond with just the question, nothing else."""

        messages = [
            SystemMessage(content=self._get_system_prompt()),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        question = response.content.strip()
        
        self.conversation_history.append({"role": "client", "content": question})
        return question
    
    def respond_to_advisor(self, advisor_response: str) -> str:
        """
        React to the advisor's response.
        
        Either ask a follow-up question or indicate satisfaction.
        """
        self.conversation_history.append({"role": "advisor", "content": advisor_response})
        
        # Count conversation turns (each client message is a turn)
        turn_count = len([m for m in self.conversation_history if m["role"] == "client"])
        
        # Build conversation context
        conv_text = "\n".join([
            f"{'Client' if turn['role'] == 'client' else 'Advisor'}: {turn['content']}"
            for turn in self.conversation_history
        ])
        
        # Adjust behavior based on conversation progress
        if turn_count <= 1:
            turn_guidance = """This is early in the conversation. You likely have follow-up questions about:
- Specific investment recommendations or fund names
- How to implement the advice practically
- Potential risks or downsides
- Timeline or next steps

Ask a thoughtful follow-up question to dig deeper."""
        elif turn_count == 2:
            turn_guidance = """You've had a couple exchanges. Consider whether:
- You have one more clarifying question, OR
- The advisor has given you enough actionable information to feel satisfied

If you have lingering concerns, ask them. If you feel informed, express satisfaction."""
        else:
            turn_guidance = """You've had several exchanges and received detailed advice. 
Unless you have a genuinely new concern, it's natural to express satisfaction and thank the advisor.
Say something like "That answers all my questions, thank you for the thorough explanation!" """
        
        prompt = f"""The conversation so far:
{conv_text}

Turn {turn_count + 1} - {turn_guidance}

Remember your profile:
- Age: {self.profile.age}
- Risk tolerance: {self.profile.risk_tolerance}
- Goals: {self.profile.investment_goals}

Respond naturally as this client. Either ask a genuine follow-up OR express clear satisfaction."""

        messages = [
            SystemMessage(content=self._get_system_prompt()),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        reply = response.content.strip()
        
        self.conversation_history.append({"role": "client", "content": reply})
        
        # Detect resolution
        reply_lower = reply.lower()
        
        # Check if reply contains a question (likely a follow-up)
        has_question = "?" in reply
        has_followup_phrases = any(phrase in reply_lower for phrase in [
            "can you", "could you", "what about", "how do", "how should",
            "what would", "should i", "do you recommend", "what if",
            "tell me more", "explain", "elaborate", "wondering"
        ])
        
        # Satisfaction indicators - be more inclusive
        satisfaction_indicators = [
            "thank you", "thanks", "that answers", "that's all",
            "i have no more questions", "very helpful", "great advice",
            "i feel", "i'm satisfied", "nothing else", "covers everything",
            "i understand", "makes sense", "i appreciate", "helpful",
            "i'll do that", "i'll consider", "i'll look into"
        ]
        
        # Mark satisfied if showing gratitude without asking more questions
        is_satisfied = (
            any(indicator in reply_lower for indicator in satisfaction_indicators)
            and not has_question
            and not has_followup_phrases
        )
        
        if is_satisfied:
            self.is_satisfied = True
        
        return reply
    
    def get_profile_summary(self) -> Dict[str, Any]:
        """Return profile as dict for the Advisor to reference."""
        return {
            "name": self.profile.name,
            "age": self.profile.age,
            "risk_tolerance": self.profile.risk_tolerance,
            "portfolio": self.profile.portfolio,
            "investment_goals": self.profile.investment_goals
        }


# Sample client profiles for demonstration
SAMPLE_PROFILES = [
    ClientProfile(
        name="Sarah Chen",
        age=35,
        risk_tolerance="moderate",
        portfolio=[
            {"symbol": "AAPL", "quantity": 50, "purchase_price": 150},
            {"symbol": "MSFT", "quantity": 30, "purchase_price": 280},
            {"symbol": "VTI", "quantity": 100, "purchase_price": 200},
        ],
        investment_goals="Save for retirement and build long-term wealth"
    ),
    ClientProfile(
        name="Robert Martinez",
        age=58,
        risk_tolerance="conservative",
        portfolio=[
            {"symbol": "BND", "quantity": 200, "purchase_price": 75},
            {"symbol": "VYM", "quantity": 150, "purchase_price": 110},
            {"symbol": "AAPL", "quantity": 25, "purchase_price": 140},
        ],
        investment_goals="Preserve capital and generate income for retirement in 7 years"
    ),
    ClientProfile(
        name="Alex Thompson",
        age=28,
        risk_tolerance="aggressive",
        portfolio=[
            {"symbol": "NVDA", "quantity": 40, "purchase_price": 450},
            {"symbol": "TSLA", "quantity": 30, "purchase_price": 250},
            {"symbol": "ARKK", "quantity": 100, "purchase_price": 45},
        ],
        investment_goals="Maximize growth over the next 20+ years"
    ),
]


def generate_random_profile(model: str = "gpt-4o-mini") -> ClientProfile:
    """
    Use LLM to generate a random, realistic client profile.
    
    Returns:
        A randomly generated ClientProfile
    """
    llm = ChatOpenAI(model=model, temperature=1.0)
    
    # Common stock symbols for variety
    stock_symbols = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B",
        "JPM", "V", "JNJ", "WMT", "PG", "MA", "HD", "CVX", "MRK", "ABBV",
        "PFE", "KO", "PEP", "COST", "TMO", "AVGO", "MCD", "CSCO", "ACN",
        "VTI", "VOO", "SPY", "QQQ", "VEA", "VWO", "BND", "AGG", "VYM",
        "SCHD", "ARKK", "ARKG", "VNQ", "GLD", "TLT", "IWM", "DIA"
    ]
    
    prompt = """Generate a realistic financial advisory client profile. Be creative and diverse.

Return ONLY a valid JSON object with this exact structure (no markdown, no explanation):
{
    "name": "Full Name",
    "age": <number between 22 and 75>,
    "risk_tolerance": "<conservative OR moderate OR aggressive>",
    "portfolio": [
        {"symbol": "<stock ticker>", "quantity": <number>, "purchase_price": <number>},
        {"symbol": "<stock ticker>", "quantity": <number>, "purchase_price": <number>},
        {"symbol": "<stock ticker>", "quantity": <number>, "purchase_price": <number>}
    ],
    "investment_goals": "<specific investment goals based on age and situation>"
}

Guidelines:
- Create diverse names representing different backgrounds
- Age should influence risk tolerance and goals realistically
- Conservative clients: prefer bonds (BND, AGG), dividend stocks (VYM, SCHD), stable companies
- Moderate clients: balanced mix of growth and stability (VTI, VOO, blue chips)
- Aggressive clients: growth stocks (NVDA, TSLA, ARKK), tech-heavy portfolios
- Portfolio should have 2-5 holdings with realistic quantities (10-500 shares)
- Goals should be specific and match age/risk profile

Return ONLY the JSON object."""

    messages = [
        SystemMessage(content="You are a data generator. Output only valid JSON, nothing else."),
        HumanMessage(content=prompt)
    ]
    
    try:
        response = llm.invoke(messages)
        content = response.content.strip()
        
        # Clean up response if it has markdown code blocks
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        
        data = json.loads(content)
        
        return ClientProfile(
            name=data["name"],
            age=data["age"],
            risk_tolerance=data["risk_tolerance"],
            portfolio=data["portfolio"],
            investment_goals=data.get("investment_goals")
        )
    except Exception as e:
        print(f"Warning: Failed to generate random profile ({e}), using fallback")
        # Fallback to a randomly selected sample profile
        return random.choice(SAMPLE_PROFILES)


if __name__ == "__main__":
    # Quick test
    from dotenv import load_dotenv
    load_dotenv()
    
    client = ClientAgent(SAMPLE_PROFILES[0])
    question = client.generate_initial_question()
    print(f"Client: {question}")
