"""
Client Agent - Simulated client with profile attributes.

The client has a profile (age, risk tolerance, assets/investments) and
can ask questions to the Advisor, react to responses, and indicate
when they are satisfied (conversation resolution).
"""

import json
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

When you feel your question has been adequately answered, express satisfaction 
by saying something like "Thank you, that answers my question" or "That makes sense, I appreciate the advice."
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
        
        # Build conversation context
        conv_text = "\n".join([
            f"{'Client' if turn['role'] == 'client' else 'Advisor'}: {turn['content']}"
            for turn in self.conversation_history
        ])
        
        prompt = f"""The conversation so far:
{conv_text}

Based on the advisor's response, do ONE of the following:
1. If you need clarification or have a follow-up question, ask it naturally
2. If your questions have been adequately answered, express satisfaction and thank the advisor

Remember your profile:
- Age: {self.profile.age}
- Risk tolerance: {self.profile.risk_tolerance}
- You're a real person seeking advice, not a test

Respond naturally as this client would."""

        messages = [
            SystemMessage(content=self._get_system_prompt()),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        reply = response.content.strip()
        
        self.conversation_history.append({"role": "client", "content": reply})
        
        # Detect resolution
        satisfaction_indicators = [
            "thank you", "thanks", "that answers", "makes sense", 
            "appreciate", "helpful", "got it", "understand now",
            "that's helpful", "great advice", "i'll do that"
        ]
        if any(indicator in reply.lower() for indicator in satisfaction_indicators):
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


if __name__ == "__main__":
    # Quick test
    from dotenv import load_dotenv
    load_dotenv()
    
    client = ClientAgent(SAMPLE_PROFILES[0])
    question = client.generate_initial_question()
    print(f"Client: {question}")
