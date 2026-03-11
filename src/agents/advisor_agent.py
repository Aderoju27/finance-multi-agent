"""
Advisor Agent - Orchestrates communication between Client and Analyst.

The Advisor is the sole agent permitted to interact with the Client.
It receives questions from the Client, delegates research tasks to the
Analyst, and formulates responses for the Client.
"""

import json
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.analyst_agent import AnalystAgent


class AdvisorAgent:
    """
    Advisor Agent - The orchestrator of the multi-agent system.
    
    The Advisor:
    - Is the sole agent that interacts with the Client
    - Delegates research tasks to the Analyst
    - Synthesizes Analyst findings into client-friendly responses
    - Maintains conversation context
    """
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model, temperature=0.5)
        self.analyst = AnalystAgent(model=model)
        self.conversation_history: List[Dict[str, str]] = []
    
    def _get_system_prompt(self, client_profile: Dict[str, Any]) -> str:
        """System prompt for the Advisor persona."""
        return f"""You are a professional financial advisor speaking with a client.

Client Profile:
- Name: {client_profile.get('name', 'Client')}
- Age: {client_profile.get('age', 'Unknown')}
- Risk Tolerance: {client_profile.get('risk_tolerance', 'Unknown')}
- Investment Goals: {client_profile.get('investment_goals', 'General wealth building')}

Your role:
1. Listen carefully to the client's questions and concerns
2. Use analysis from your research team (Analyst) to inform your responses
3. Communicate clearly and professionally, avoiding jargon
4. Tailor advice to the client's risk tolerance and goals
5. Be honest about limitations and risks

Remember: You are providing educational guidance, not personalized investment advice.
Always be clear, empathetic, and professional."""

    def process_query(
        self,
        client_query: str,
        client_profile: Dict[str, Any]
    ) -> str:
        """
        Process a query from the Client.
        
        1. Log the query
        2. Delegate to Analyst for research
        3. Formulate response for Client
        """
        self.conversation_history.append({
            "role": "client",
            "content": client_query
        })
        
        # Step 1: Determine what research is needed
        research_task = self._formulate_research_task(client_query, client_profile)
        
        # Step 2: Delegate to Analyst
        analyst_findings = self.analyst.analyze(research_task, client_profile)
        
        # Step 3: Formulate client-facing response
        response = self._formulate_response(
            client_query,
            client_profile,
            analyst_findings
        )
        
        self.conversation_history.append({
            "role": "advisor",
            "content": response
        })
        
        return response
    
    def _formulate_research_task(
        self,
        client_query: str,
        client_profile: Dict[str, Any]
    ) -> str:
        """Determine what research the Analyst should perform."""
        # For now, pass through the query with context
        # In a more complex system, this could identify specific data needs
        return f"""Client ({client_profile.get('risk_tolerance', 'unknown')} risk tolerance, age {client_profile.get('age', 'unknown')}) asks: {client_query}

Please analyze their portfolio and provide relevant information."""

    def _formulate_response(
        self,
        client_query: str,
        client_profile: Dict[str, Any],
        analyst_findings: Dict[str, Any]
    ) -> str:
        """Formulate a client-friendly response using Analyst findings."""
        
        # Extract key information from analyst
        synthesis = analyst_findings.get("synthesis", "")
        market_data = analyst_findings.get("market_data", {})
        
        # Build context for LLM
        data_context = ""
        if market_data and market_data.get("status") == "success":
            data_context = f"""
Key Portfolio Metrics:
- Total Value: ${market_data['total_value']:,.2f}
- Volatility: {market_data['volatility']*100:.1f}%
- Max Drawdown: {market_data['max_drawdown']*100:.1f}%
- Beta: {market_data['beta']:.2f if market_data['beta'] else 'N/A'}
- Risk Flags: {', '.join(market_data['risk_flags']) if market_data['risk_flags'] else 'None'}
"""
        
        prompt = f"""The client asked: "{client_query}"

Your research team (Analyst) has provided the following analysis:

{synthesis}

{data_context}

Now, respond to the client directly. Your response should:
1. Address their specific question
2. Reference specific data points where relevant
3. Be clear and jargon-free
4. Consider their risk tolerance ({client_profile.get('risk_tolerance', 'unknown')}) and age ({client_profile.get('age', 'unknown')})
5. Be 2-3 paragraphs maximum

Speak directly to the client in a warm, professional tone."""

        messages = [
            SystemMessage(content=self._get_system_prompt(client_profile)),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        return response.content.strip()
    
    def get_conversation_summary(self) -> str:
        """Return a summary of the conversation."""
        if not self.conversation_history:
            return "No conversation yet."
        
        summary_lines = []
        for turn in self.conversation_history:
            role = "Client" if turn["role"] == "client" else "Advisor"
            content = turn["content"][:100] + "..." if len(turn["content"]) > 100 else turn["content"]
            summary_lines.append(f"{role}: {content}")
        
        return "\n".join(summary_lines)


if __name__ == "__main__":
    # Quick test
    from dotenv import load_dotenv
    load_dotenv()
    
    advisor = AdvisorAgent()
    
    test_profile = {
        "name": "Sarah Chen",
        "age": 35,
        "risk_tolerance": "moderate",
        "portfolio": [
            {"symbol": "AAPL", "quantity": 50},
            {"symbol": "MSFT", "quantity": 30},
        ],
        "investment_goals": "Long-term wealth building"
    }
    
    response = advisor.process_query(
        "Is my portfolio too concentrated in tech stocks?",
        test_profile
    )
    
    print("=== Advisor Response ===")
    print(response)
