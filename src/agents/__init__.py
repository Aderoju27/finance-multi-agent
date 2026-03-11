"""
Multi-Agent Financial Advisory System - Agents Package

This package contains the three agents:
- ClientAgent: Simulated client with profile
- AdvisorAgent: Orchestrator that talks to client
- AnalystAgent: Research agent with internet + knowledge access
"""

from src.agents.client_agent import ClientAgent, ClientProfile, SAMPLE_PROFILES
from src.agents.advisor_agent import AdvisorAgent
from src.agents.analyst_agent import AnalystAgent

__all__ = [
    "ClientAgent",
    "ClientProfile", 
    "SAMPLE_PROFILES",
    "AdvisorAgent",
    "AnalystAgent",
]
