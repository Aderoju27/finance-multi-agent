"""
Multi-Agent Financial Advisory System
=====================================

A multi-agent system where agents collaborate to provide financial
investment guidance. The system includes:

- Client Agent: Simulated client with profile (age, risk tolerance, portfolio)
- Advisor Agent: Orchestrates conversation, delegates to Analyst
- Analyst Agent: Fetches market data (internet) and retrieves knowledge

Usage:
    python src/main.py
    
    # Or with a specific profile index (0, 1, or 2)
    python src/main.py --profile 1
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

from src.agents.client_agent import ClientAgent, ClientProfile, SAMPLE_PROFILES
from src.agents.advisor_agent import AdvisorAgent


def run_conversation(
    client_profile: ClientProfile,
    max_turns: int = 5,
    verbose: bool = True
) -> dict:
    """
    Run a multi-agent conversation between Client and Advisor.
    
    Args:
        client_profile: The client's profile
        max_turns: Maximum conversation turns before forced stop
        verbose: Whether to print conversation in real-time
        
    Returns:
        Conversation summary dict
    """
    # Initialize agents
    client = ClientAgent(client_profile)
    advisor = AdvisorAgent()
    
    conversation_log = []
    start_time = datetime.now()
    
    if verbose:
        print("\n" + "="*60)
        print("MULTI-AGENT FINANCIAL ADVISORY CONVERSATION")
        print("="*60)
        print(f"\nClient Profile:")
        print(f"  Name: {client_profile.name}")
        print(f"  Age: {client_profile.age}")
        print(f"  Risk Tolerance: {client_profile.risk_tolerance}")
        print(f"  Portfolio: {client_profile.total_portfolio_description()}")
        print(f"  Goals: {client_profile.investment_goals}")
        print("\n" + "-"*60 + "\n")
    
    # Client initiates conversation
    client_question = client.generate_initial_question()
    conversation_log.append({"role": "client", "content": client_question})
    
    if verbose:
        print(f"🧑 Client: {client_question}\n")
    
    # Conversation loop
    for turn in range(max_turns):
        # Advisor processes query (delegates to Analyst internally)
        if verbose:
            print("   [Advisor consulting with Analyst...]")
        
        advisor_response = advisor.process_query(
            client_question,
            client.get_profile_summary()
        )
        conversation_log.append({"role": "advisor", "content": advisor_response})
        
        if verbose:
            print(f"\n👔 Advisor: {advisor_response}\n")
        
        # Client responds
        client_reply = client.respond_to_advisor(advisor_response)
        conversation_log.append({"role": "client", "content": client_reply})
        
        if verbose:
            print(f"🧑 Client: {client_reply}\n")
        
        # Check for resolution
        if client.is_satisfied:
            if verbose:
                print("-"*60)
                print("✅ CONVERSATION CONCLUDED: Resolution reached")
                print("-"*60)
            break
        
        # Prepare for next turn
        client_question = client_reply
    else:
        # Max turns reached
        if verbose:
            print("-"*60)
            print("⏱️ CONVERSATION ENDED: Maximum turns reached")
            print("-"*60)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    return {
        "client_name": client_profile.name,
        "turns": len(conversation_log) // 2,
        "resolved": client.is_satisfied,
        "duration_seconds": duration,
        "conversation": conversation_log
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Financial Advisory System"
    )
    parser.add_argument(
        "--profile",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Client profile to use (0=Sarah/moderate, 1=Robert/conservative, 2=Alex/aggressive)"
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=5,
        help="Maximum conversation turns"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress real-time output"
    )
    
    args = parser.parse_args()
    
    # Load environment variables
    env_path = project_root / ".env"
    load_dotenv(dotenv_path=env_path)
    
    # Verify API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set in environment")
        print("Please copy .env.example to .env and add your API keys")
        sys.exit(1)
    
    if not os.getenv("ALPHA_VANTAGE_API_KEY"):
        print("WARNING: ALPHA_VANTAGE_API_KEY not set - market data will be unavailable")
    
    # Select client profile
    profile = SAMPLE_PROFILES[args.profile]
    
    # Run conversation
    result = run_conversation(
        client_profile=profile,
        max_turns=args.max_turns,
        verbose=not args.quiet
    )
    
    # Print summary
    print("\n" + "="*60)
    print("CONVERSATION SUMMARY")
    print("="*60)
    print(f"Client: {result['client_name']}")
    print(f"Turns: {result['turns']}")
    print(f"Resolved: {'Yes' if result['resolved'] else 'No'}")
    print(f"Duration: {result['duration_seconds']:.1f} seconds")
    print("="*60)


if __name__ == "__main__":
    main()
