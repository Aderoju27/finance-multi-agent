"""
Streamlit UI for Multi-Agent Financial Advisory System
======================================================

A visual interface for the multi-agent conversation system.

Run with:
    streamlit run app.py
"""

import os
import sys
from pathlib import Path
from datetime import datetime

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=project_root / ".env", override=True)

from src.agents.client_agent import ClientAgent, ClientProfile, SAMPLE_PROFILES, generate_random_profile
from src.agents.advisor_agent import AdvisorAgent


# Page configuration
st.set_page_config(
    page_title="Financial Advisory System",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .client-message {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #2196f3;
    }
    .advisor-message {
        background-color: #f3e5f5;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #9c27b0;
    }
    .analyst-activity {
        background-color: #fff3e0;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        font-size: 0.9em;
        border-left: 4px solid #ff9800;
    }
    .agent-label {
        font-weight: bold;
        margin-bottom: 5px;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


def create_portfolio_chart(portfolio: list) -> go.Figure:
    """Create a pie chart of portfolio holdings."""
    if not portfolio:
        return None
    
    symbols = [h.get("symbol", "Unknown") for h in portfolio]
    quantities = [h.get("quantity", 0) for h in portfolio]
    
    fig = px.pie(
        values=quantities,
        names=symbols,
        title="Portfolio Allocation (by shares)",
        hole=0.4
    )
    fig.update_layout(
        height=300,
        margin=dict(t=40, b=20, l=20, r=20)
    )
    return fig


def display_message(role: str, content: str, turn: int = None):
    """Display a conversation message with styling."""
    if role == "client":
        icon = "🧑"
        label = "Client"
        css_class = "client-message"
    else:
        icon = "👔"
        label = "Advisor"
        css_class = "advisor-message"
    
    turn_label = f" (Turn {turn})" if turn else ""
    
    st.markdown(f"""
    <div class="{css_class}">
        <div class="agent-label">{icon} {label}{turn_label}</div>
        {content}
    </div>
    """, unsafe_allow_html=True)


def display_analyst_activity(message: str):
    """Display analyst activity indicator."""
    st.markdown(f"""
    <div class="analyst-activity">
        📊 <strong>Analyst:</strong> {message}
    </div>
    """, unsafe_allow_html=True)


def run_conversation_step(client: ClientAgent, advisor: AdvisorAgent, 
                          client_message: str, conversation_container):
    """Run a single conversation step and update UI."""
    
    with conversation_container:
        # Show client message
        turn = len(st.session_state.conversation) // 2 + 1
        display_message("client", client_message, turn)
        st.session_state.conversation.append({
            "role": "client", 
            "content": client_message,
            "turn": turn
        })
        
        # Show analyst working
        with st.spinner("📊 Advisor consulting with Analyst..."):
            advisor_response = advisor.process_query(
                client_message,
                client.get_profile_summary()
            )
        
        # Show advisor response
        display_message("advisor", advisor_response, turn)
        st.session_state.conversation.append({
            "role": "advisor",
            "content": advisor_response,
            "turn": turn
        })
        
        return advisor_response


def main():
    """Main Streamlit app."""
    
    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = []
    if "client" not in st.session_state:
        st.session_state.client = None
    if "advisor" not in st.session_state:
        st.session_state.advisor = None
    if "profile" not in st.session_state:
        st.session_state.profile = None
    if "conversation_active" not in st.session_state:
        st.session_state.conversation_active = False
    if "current_turn" not in st.session_state:
        st.session_state.current_turn = 0
    if "conversation_mode" not in st.session_state:
        st.session_state.conversation_mode = "🔄 Automatic"
    
    # Header
    st.title("💼 Multi-Agent Financial Advisory System")
    st.markdown("*JP Morgan Chase Technical Assessment*")
    st.divider()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key Status
        st.subheader("API Status")
        openai_key = os.getenv("OPENAI_API_KEY")
        av_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        
        if openai_key:
            st.success("✅ OpenAI API Key configured")
        else:
            st.error("❌ OpenAI API Key missing")
            st.stop()
        
        if av_key:
            st.success("✅ Alpha Vantage API Key configured")
        else:
            st.warning("⚠️ Alpha Vantage Key missing (no market data)")
        
        st.divider()
        
        # Profile Selection
        st.subheader("Client Profile")
        profile_option = st.radio(
            "Profile Type",
            ["🎲 Generate Random", "📋 Use Preset"],
            index=0
        )
        
        if profile_option == "📋 Use Preset":
            preset_names = [
                f"{p.name} ({p.risk_tolerance}, age {p.age})" 
                for p in SAMPLE_PROFILES
            ]
            selected_preset = st.selectbox(
                "Select Profile",
                preset_names
            )
            preset_index = preset_names.index(selected_preset)
        
        # Conversation Settings
        st.subheader("Settings")
        max_turns = st.slider("Max Turns", 1, 10, 5)
        
        # Conversation Mode
        st.subheader("Conversation Mode")
        conversation_mode = st.radio(
            "Mode",
            ["🔄 Automatic", "👆 Step-by-Step"],
            index=0,
            help="Automatic runs the full conversation. Step-by-Step lets you control each turn."
        )
        
        st.divider()
        
        # Start/Reset buttons
        col1, col2 = st.columns(2)
        
        with col1:
            start_button = st.button(
                "▶️ Start",
                type="primary",
                use_container_width=True,
                disabled=st.session_state.conversation_active
            )
        
        with col2:
            reset_button = st.button(
                "🔄 Reset",
                use_container_width=True
            )
    
    # Handle Reset
    if reset_button:
        st.session_state.conversation = []
        st.session_state.client = None
        st.session_state.advisor = None
        st.session_state.profile = None
        st.session_state.conversation_active = False
        st.session_state.current_turn = 0
        st.rerun()
    
    # Handle Start
    if start_button:
        with st.spinner("Initializing agents..."):
            # Generate or select profile
            if profile_option == "🎲 Generate Random":
                st.session_state.profile = generate_random_profile()
            else:
                st.session_state.profile = SAMPLE_PROFILES[preset_index]
            
            # Initialize agents
            st.session_state.client = ClientAgent(st.session_state.profile)
            st.session_state.advisor = AdvisorAgent()
            st.session_state.conversation_active = True
            st.session_state.current_turn = 0
            st.session_state.conversation = []
            st.session_state.conversation_mode = conversation_mode
        
        st.rerun()
    
    # Main content area
    if st.session_state.profile:
        # Two-column layout
        col_left, col_right = st.columns([2, 1])
        
        with col_right:
            # Profile Card
            st.subheader("📋 Client Profile")
            profile = st.session_state.profile
            
            st.markdown(f"**Name:** {profile.name}")
            st.markdown(f"**Age:** {profile.age}")
            st.markdown(f"**Risk Tolerance:** {profile.risk_tolerance.title()}")
            st.markdown(f"**Goals:** {profile.investment_goals or 'General wealth building'}")
            
            # Portfolio Chart
            st.subheader("📊 Portfolio")
            fig = create_portfolio_chart(profile.portfolio)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Holdings Table
            if profile.portfolio:
                holdings_df = pd.DataFrame(profile.portfolio)
                st.dataframe(
                    holdings_df[["symbol", "quantity"]],
                    hide_index=True,
                    use_container_width=True
                )
            
            # Conversation Stats
            if st.session_state.conversation:
                st.subheader("📈 Stats")
                turns = st.session_state.current_turn
                st.metric("Turns", turns)
                
                is_satisfied = (
                    st.session_state.client and 
                    st.session_state.client.is_satisfied
                )
                status = "✅ Resolved" if is_satisfied else "💬 Active"
                st.metric("Status", status)
        
        with col_left:
            st.subheader("💬 Conversation")
            
            # Conversation container
            conversation_container = st.container()
            
            # Display existing conversation
            with conversation_container:
                for msg in st.session_state.conversation:
                    display_message(msg["role"], msg["content"], msg.get("turn"))
            
            # Run conversation if active
            if st.session_state.conversation_active:
                client = st.session_state.client
                advisor = st.session_state.advisor
                is_automatic = st.session_state.conversation_mode == "🔄 Automatic"
                
                # Check if we need to start or continue
                if st.session_state.current_turn == 0:
                    # Generate initial question
                    with st.spinner("Client formulating question..."):
                        initial_question = client.generate_initial_question()
                    
                    st.session_state.current_turn = 1
                    
                    # Run first exchange
                    with conversation_container:
                        display_message("client", initial_question, 1)
                        st.session_state.conversation.append({
                            "role": "client",
                            "content": initial_question,
                            "turn": 1
                        })
                        
                        with st.spinner("📊 Advisor consulting with Analyst..."):
                            advisor_response = advisor.process_query(
                                initial_question,
                                client.get_profile_summary()
                            )
                        
                        display_message("advisor", advisor_response, 1)
                        st.session_state.conversation.append({
                            "role": "advisor",
                            "content": advisor_response,
                            "turn": 1
                        })
                    
                    st.rerun()
                
                # AUTOMATIC MODE: Continue conversation automatically
                if is_automatic and not client.is_satisfied and st.session_state.current_turn < max_turns:
                    with st.spinner("Continuing conversation..."):
                        last_advisor_msg = st.session_state.conversation[-1]["content"]
                        client_reply = client.respond_to_advisor(last_advisor_msg)
                    
                    st.session_state.current_turn += 1
                    turn = st.session_state.current_turn
                    
                    with conversation_container:
                        display_message("client", client_reply, turn)
                        st.session_state.conversation.append({
                            "role": "client",
                            "content": client_reply,
                            "turn": turn
                        })
                        
                        if not client.is_satisfied:
                            with st.spinner("📊 Advisor consulting with Analyst..."):
                                advisor_response = advisor.process_query(
                                    client_reply,
                                    client.get_profile_summary()
                                )
                            
                            display_message("advisor", advisor_response, turn)
                            st.session_state.conversation.append({
                                "role": "advisor",
                                "content": advisor_response,
                                "turn": turn
                            })
                    
                    st.rerun()
                
                # STEP-BY-STEP MODE: Show continue button
                if not is_automatic and not client.is_satisfied and st.session_state.current_turn < max_turns:
                    if st.button("➡️ Continue Conversation", type="primary"):
                        with st.spinner("Client responding..."):
                            last_advisor_msg = st.session_state.conversation[-1]["content"]
                            client_reply = client.respond_to_advisor(last_advisor_msg)
                        
                        st.session_state.current_turn += 1
                        turn = st.session_state.current_turn
                        
                        with conversation_container:
                            display_message("client", client_reply, turn)
                            st.session_state.conversation.append({
                                "role": "client",
                                "content": client_reply,
                                "turn": turn
                            })
                            
                            if not client.is_satisfied:
                                with st.spinner("📊 Advisor consulting with Analyst..."):
                                    advisor_response = advisor.process_query(
                                        client_reply,
                                        client.get_profile_summary()
                                    )
                                
                                display_message("advisor", advisor_response, turn)
                                st.session_state.conversation.append({
                                    "role": "advisor",
                                    "content": advisor_response,
                                    "turn": turn
                                })
                        
                        st.rerun()
                
                # Show completion status
                if client.is_satisfied:
                    st.success("✅ **Conversation Concluded:** Client is satisfied!")
                    st.session_state.conversation_active = False
                    st.balloons()
                elif st.session_state.current_turn >= max_turns:
                    st.warning(f"⏱️ **Conversation Ended:** Maximum turns ({max_turns}) reached")
                    st.session_state.conversation_active = False
    
    else:
        # Welcome screen
        st.markdown("""
        ## Welcome! 👋
        
        This system demonstrates multi-agent collaboration in a financial advisory context.
        
        ### 🤖 Agents
        
        | Agent | Role |
        |-------|------|
        | **Client** | Simulated client with profile, asks questions |
        | **Advisor** | Orchestrates conversation, talks to client |
        | **Analyst** | Fetches market data, retrieves knowledge |
        
        ### 🚀 Getting Started
        
        1. Choose a **profile type** in the sidebar (random or preset)
        2. Adjust **max turns** if desired
        3. Click **Start** to begin the conversation
        4. Watch the agents collaborate in real-time!
        
        ---
        *Use the sidebar to configure and start a conversation.*
        """)


if __name__ == "__main__":
    main()
