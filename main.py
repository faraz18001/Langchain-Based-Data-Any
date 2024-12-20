import streamlit as st
import os
import pandas as pd
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_experimental.agents.agent_toolkits.pandas.base import _get_functions_single_prompt
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain_openai import ChatOpenAI
import tiktoken
from langchain.prompts import MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables.history import RunnableWithMessageHistory
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide"
)

# Add custom CSS with enhanced UI styling
st.markdown("""
    <style>
    /* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600&family=Space+Grotesk:wght@400;500;600&display=swap');

/* General styles */
.stApp {
    font-family: 'Space Grotesk', sans-serif;
    background: linear-gradient(135deg, #f5f7ff 0%, #ffffff 100%);
}

/* Header styling */
.main-header {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    padding: 2.5rem;
    border-radius: 20px;
    margin-bottom: 2rem;
    border: none;
    box-shadow: 0 10px 20px rgba(99, 102, 241, 0.1);
}

.main-header h1 {
    color: #ffffff !important;
    font-family: 'Poppins', sans-serif;
    font-weight: 600;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
}

.main-header p {
    color: rgba(255, 255, 255, 0.9) !important;
}

/* Chat container */
.stChat {
    padding: 25px;
    background: rgba(255, 255, 255, 0.9);
    border-radius: 20px;
    box-shadow: 0 8px 32px rgba(99, 102, 241, 0.1);
}

/* Message bubbles */
.chat-bubble {
    padding: 15px 22px;
    border-radius: 18px;
    margin: 12px 0;
    max-width: 85%;
    line-height: 1.6;
    font-size: 1.05rem;
}

.user-message {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    color: white;
    border: none;
    margin-left: auto;
}

.assistant-message {
    background: linear-gradient(135deg, #f5f7ff 0%, #ffffff 100%);
    border: 2px solid #e1e5ff;
}

/* Metrics container */
.metrics-container {
    background: rgba(255, 255, 255, 0.9);
    padding: 25px;
    border-radius: 20px;
    margin: 20px 0;
    border: 2px solid #e1e5ff;
    box-shadow: 0 8px 32px rgba(99, 102, 241, 0.1);
}

/* File uploader */
.uploadedFile {
    border: 3px dashed #6366f1;
    border-radius: 20px;
    padding: 25px;
    text-align: center;
    background: rgba(255, 255, 255, 0.9);
    transition: all 0.3s ease;
}

.uploadedFile:hover {
    border-color: #8b5cf6;
    background: #f5f7ff;
    transform: translateY(-2px);
}

/* Dataset info box */
.dataset-info {
    background: linear-gradient(135deg, #f5f7ff 0%, #ffffff 100%);
    padding: 20px;
    border-radius: 20px;
    margin: 15px 0;
    border: 2px solid #e1e5ff;
}

/* Thinking animation */
.thinking {
    display: inline-block;
    padding: 15px 25px;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    border-radius: 25px;
    margin: 12px 0;
    animation: gentle-pulse 1.5s infinite;
    border: none;
    color: white;
    font-weight: 500;
}

@keyframes gentle-pulse {
    0% { opacity: 0.8; transform: scale(0.98); }
    50% { opacity: 1; transform: scale(1.02); }
    100% { opacity: 0.8; transform: scale(0.98); }
}

/* Buttons */
.stButton button {
    border-radius: 25px;
    padding: 6px 24px;
    font-weight: 600;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    color: white;
    border: none;
    transition: all 0.3s ease;
}

.stButton button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(99, 102, 241, 0.2);
}

/* Text inputs */
.stTextInput input {
    border-radius: 25px;
    border: 2px solid #0000FF;
    padding: 12px 20px;
    font-family: 'Space Grotesk', sans-serif;
    transition: all 0.3s ease;
}

.stTextInput input:focus {
    border-color: #6366f1;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
}
    </style>
    """, unsafe_allow_html=True)

# Model pricing per 1000 tokens (as of 2024)
MODEL_PRICING = {
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002}
}

# Create a directory to store visualizations
VISUALIZATION_DIR = "visualizations"
if not os.path.exists(VISUALIZATION_DIR):
    os.makedirs(VISUALIZATION_DIR)

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0
if "total_cost" not in st.session_state:
    st.session_state.total_cost = 0.0
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "gpt-4"
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "df" not in st.session_state:
    st.session_state.df = None

class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    messages: List[BaseMessage] = Field(default_factory=list)

    def add_message(self, message: BaseMessage) -> None:
        self.messages.append(message)

    def clear(self) -> None:
        self.messages = []

    class Config:
        arbitrary_types_allowed = True

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]

def calculate_cost(tokens: int, model: str, is_input: bool = True) -> float:
    """Calculate cost based on tokens and model pricing"""
    token_type = "input" if is_input else "output"
    cost_per_1k = MODEL_PRICING[model][token_type]
    return (tokens / 1000) * cost_per_1k

def update_token_metrics(input_text: str, output_text: str) -> None:
    """Update token count and cost metrics"""
    encoding = tiktoken.get_encoding('cl100k_base')
    input_tokens = len(encoding.encode(input_text))
    output_tokens = len(encoding.encode(output_text))
    
    # Calculate costs
    input_cost = calculate_cost(input_tokens, st.session_state.selected_model, True)
    output_cost = calculate_cost(output_tokens, st.session_state.selected_model, False)
    
    # Update session state
    st.session_state.total_tokens += input_tokens + output_tokens
    st.session_state.total_cost += input_cost + output_cost

def display_metrics() -> None:
    """Display token usage and cost metrics in the sidebar"""
    st.sidebar.title("Session Metrics")
    
    # Model selection
    st.sidebar.selectbox(
        "Select Model",
        options=list(MODEL_PRICING.keys()),
        key="selected_model"
    )
    
    # Display metrics in a styled container
    with st.sidebar.container():
        st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
        
        # Token usage
        st.metric(
            "Total Tokens Used",
            f"{st.session_state.total_tokens:,}"
        )
        
        # Cost
        st.metric(
            "Total Cost",
            f"${st.session_state.total_cost:.4f}"
        )
        
        # Current pricing
        st.markdown("### Current Model Pricing")
        st.markdown(f"""
        - Input: ${MODEL_PRICING[st.session_state.selected_model]['input']}/1K tokens
        - Output: ${MODEL_PRICING[st.session_state.selected_model]['output']}/1K tokens
        """)
        
        st.markdown('</div>', unsafe_allow_html=True)

def save_current_plot() -> List[str]:
    """Save any currently open matplotlib plots"""
    saved_files = []
    for i in plt.get_fignums():
        fig = plt.figure(i)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"visualization_{timestamp}_{i}.png"
        filepath = os.path.join(VISUALIZATION_DIR, filename)
        fig.savefig(filepath, format='png', dpi=300, bbox_inches='tight')
        saved_files.append(filepath)
    plt.close('all')
    return saved_files

def initialize_agent(df):
    """Initialize the LangChain agent with the current model and dataframe"""
    system_prompt = """You are an advanced AI data analyst with expertise in interpreting CSV datasets. Your responsibilities include:

    1. Provide clear and concise analysis of the data
    2. Perform accurate statistical calculations
    3. Identify key insights and trends
    4. Explain complex data points in simple terms
    5. Offer recommendations based on data analysis

    Guidelines:
    - Always show your work and reasoning
    - Use appropriate statistical methods
    - Be objective and data-driven
    - If a query is unclear, ask for clarification
    - Protect sensitive information
    - Avoid making assumptions without data evidence
    - Consider previous conversation context when providing answers
    - Maintain consistency with previous responses
    - If referring to previous conversation, explicitly mention it

    When analyzing the dataset, focus on:
    - Descriptive statistics
    - Correlation and relationships between variables
    - Potential patterns or anomalies
    - Actionable insights

    Communicate in a professional, clear, and helpful manner."""

    # Set up environment
    os.environ["OPENAI_API_KEY"] = ""

    # Create tools and prompt
    tools = [PythonAstREPLTool(locals={"df": df})]
    prompt = _get_functions_single_prompt(df)
    prompt.input_variables.append("chat_history")
    prompt.messages.insert(1, MessagesPlaceholder(variable_name="chat_history"))

    # Create model and agent
    chat_model = ChatOpenAI(
        temperature=0.7,
        model=st.session_state.selected_model
    )
    agent = create_openai_functions_agent(chat_model, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # Create chain with message history
    chain = RunnableWithMessageHistory(
        agent_executor,
        get_session_history=get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    
    return chain

def main():
    # Enhanced header with gradient background
    st.markdown(
    """
    <div class="main-header">
        <h1 style='text-align: center; font-family: "Poppins", sans-serif; margin: 0;'>DASH: Data Analytics & Statistical Hub</h1>
        <p style='text-align: center; margin-top: 10px;'>
            Your Intelligent Data Analysis Assistant
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
    
    # File uploader in the sidebar
    with st.sidebar:
        uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
        
        if uploaded_file is not None and (st.session_state.uploaded_file != uploaded_file):
            try:
                st.session_state.df = pd.read_csv(uploaded_file)
                st.session_state.uploaded_file = uploaded_file
                st.session_state.chain = initialize_agent(st.session_state.df)
                st.success("File successfully uploaded and processed!")
                
                # Display dataset info
                st.markdown(
                    """
                    <div class="dataset-info">
                        <h3 style='color: #1a73e8; margin-bottom: 15px;'>Dataset Information</h3>
                        <div style='background: white; padding: 15px; border-radius: 8px;'>
                    """,
                    unsafe_allow_html=True
                )
                st.markdown(f"""
                    <div style='margin-bottom: 10px;'>
                        <span style='color: #5f6368; font-weight: 500;'>📊 Rows:</span>
                        <span style='color: #1a73e8; font-weight: 500;'>{st.session_state.df.shape[0]:,}</span>
                    </div>
                    <div style='margin-bottom: 10px;'>
                        <span style='color: #5f6368; font-weight: 500;'>📋 Columns:</span>
                        <span style='color: #1a73e8; font-weight: 500;'>{st.session_state.df.shape[1]}</span>
                    </div>
                    <div style='margin-bottom: 10px;'>
                        <span style='color: #5f6368; font-weight: 500;'>🏷️ Column Names:</span>
                        <div style='color: #202124; margin-top: 5px; line-height: 1.6;'>
                            {', '.join([f'<span style="background: #f1f8ff; padding: 2px 8px; border-radius: 12px; font-size: 0.9em;">{col}</span>' for col in st.session_state.df.columns])}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.markdown("</div></div>", unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.session_state.df = None
    
    # Display metrics in sidebar
    display_metrics()
    
    # Main chat interface
    if st.session_state.df is None:
        st.info("Please upload a CSV file to begin the analysis.")
        return
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "image" in message:
                st.image(message["image"])

    # Chat input
    if prompt := st.chat_input("What would you like to know about the data?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)

        # Show thinking animation
        with st.chat_message("assistant"):
            thinking_placeholder = st.empty()
            thinking_placeholder.markdown('<div class="thinking">AI is analyzing your request...</div>', unsafe_allow_html=True)
            
            try:
                # Get AI response
                result = st.session_state.chain.invoke(
                    {"input": prompt},
                    config={"configurable": {"session_id": "default_session"}}
                )
                
                ai_response = result['output']
                
                # Update token metrics
                update_token_metrics(prompt, ai_response)
                
                # Save any generated plots
                saved_paths = save_current_plot()
                
                # Clear thinking animation and display response
                thinking_placeholder.empty()
                st.write(ai_response)
                
                # Display any generated visualizations
                for path in saved_paths:
                    st.image(path)
                    
                # Add AI response to chat history
                message_data = {
                    "role": "assistant",
                    "content": ai_response
                }
                if saved_paths:
                    message_data["image"] = saved_paths
                st.session_state.messages.append(message_data)

            except Exception as e:
                thinking_placeholder.empty()
                st.error(f"An error occurred: {str(e)}")

    # Add a clear chat button
    if st.sidebar.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.total_tokens = 0
        st.session_state.total_cost = 0.0
        store.clear()

if __name__ == "__main__":
    main()
