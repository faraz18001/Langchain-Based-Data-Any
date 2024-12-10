import os
import pandas as pd
import streamlit as st
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import tiktoken
from langchain.schema import SystemMessage

# System prompt (unchanged from previous version)
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

When analyzing the dataset, focus on:
- Descriptive statistics
- Correlation and relationships between variables
- Potential patterns or anomalies
- Actionable insights

Communicate in a professional, clear, and helpful manner."""

def total_token_counter(input_message: str, output_ai_message: str):
    try:
        encoding = tiktoken.get_encoding('cl100k_base')
        input_tokens = encoding.encode(input_message)
        output_tokens = encoding.encode(output_ai_message)
        total_tokens = len(input_tokens) + len(output_tokens)
        return total_tokens
    except Exception as e:
        st.error(f'Error counting tokens: {str(e)}')
        return 0

def initialize_session_state():
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'total_tokens' not in st.session_state:
        st.session_state.total_tokens = 0
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ''

def main():
    st.set_page_config(page_title="AI Data Analyst", page_icon="📊", layout="wide")
    
    st.title("AI Data Analyst Assistant 📊")
    
    initialize_session_state()
    
    # API Key Input
    st.sidebar.header("OpenAI API Configuration")
    api_key = st.sidebar.text_input("Enter OpenAI API Key", 
                                    type="password", 
                                    value=st.session_state.api_key,
                                    help="Your API key is used to authenticate with OpenAI's services.")
    
    # Validate and store API key
    if api_key:
        st.session_state.api_key = api_key
        os.environ["OPENAI_API_KEY"] = api_key
    
    # Disable further interactions if no API key
    if not api_key:
        st.warning("Please enter your OpenAI API Key to proceed.")
        return
    
    # File uploader
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
    
    if uploaded_file is not None:
        # Load the data
        df = pd.read_csv(uploaded_file)
        
        # Display data preview
        with st.expander("Preview Dataset"):
            st.dataframe(df.head())
        
        # Initialize the agent
        try:
            llm = ChatOpenAI(temperature=0.7, model="gpt-4")
            system_message = SystemMessage(content=system_prompt)
            
            agent = create_pandas_dataframe_agent(
                llm,
                df,
                verbose=True,
                agent_type=AgentType.OPENAI_FUNCTIONS,
                allow_dangerous_code=True,
                system_message=system_message
            )
            
            # Chat interface
            st.subheader("Chat with your Data")
            
            # Display conversation history
            for message in st.session_state.conversation_history:
                role = message["role"]
                content = message["content"]
                
                if role == "user":
                    st.write(f"You: {content}")
                else:
                    st.write(f"AI: {content}")
            
            # User input
            user_input = st.text_area("Ask me anything about your data:", height=100)
            
            if st.button("Send"):
                if user_input:
                    # Add user message to history
                    st.session_state.conversation_history.append({"role": "user", "content": user_input})
                    
                    # Get AI response
                    with st.spinner("Thinking..."):
                        result = agent.invoke({"input": user_input, "chat_history": st.session_state.conversation_history})
                        ai_response = result['output']
                    
                    # Add AI response to history
                    st.session_state.conversation_history.append({"role": "assistant", "content": ai_response})
                    
                    # Calculate tokens
                    conversation_tokens = total_token_counter(user_input, ai_response)
                    st.session_state.total_tokens += conversation_tokens
                    
                    # Force refresh
                    st.rerun()
            
            # Display token usage
            st.sidebar.subheader("Token Usage")
            st.sidebar.metric("Total Tokens Used", st.session_state.total_tokens)
            
            # Clear conversation button
            if st.sidebar.button("Clear Conversation"):
                st.session_state.conversation_history = []
                st.session_state.total_tokens = 0
                st.rerun()
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    
    else:
        st.info("Please upload a CSV file to begin analysis.")
        
    # Footer
    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit and LangChain")

if __name__ == "__main__":
    main()
