import os
import pandas as pd
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import tiktoken

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = ""

# Load the Excel file into a DataFrame
df = pd.read_excel('DU Cash Card Aug.xlsx', sheet_name=0)

# Create the Pandas DataFrame agent with security override
llm = ChatOpenAI(temperature=0.7, model="gpt-4")
agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    allow_dangerous_code=True
)

def num_tokens_from_messages(messages, model="gpt-4"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    num_tokens = 0
    for message in messages:
        num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":  # if there's a name, the role is omitted
                num_tokens += -1  # role is always required and always 1 token
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens

def chat_with_agent():
    total_tokens = 0
    conversation_history = []
    
    print("Hello! I'm your AI assistant. I can help you analyze the Excel data. What would you like to know? (Type 'exit' to end the conversation)")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("AI: Thank you for chatting. Goodbye!")
            break
        
        conversation_history.append({"role": "user", "content": user_input})
        
        result = agent.invoke({"input": user_input, "chat_history": conversation_history})
        
        ai_response = result['output']
        print(f"AI: {ai_response}")
        
        conversation_history.append({"role": "assistant", "content": ai_response})
        
        # Count tokens
        tokens_used = num_tokens_from_messages(conversation_history)
        total_tokens += tokens_used
        
        print(f"\nTokens used in this interaction: {tokens_used}")
        print(f"Total tokens used in this session: {total_tokens}\n")

if __name__ == "__main__":
    chat_with_agent()
    