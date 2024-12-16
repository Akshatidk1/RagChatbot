import streamlit as st
from langchain.agents import create_pandas_dataframe_agent, initialize_agent
from langchain.memory import ConversationBufferMemory
from config.llm_config import llm
from config.vectordb_config import add_to_vectordb, query_vectordb
from tools.stats_tool import dataset_stats
import pandas as pd
from io import StringIO

st.title("Conversational CSV Chatbot with Pandas Agent")
st.write("Upload a CSV file to start your conversation!")

# Initialize memory for conversational context
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# File Upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    # Read CSV and store it as a DataFrame
    df = pd.read_csv(uploaded_file)
    add_to_vectordb(df)  # Add the dataset to VectorDB
    st.session_state["df"] = df  # Store DataFrame in session state
    st.write("Dataset successfully uploaded and added to the vector database.")
    st.write(df.head())  # Show a preview of the dataset

# Tools
if "df" in st.session_state:
    # Create Pandas DataFrame Agent
    pandas_agent = create_pandas_dataframe_agent(llm, st.session_state["df"], verbose=False)

    tools = [
        {"name": "StatsTool", "func": dataset_stats, "description": "Retrieve dataset statistics."}
    ]

    # Initialize the conversational agent with both tools
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent="conversational-react-description",
        memory=memory,
        verbose=False,
    )

    st.write("Chatbot is ready! Start a conversation below:")
    user_input = st.text_input("Your message:", key="input")

    if user_input:
        with st.spinner("Thinking..."):
            try:
                # Handle Pandas-specific queries
                if "stats" in user_input.lower() or "analyze" in user_input.lower():
                    response = pandas_agent.run(user_input)
                else:
                    # Use the conversational agent for general queries
                    response = agent.run(input=user_input)
                # Append conversation history
                st.session_state["chat_history"].append((user_input, response))
            except Exception as e:
                response = f"Error: {e}"
                st.session_state["chat_history"].append((user_input, response))

    # Display Chat History
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for user_msg, bot_msg in st.session_state["chat_history"]:
        st.write(f"**You:** {user_msg}")
        st.write(f"**Bot:** {bot_msg}")
