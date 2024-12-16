Here is an end-to-end chatbot implementation that uses a normal LangChain architecture where query classification and agent handling are dynamically managed. It avoids explicit conditional handling but ensures that the appropriate tools (e.g., pandas agent, custom stats agent) are invoked based on the input. This design leverages tool binding with the LLM to allow it to decide which tool to use naturally.

File Structure

chatbot/
│
├── app.py                # Main Streamlit app
├── config/
│   ├── __init__.py       # Configuration package initialization
│   ├── llm_config.py     # LLM and embedding model configurations
│   ├── vectordb_config.py # Vector DB configuration
│
├── tools/
│   ├── __init__.py       # Tools package initialization
│   ├── pandas_tool.py    # Pandas agent logic
│   ├── stats_tool.py     # Custom stats tool for dataset
│
├── utils/
│   ├── __init__.py       # Utilities package initialization
│   ├── file_handler.py   # File handling logic
│
├── requirements.txt      # Required libraries
└── README.md             # Project documentation

File Details

1. config/llm_config.py

Centralizes the LLM and embedding model configuration.

from langchain_openai import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings

# LLM initialization
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Embedding model initialization
embedding_model = OpenAIEmbeddings()

2. config/vectordb_config.py

Sets up the vector database for contextual conversation handling.

from langchain.vectorstores import FAISS
from langchain.schema import Document

# Vector database initialization
vector_db = FAISS(embedding_function=embedding_model.embedding_function, index=None)

def add_to_vector_db(data):
    """
    Adds textual data to the vector database for semantic search.
    """
    documents = [Document(page_content=doc) for doc in data]
    vector_db.add_documents(documents)

3. tools/pandas_tool.py

Provides pandas-specific dataset manipulation functionality.

from langchain.agents import tool
import pandas as pd

@tool
def manipulate_dataset(csv_path: str, query: str):
    """
    Applies a pandas operation to the dataset based on the user query.
    """
    try:
        df = pd.read_csv(csv_path)
        result = eval(query)  # WARNING: Use carefully in production
        return result.to_string()
    except Exception as e:
        return f"Error manipulating dataset: {str(e)}"

4. tools/stats_tool.py

Implements dataset statistics functionality.

from langchain.agents import tool
import pandas as pd

@tool
def dataset_stats(csv_path: str):
    """
    Returns basic statistics about the uploaded dataset.
    """
    try:
        df = pd.read_csv(csv_path)
        stats = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
        }
        return stats
    except Exception as e:
        return f"Error generating stats: {str(e)}"

5. utils/file_handler.py

Handles file uploads in Streamlit.

import os

def save_uploaded_file(uploaded_file):
    """
    Saves the uploaded file locally.
    """
    file_path = os.path.join("uploaded_files", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

6. app.py

The main Streamlit app integrates tools, LLM, and file handling.

import streamlit as st
from langchain.agents import AgentExecutor
from config.llm_config import llm
from config.vectordb_config import add_to_vector_db, vector_db
from tools.pandas_tool import manipulate_dataset
from tools.stats_tool import dataset_stats
from utils.file_handler import save_uploaded_file

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser

# Streamlit UI
st.title("AI-Powered CSV Chatbot")

# File upload section
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    # Save file locally
    file_path = save_uploaded_file(uploaded_file)
    st.write(f"File successfully uploaded: {file_path}")

    # Add file content to vector DB
    with open(file_path, "r") as f:
        add_to_vector_db(f.readlines())

    # Define tools
    tools = [manipulate_dataset, dataset_stats]

    # Create prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an assistant capable of answering questions and analyzing datasets."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # Define agent
    agent = (
        {
            "input": lambda x: x["input"],
            "chat_history": lambda x: x["chat_history"],
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
        }
        | prompt
        | llm.bind_tools(tools)
        | OpenAIToolsAgentOutputParser()
    )

    # Agent executor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # Chat functionality
    st.write("Chatbot ready for interaction!")
    chat_history = []

    while True:
        user_input = st.text_input("Ask a question:")
        if user_input:
            result = agent_executor.invoke({"input": user_input, "chat_history": chat_history})
            st.write(result["output"])

            # Track conversation
            chat_history.append({"user": user_input, "bot": result["output"]})

Explanation
	1.	Dynamic Tool Usage:
	•	Tools are bound to the LLM using llm.bind_tools(tools). This allows the LLM to invoke the correct tool automatically based on the user query.
	2.	No Conditional Logic:
	•	There is no explicit query classification or conditional handling. The LLM decides which tool to invoke (e.g., pandas_tool or stats_tool) based on the query context.
	3.	Vector DB:
	•	A FAISS vector database is initialized for storing dataset embeddings. This enables contextual understanding of the dataset for semantic searches.
	4.	Streamlit:
	•	Provides a user-friendly interface for file uploads and chatting.

Run the Application
	1.	Install dependencies:

pip install -r requirements.txt


	2.	Start the app:

streamlit run app.py


	3.	Upload a CSV file and start chatting!

Sample Queries
	1.	General Question:
	•	“What is the capital of Norway?”
	•	The LLM responds as a general chatbot.
	2.	Pandas Manipulation:
	•	“Show me the rows where column ‘A’ > 10.”
	•	The manipulate_dataset tool processes this.
	3.	Dataset Statistics:
	•	“Give me the stats of the dataset.”
	•	The dataset_stats tool generates a detailed summary.

This structure ensures clean separation of concerns, dynamic tool usage, and seamless conversational experience.
