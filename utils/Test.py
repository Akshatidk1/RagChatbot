Here’s an end-to-end chatbot implementation based on your requirements. It combines LangChain, Streamlit, and pandas to create a conversational chatbot that:
	1.	Takes a CSV file as input.
	2.	Allows for general conversational AI.
	3.	Provides pandas-based dataset manipulation.
	4.	Returns dataset statistics using a custom tool.
	5.	Uses a vector database for embeddings to answer contextual questions.

File Structure

chatbot/
│
├── app.py                # Main Streamlit app
├── config/
│   ├── __init__.py       # Configuration package initialization
│   ├── llm_config.py     # LLM and embedding model configurations
│   ├── vectordb_config.py # Vector DB configuration
│
├── agents/
│   ├── __init__.py       # Agents package initialization
│   ├── query_classifier.py # Query classification logic
│   ├── pandas_agent.py   # Pandas agent logic
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

Configures the LLM and embedding model.

from langchain_openai import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings

# Initialize the LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Initialize the embedding model
embedding_model = OpenAIEmbeddings()

2. config/vectordb_config.py

Sets up a vector database for storing and retrieving embeddings.

from langchain.vectorstores import FAISS
from langchain.schema import Document

# Initialize the FAISS vector database
vector_db = FAISS(embedding_model.embedding_function, index=None)

# Function to add documents to the vector database
def add_to_vector_db(data):
    documents = [Document(page_content=doc) for doc in data]
    vector_db.add_documents(documents)

3. agents/query_classifier.py

Implements query classification logic.

def classify_query(user_input):
    """
    Classifies the user's query into:
    1. General Question
    2. Pandas Manipulation
    3. Dataset Stats
    """
    if "stats" in user_input.lower():
        return "stats"
    elif "pandas" in user_input.lower() or "filter" in user_input.lower():
        return "pandas"
    else:
        return "general"

4. agents/pandas_agent.py

Handles dataset manipulation using pandas.

from langchain.agents import tool
import pandas as pd

@tool
def manipulate_dataset(csv_path: str, query: str):
    """Applies the user-defined query on the dataset."""
    try:
        df = pd.read_csv(csv_path)
        result = eval(query)  # WARNING: Use `eval` cautiously. Prefer pandas query methods.
        return result.to_string()
    except Exception as e:
        return f"Error manipulating dataset: {str(e)}"

5. agents/stats_tool.py

Generates statistics for the dataset.

from langchain.agents import tool
import pandas as pd

@tool
def dataset_stats(csv_path: str):
    """Returns basic statistics of the dataset."""
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

6. utils/file_handler.py

Handles file uploads and storage.

import os

def save_uploaded_file(uploaded_file):
    """Saves the uploaded file to disk."""
    file_path = os.path.join("uploaded_files", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

7. app.py

Main Streamlit app to bring everything together.

import streamlit as st
from config.llm_config import llm
from config.vectordb_config import add_to_vector_db, vector_db
from agents.query_classifier import classify_query
from agents.pandas_agent import manipulate_dataset
from agents.stats_tool import dataset_stats
from utils.file_handler import save_uploaded_file

st.title("AI-Powered CSV Chatbot")

# File upload
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    # Save the file and prepare vector DB
    file_path = save_uploaded_file(uploaded_file)
    st.write(f"Uploaded file saved at: {file_path}")

    # Add CSV content to vector DB
    with open(file_path, "r") as f:
        add_to_vector_db(f.readlines())

    # Start conversation
    st.write("Chatbot is ready to interact!")
    chat_history = []
    
    while True:
        user_input = st.text_input("Ask a question or give a task:")

        if user_input:
            # Classify query
            query_type = classify_query(user_input)
            
            if query_type == "general":
                # Handle general questions
                response = llm.invoke(user_input)
                st.write(response)
            elif query_type == "pandas":
                # Handle pandas queries
                response = manipulate_dataset(csv_path=file_path, query=user_input)
                st.write(response)
            elif query_type == "stats":
                # Handle dataset statistics
                response = dataset_stats(csv_path=file_path)
                st.write(response)
            
            # Update chat history
            chat_history.append({"user": user_input, "bot": response})

8. requirements.txt

streamlit
langchain
faiss-cpu
openai
pandas

How It Works
	1.	File Upload:
	•	The user uploads a CSV file.
	•	The app saves the file and prepares embeddings for the content.
	2.	Chat:
	•	Users can ask general questions, request pandas manipulations, or ask for dataset statistics.
	•	The classify_query function determines which tool/agent to invoke.
	3.	Tools:
	•	manipulate_dataset: For pandas manipulations.
	•	dataset_stats: For statistics about the dataset.
	•	General LLM responses for other queries.
	4.	Execution:
	•	Streamlit handles the UI, while LangChain agents handle the logic.

You can run the app with:

streamlit run app.py

Enjoy your AI-powered CSV chatbot!
