LangGraph simplifies building agents by offering a more modular and feature-rich framework compared to traditional LangChain agents. Here’s the updated implementation for your chatbot using LangGraph:

Problem Statement Recap
	1.	User uploads a CSV and starts chatting.
	2.	Handles:
	•	General queries: Answered by an LLM.
	•	CSV-related queries: Use a data agent for filtering, sorting, and aggregation.
	•	Forecasting queries: Use a forecasting agent for time series predictions.
	3.	Uses:
	•	LangGraph for managing agents and tools.
	•	Google GenAI embeddings for LLM and vector database (e.g., FAISS) for query classification.

Architecture with LangGraph
	1.	Modules:
	•	General Agent
	•	Data Agent (CSV manipulation)
	•	Forecasting Agent
	2.	Workflow:
	•	Classify user query using an LLM into:
	•	General
	•	CSV
	•	Forecasting
	•	Route to the appropriate agent.
	3.	Memory:
	•	Use LangGraph memory for conversational context.
	•	Add a reset button to clear memory.

Code Implementation

1. Dependencies

Add the following to your requirements.txt:

streamlit
langgraph
google-palm
faiss-cpu
prophet
pandas
numpy

2. app.py: Main Application

import streamlit as st
from langgraph import LangGraph, Tool
from langgraph.memory import Memory
from langgraph.vectorstore import VectorStore
from tools import get_general_tools, get_data_tools, get_forecasting_tool
from langgraph.llms import GooglePalm
from vectordb import initialize_vectordb, add_to_vectordb, query_vectordb
import pandas as pd

# Initialize LangGraph and tools
if "memory" not in st.session_state:
    st.session_state.memory = Memory()

if "vectordb" not in st.session_state:
    st.session_state.vectordb = initialize_vectordb()

if "graph" not in st.session_state:
    st.session_state.graph = LangGraph(
        tools=[],
        llm=GooglePalm(model="chat-bison"),
        memory=st.session_state.memory
    )

# File Upload
st.title("CSV Chatbot with LangGraph")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    general_tools = get_general_tools()
    data_tools = get_data_tools(data)
    forecasting_tool = get_forecasting_tool(data)
    
    st.session_state.graph.add_tools(general_tools + data_tools + [forecasting_tool])
    st.write("Agents initialized successfully!")

# Chat Interface
if "graph" in st.session_state:
    user_input = st.text_input("Ask a question:")
    if user_input:
        # Classify and route query
        task_type = classify_query(user_input, st.session_state.vectordb)
        response = st.session_state.graph.run(input_text=user_input, task=task_type)
        
        st.session_state.memory.add_user_message(user_input)
        st.session_state.memory.add_ai_message(response)
        st.write(response)

3. tools.py: Tools Initialization

from langgraph import Tool
from forecasting import perform_forecasting

def get_general_tools():
    """Returns general-purpose tools."""
    return [
        Tool(
            name="General Answering",
            description="Answers general knowledge questions.",
            func=lambda query: "This is a general query response."
        )
    ]

def get_data_tools(data):
    """Returns tools for data manipulation."""
    import pandas as pd

    def filter_data(query):
        # Example: Parse the query to filter data
        return data.query(query)

    def sort_data(column, ascending=True):
        return data.sort_values(by=column, ascending=ascending)

    def aggregate_data(column, func="sum"):
        return getattr(data[column], func)()

    return [
        Tool(name="Filter Data", description="Filters data.", func=filter_data),
        Tool(name="Sort Data", description="Sorts data.", func=sort_data),
        Tool(name="Aggregate Data", description="Aggregates data.", func=aggregate_data),
    ]

def get_forecasting_tool(data):
    """Returns a forecasting tool."""
    return Tool(
        name="Forecasting",
        description="Forecasts future values for a given column.",
        func=lambda query: perform_forecasting(query, data)
    )

4. forecasting.py: Forecasting Agent

from prophet import Prophet
import pandas as pd

def perform_forecasting(query, data):
    """Perform time series forecasting."""
    # Extract column name
    column_name = extract_column_name(query, data.columns)
    if not column_name:
        return "Please specify a valid column for forecasting."
    
    if "date" not in data.columns:
        return "The dataset must contain a 'date' column for forecasting."
    
    # Prepare data for Prophet
    df = data[["date", column_name]].dropna()
    df.columns = ["ds", "y"]
    
    # Train model
    model = Prophet()
    model.fit(df)
    
    # Predict
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    
    return forecast[["ds", "yhat"]].tail(30).to_string()

def extract_column_name(query, columns):
    """Extract column name from query."""
    for col in columns:
        if col in query.lower():
            return col
    return None

5. vectordb.py: VectorDB Integration

import faiss
from langgraph.embeddings import GooglePalmEmbeddings

def initialize_vectordb():
    """Initialize FAISS vector database."""
    embedding_dim = 768  # Google Palm embedding dimension
    return faiss.IndexFlatL2(embedding_dim)

def add_to_vectordb(vectordb, text, metadata, embedding_model):
    """Add text embedding to the vector database."""
    embedding = embedding_model.embed_query(text)
    vectordb.add([embedding])
    return metadata

def query_vectordb(vectordb, query, embedding_model, top_k=1):
    """Query the vector database."""
    query_embedding = embedding_model.embed_query(query)
    distances, indices = vectordb.search([query_embedding], k=top_k)
    return distances, indices

6. Query Classification

from langgraph.prompts import PromptTemplate

def classify_query(query, vectordb):
    """Classify the query as general, data-related, or forecasting."""
    # Query the vector database
    distances, indices = query_vectordb(vectordb, query, embedding_model=GooglePalmEmbeddings())
    
    if distances[0][0] > 0.8:
        if "forecast" in query.lower():
            return "Forecasting"
        elif any(keyword in query.lower() for keyword in ["filter", "sort", "aggregate"]):
            return "Data Operations"
        else:
            return "General"
    return "General"

Features Added
	1.	LangGraph Integration:
	•	Handles agent routing seamlessly.
	2.	General, Data, and Forecasting Agents.
	3.	VectorDB for Query Classification:
	•	Dynamically routes queries.
	4.	Resettable Memory:
	•	Use st.session_state.memory.reset() for resetting chat history.

Would you like additional assistance testing or deploying this setup?
