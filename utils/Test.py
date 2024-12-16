Here’s the updated end-to-end code incorporating all the feedback and improvements:
	•	Uses LangGraph for dynamic agent management.
	•	Replaces manual query classification with LLM-based classification.
	•	Integrates create_pandas_dataframe_agent for CSV operations.
	•	Allows dynamic addition of agents.
	•	Includes memory for the chatbot and a reset memory button.

Directory Structure

project/
│
├── app.py                # Main Streamlit app
├── agents.py             # Agent management and dynamic agent setup
├── query_classifier.py   # LLM-based query classification
├── requirements.txt      # Required libraries
└── utils.py              # Utility functions for modularity

app.py (Main Streamlit App)

import streamlit as st
from agents import AgentManager
from query_classifier import QueryClassifier

# Initialize the agent manager
agent_manager = AgentManager()
query_classifier = QueryClassifier()

# Streamlit app UI
st.title("Dynamic Chatbot with CSV and Forecasting Agents")

# File upload for CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    agent_manager.load_csv_agent(uploaded_file)
    st.write("CSV file uploaded and agent initialized.")

# Reset memory button
if st.button("Reset Chat Memory"):
    agent_manager.reset_memory()
    st.write("Chat memory reset.")

# Chat interface
st.text_input("User Input", placeholder="Ask your question here...", key="user_input")
user_input = st.session_state.get("user_input")

if user_input:
    # Classify the query type
    query_type = query_classifier.classify(user_input)

    # Process query based on its type
    response = agent_manager.handle_query(user_input, query_type)
    st.write(response)

agents.py (Agent Management)

import pandas as pd
from langgraph.agents import AgentManager as LGAgentManager
from langchain_experimental.tools import create_pandas_dataframe_agent
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI

class AgentManager:
    def __init__(self):
        """
        Initializes the AgentManager with LangGraph and pre-defined agents.
        """
        self.memory = ConversationBufferMemory(chat_memory=True, return_messages=True)
        self.model = ChatOpenAI(temperature=0, model="gpt-4")  # LLM used for agents
        self.csv_agent = None
        self.agents = LGAgentManager(memory=self.memory)

    def load_csv_agent(self, file):
        """
        Load a CSV file and initialize a Pandas agent.
        """
        data = pd.read_csv(file)
        self.csv_agent = create_pandas_dataframe_agent(
            llm=self.model,
            df=data,
            memory=self.memory,
            verbose=True,
        )
        self.agents.add_tool("CSV Agent", self.csv_agent)

    def handle_query(self, query, query_type):
        """
        Handles the user query by routing it to the appropriate agent.
        """
        if query_type == "csv":
            if not self.csv_agent:
                return "No CSV agent is available. Please upload a CSV file."
            return self.csv_agent.run(query)
        elif query_type == "forecasting":
            # Future: Add forecasting agent here
            return "Forecasting agent is not yet implemented."
        elif query_type == "general":
            return self.agents.run_tool(query)
        else:
            return "I could not classify your query. Please try again."

    def reset_memory(self):
        """
        Reset conversation memory.
        """
        self.memory.clear()

query_classifier.py (LLM-Based Query Classification)

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

class QueryClassifier:
    def __init__(self):
        """
        Initialize the query classifier with a prompt and an LLM.
        """
        self.model = ChatOpenAI(temperature=0, model="gpt-4")
        self.prompt = PromptTemplate(
            input_variables=["query"],
            template="""
            Classify the user query into one of the following categories:
            - general: For general knowledge questions.
            - csv: For questions related to uploaded CSV data.
            - forecasting: For questions requesting time-series forecasting.

            User Query: {query}
            Classification:
            """
        )
        self.chain = LLMChain(llm=self.model, prompt=self.prompt)

    def classify(self, query):
        """
        Classify the query using the LLM.
        """
        result = self.chain.run(query)
        return result.strip().lower()

utils.py (Utility Functions)

import pandas as pd

def load_csv(file):
    """
    Utility to load a CSV file into a Pandas DataFrame.
    """
    return pd.read_csv(file)

def extract_forecasting_column(query, data):
    """
    Extracts the column to forecast based on the user query.
    """
    # Example: Parse the query to find the column name for forecasting.
    for col in data.columns:
        if col.lower() in query.lower():
            return col
    return None

Requirements (requirements.txt)

streamlit
langchain
langgraph
pandas
openai

How It Works
	1.	Dynamic Agent Management with LangGraph:
	•	Uses LangGraph.AgentManager to dynamically add, manage, and invoke agents.
	2.	LLM-Based Query Classification:
	•	QueryClassifier uses GPT-4 to classify queries into three types: general, csv, and forecasting.
	3.	CSV Agent with create_pandas_dataframe_agent:
	•	Handles filtering, sorting, and aggregation queries on the uploaded CSV.
	4.	Reset Memory Button:
	•	Clears conversation memory for a fresh chat.
	5.	Future Extensibility:
	•	Easy to add a forecasting agent or any additional agents.

Output Example

Interaction:
	1.	User Uploads CSV
Agent initialized.
	2.	User Query: “What are the top 5 rows?”
CSV Agent responds.
	3.	User Query: “Tell me about Norway.”
General Knowledge Agent responds.
	4.	User Query: “Predict sales for the next month based on the ‘Sales’ column.”
Forecasting Agent processes (if implemented).

Let me know if you’d like further refinements or extensions!
