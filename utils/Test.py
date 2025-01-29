import pandas as pd
import sqlite3
import langgraph
from langchain.chat_models import ChatOpenAI
from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from sqlalchemy import create_engine
from typing import Dict, Any
from langgraph.prebuilt import ToolExecutor
from langgraph.graph import StateGraph

# Load dataset into Pandas DataFrame
df = pd.read_csv("your_dataset.csv")  # Change to your dataset file

# Create SQLite Database & Load Data into SQL Table
engine = create_engine("sqlite:///database.db", echo=True)
df.to_sql("data_table", engine, if_exists="replace", index=False)

# Define Pandas Agent (Executes Pandas Queries)
@tool
def pandas_query(query: str) -> str:
    """Executes a Pandas query on the dataset and returns results."""
    try:
        result = eval(f"df.{query}")
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

# Define SQL Agent (Executes SQL Queries)
@tool
def sql_query(query: str) -> str:
    """Executes an SQL query on the database and returns results."""
    try:
        with engine.connect() as conn:
            result = conn.execute(query).fetchall()
            return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

# Initialize LLM
llm = ChatOpenAI(model="gpt-4")

# Define Tools
tools = [pandas_query, sql_query]
tool_executor = ToolExecutor(tools)

# Create LangChain Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
)

# Define StateGraph for LangGraph
class ChatbotGraph:
    def __init__(self):
        self.graph = StateGraph(Dict[str, Any])

        self.graph.add_node("llm_decision", self.llm_decision)
        self.graph.add_node("execute_tool", tool_executor)

        # LLM decides which tool to call
        self.graph.set_entry_point("llm_decision")
        self.graph.add_edge("llm_decision", "execute_tool")

        self.app = self.graph.compile()

    def llm_decision(self, state: Dict[str, Any]):
        """LLM decides whether to use Pandas or SQL tool."""
        user_input = state["input"]
        response = agent.invoke({"input": user_input})
        return {"tool_calls": response}

# Initialize chatbot
chatbot = ChatbotGraph()

# Example Queries
user_input = "Show me the first 5 rows using Pandas"
response = chatbot.app.invoke({"input": user_input})
print(response)

user_input = "Select * from data_table limit 5"
response = chatbot.app.invoke({"input": user_input})
print(response)
