import pandas as pd
import sqlite3
from typing import Dict, Any
from sqlalchemy import create_engine
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.tools import tool
from langgraph.graph import StateGraph

# Load dataset into Pandas DataFrame
df = pd.read_csv("your_dataset.csv")  # Change to your dataset file

# Create SQLite Database & Load Data into SQL Table
engine = create_engine("sqlite:///database.db", echo=True)
df.to_sql("data_table", engine, if_exists="replace", index=False)

# Define Pandas Query Tool
@tool
def pandas_query(query: str) -> str:
    """Executes a Pandas query on the dataset and returns results."""
    try:
        result = eval(f"df.{query}")
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

# Define SQL Query Tool
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
TOOLS = {"pandas_query": pandas_query, "sql_query": sql_query}

# Create LangChain Agent
agent = initialize_agent(
    tools=list(TOOLS.values()),  # Pass list of tools
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
        self.graph.add_node("execute_tool", self.execute_tool)

        # LLM decides which tool to call
        self.graph.set_entry_point("llm_decision")
        self.graph.add_edge("llm_decision", "execute_tool")

        self.app = self.graph.compile()

    def llm_decision(self, state: Dict[str, Any]):
        """LLM decides which tool to use (Pandas or SQL)."""
        user_input = state["input"]
        response = agent.invoke({"input": user_input})

        # Extract tool name
        if "pandas" in user_input.lower():
            tool_name = "pandas_query"
        else:
            tool_name = "sql_query"

        return {"tool_name": tool_name, "query": user_input}

    def execute_tool(self, state: Dict[str, Any]):
        """Executes the chosen tool."""
        tool_name = state["tool_name"]
        query = state["query"]

        if tool_name in TOOLS:
            result = TOOLS[tool_name].run(query)
        else:
            result = "Invalid tool selection"

        return {"output": result}

# Initialize chatbot
chatbot = ChatbotGraph()

# Example Queries
user_input = "Show me the first 5 rows using Pandas"
response = chatbot.app.invoke({"input": user_input})
print(response["output"])

user_input = "Select * from data_table limit 5"
response = chatbot.app.invoke({"input": user_input})
print(response["output"])
