Below is the corrected and complete implementation using AgentExecutor, Tool, and query classification using the LLM. Additionally, a custom ChatPromptTemplate is defined to guide the interactions.

Updated File Structure

chatbot/
├── config/
│   ├── embeddings_config.py
│   ├── llm_config.py
│   ├── vectordb_config.py
├── tools/
│   ├── pandas_tool.py
│   ├── stats_tool.py
├── agents/
│   ├── query_handler.py
├── main.py
├── requirements.txt
└── README.md

Code Implementation

1. requirements.txt

langchain
streamlit
pandas
numpy
openai
sentence-transformers
pinecone-client

2. config/embeddings_config.py

from langchain.embeddings import OpenAIEmbeddings

def get_embeddings():
    return OpenAIEmbeddings()

3. config/llm_config.py

from langchain.llms import OpenAI

def get_llm():
    return OpenAI(model="gpt-4", temperature=0.7)

4. config/vectordb_config.py

from langchain.vectorstores import Pinecone
import pinecone
from config.embeddings_config import get_embeddings

def get_vectordb(index_name="chatbot_index"):
    pinecone.init(api_key="your-pinecone-api-key", environment="us-west1-gcp")
    return Pinecone.from_existing_index(index_name, embedding=get_embeddings())

5. tools/pandas_tool.py

import pandas as pd
from langchain.tools import Tool

def pandas_query_tool(dataframe: pd.DataFrame):
    def pandas_query_func(query: str):
        try:
            result = eval(query)
            return str(result)
        except Exception as e:
            return f"Error processing query: {str(e)}"

    return Tool(
        name="PandasTool",
        description="Use this tool for manipulating or querying the dataset. Provide Pythonic commands for operations.",
        func=pandas_query_func,
    )

6. tools/stats_tool.py

from langchain.tools import Tool

def dataset_stats_tool(dataframe):
    def stats_func(_):
        stats = {
            "columns": dataframe.columns.tolist(),
            "shape": dataframe.shape,
            "missing_values": dataframe.isnull().sum().to_dict(),
            "data_types": dataframe.dtypes.to_dict(),
            "summary": dataframe.describe().to_dict(),
        }
        return stats

    return Tool(
        name="StatsTool",
        description="Use this tool to retrieve dataset statistics such as shape, columns, missing values, and summary.",
        func=stats_func,
    )

7. agents/query_handler.py

from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
from config.llm_config import get_llm

def query_classifier():
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                "You are a smart query classifier. Based on the input query, classify it into one of these categories: `stats`, `dataframe`, or `general`."
            ),
            HumanMessagePromptTemplate.from_template("{query}"),
        ]
    )
    return LLMChain(llm=get_llm(), prompt=prompt)

8. main.py

import streamlit as st
import pandas as pd
from tools.pandas_tool import pandas_query_tool
from tools.stats_tool import dataset_stats_tool
from agents.query_handler import query_classifier
from config.llm_config import get_llm
from langchain.agents import Tool, AgentExecutor
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

st.title("Conversational Chatbot with CSV Support")

# Upload CSV
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    dataframe = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.write(dataframe.head())

    # Initialize Tools
    pandas_tool = pandas_query_tool(dataframe)
    stats_tool = dataset_stats_tool(dataframe)
    
    tools = [pandas_tool, stats_tool]

    # Query Classifier
    classifier = query_classifier()

    # Define Chat Prompt Template
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                "You are an intelligent assistant that can answer general queries, handle dataset manipulations, and provide dataset statistics. "
                "Classify the query first, then use the appropriate tool."
            ),
            HumanMessagePromptTemplate.from_template("{input}")
        ]
    )

    # Agent Executor
    agent_executor = AgentExecutor.from_agent_and_tools(
        llm=get_llm(),
        tools=tools,
        verbose=True,
        agent_prompt=prompt,
    )

    st.header("Chat Interface")
    user_input = st.text_input("Ask your question:")

    if user_input:
        # Classify the query
        classification = classifier.run({"query": user_input}).strip()

        st.write(f"Query classified as: `{classification}`")
        
        if classification == "general":
            response = get_llm().run(user_input)
        else:
            response = agent_executor.run(input=user_input)
        
        st.write("Chatbot Response:")
        st.write(response)

How It Works
	1.	File Upload:
Users upload a CSV file, and the system parses it into a pandas dataframe.
	2.	Query Classification:
	•	The LLM classifies the query into one of three categories:
	•	stats: Dataset statistics.
	•	dataframe: Pandas manipulation or query.
	•	general: General conversational response.
	3.	AgentExecutor and Tools:
	•	StatsTool: Handles dataset statistics.
	•	PandasTool: Handles pandas operations.
	•	Both tools are registered with the AgentExecutor.
	4.	Chat Prompt Template:
Guides the chatbot with a clear context to handle user queries and decide appropriate actions.
	5.	Response Generation:
	•	Based on the classified query, the relevant tool is invoked via AgentExecutor.
	•	For general queries, the LLM directly responds.

Key Improvements
	1.	AgentExecutor Integration: Ensures tools are efficiently used with LLM for specific tasks.
	2.	Tool Creation: Custom Tool objects for pandas operations and statistics.
	3.	LLM-Based Classification: Dynamically routes queries using the LLM.
	4.	Modular Structure: Keeps code organized and extensible.

Let me know if you’d like further clarifications!
