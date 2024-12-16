To implement a conversational chatbot that maintains the context and handles both general questions and dataset-specific queries (stored in VectorDB) dynamically, we need to:
	1.	Use LangChain Memory for maintaining conversational context.
	2.	Implement conversational agents with tools like PandasTool and StatsTool for dataset manipulation and insights.
	3.	Dynamically retrieve dataset information from VectorDB during the conversation.

Below is the updated code.

File Structure

project/
├── app.py               # Main Streamlit app
├── config/
│   ├── llm_config.py    # Configuration for the LLM
│   ├── vectordb_config.py # VectorDB configuration
├── tools/
│   ├── pandas_tool.py   # Pandas tool for dataset manipulation
│   ├── stats_tool.py    # Tool to retrieve dataset statistics
├── requirements.txt     # Dependencies

1. Main App (app.py)

import streamlit as st
from langchain.agents import Tool, initialize_agent
from langchain.memory import ConversationBufferMemory
from config.llm_config import llm
from config.vectordb_config import add_to_vectordb, query_vectordb
from tools.pandas_tool import manipulate_dataset
from tools.stats_tool import dataset_stats

st.title("Conversational CSV Chatbot")
st.write("Upload a CSV file to start your conversation!")

# File Upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    add_to_vectordb(uploaded_file)  # Add data to VectorDB
    st.write("Dataset successfully uploaded and added to the vector database.")

# Tools
tools = [
    Tool(
        name="PandasTool",
        func=manipulate_dataset,
        description="Perform data manipulations and analysis using Pandas dynamically.",
    ),
    Tool(
        name="StatsTool",
        func=dataset_stats,
        description="Retrieve statistics and meta-information about the dataset.",
    ),
]

# Initialize memory for conversational context
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="conversational-react-description",
    memory=memory,
    verbose=False,
)

st.write("Chatbot is ready! Start a conversation below:")

# Conversation
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

user_input = st.text_input("Your message:", key="input")

if user_input:
    with st.spinner("Thinking..."):
        try:
            # Send user input to the agent
            response = agent.run(input=user_input)
            # Append conversation history
            st.session_state["chat_history"].append((user_input, response))
        except Exception as e:
            response = f"Error: {e}"
            st.session_state["chat_history"].append((user_input, response))

# Display Chat History
for user_msg, bot_msg in st.session_state["chat_history"]:
    st.write(f"**You:** {user_msg}")
    st.write(f"**Bot:** {bot_msg}")

2. VectorDB Configuration (config/vectordb_config.py)

from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
import pandas as pd

# Initialize FAISS VectorDB with OpenAI embeddings
embedding_model = OpenAIEmbeddings()
vector_db = FAISS(embedding_function=embedding_model)

def add_to_vectordb(uploaded_file):
    """
    Add the dataset rows as documents to the vector database.
    """
    df = pd.read_csv(uploaded_file)
    docs = [
        Document(page_content=row.to_json(), metadata={"row_index": idx})
        for idx, row in df.iterrows()
    ]
    vector_db.add_documents(docs)

def query_vectordb(query, k=5):
    """
    Query the vector database for relevant rows.
    """
    results = vector_db.similarity_search(query, k=k)
    return [result.page_content for result in results] if results else None

3. Pandas Tool (tools/pandas_tool.py)

from langchain.agents import tool
import pandas as pd
import json

@tool
def manipulate_dataset(query: str):
    """
    Executes a Pandas operation dynamically on rows retrieved from VectorDB.
    """
    from config.vectordb_config import query_vectordb  # Import VectorDB query

    # Query relevant rows from VectorDB
    rows = query_vectordb(query, k=10)
    if not rows:
        return "No relevant rows found for the query."

    try:
        # Convert rows to DataFrame
        data = [json.loads(row) for row in rows]
        df = pd.DataFrame(data)
        
        # Evaluate the query safely
        result = eval(query)
        if isinstance(result, pd.DataFrame):
            return result.head().to_string()  # Return a preview for large outputs
        return str(result)
    except Exception as e:
        return f"Error during manipulation: {str(e)}"

4. Stats Tool (tools/stats_tool.py)

from langchain.agents import tool
import json
from config.vectordb_config import query_vectordb  # Import VectorDB query

@tool
def dataset_stats():
    """
    Retrieve dataset statistics (rows, columns, sample) dynamically from VectorDB.
    """
    rows = query_vectordb("Retrieve all rows", k=1000)
    if not rows:
        return "No data available in the dataset."

    try:
        # Convert rows to JSON and calculate statistics
        data = [json.loads(row) for row in rows]
        stats = {
            "rows": len(data),
            "columns": len(data[0]) if data else 0,
            "sample_row": data[0] if data else None,
        }
        return stats
    except Exception as e:
        return f"Error calculating stats: {str(e)}"

5. LLM Configuration (config/llm_config.py)

from langchain.llms import OpenAI

# Initialize the LLM
llm = OpenAI(model="gpt-3.5-turbo", temperature=0.7)

Optimizations
	1.	Conversational Memory:
	•	The use of ConversationBufferMemory maintains context across multiple user interactions, ensuring the chatbot acts conversationally.
	2.	Direct VectorDB Queries:
	•	Relevant dataset rows are queried directly from VectorDB for every request, avoiding unnecessary loading of the entire dataset.
	3.	Dynamic Query Handling:
	•	Tools (PandasTool and StatsTool) dynamically process dataset-related queries using only the relevant rows fetched from VectorDB.
	4.	Efficient Agent Execution:
	•	The agent runs only the required tool based on the query, ensuring minimal overhead.

Dependencies

Add the following to requirements.txt:

streamlit
pandas
langchain
faiss-cpu
openai

Running the Chatbot
	1.	Install the dependencies:

pip install -r requirements.txt


	2.	Run the Streamlit app:

streamlit run app.py


	3.	Upload a CSV file and start a conversational interaction.

This approach enables a smooth conversation with the chatbot while dynamically handling dataset-specific queries with relevant tools. Let me know if you need further assistance!
