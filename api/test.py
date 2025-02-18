import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, AIMessage, HumanMessage
from langchain.schema.runnable import RunnableWithMessageHistory

# Initialize memory
memory = ConversationBufferMemory(return_messages=True)

# Initialize LLM
llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)

# System prompt for LLM
SYSTEM_PROMPT = """
You are a strict assistant that first collects all required information before answering any questions.
The user can update their answers at any time.
Required information includes:
1. Name
2. Email address
3. Purpose of request
Once all details are collected, you may proceed with normal conversation.
"""

# Define agent logic
def agent_logic(inputs):
    messages = inputs.get("messages", [])
    
    # Let LLM decide the next step
    return llm.predict_messages(messages)

# Wrap with message history
agent = RunnableWithMessageHistory(
    agent_logic,
    lambda session_id: memory,
)

# Streamlit UI
st.title("AI Assistant - Collecting Required Info First")

# Initialize conversation state
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# Display chat history
for msg in st.session_state.conversation:
    role = "assistant" if msg["role"] == "assistant" else "human"
    st.chat_message(role).write(msg["content"])

# Handle user input
user_input = st.chat_input("Your response...")

if user_input:
    # Add user input to conversation
    st.session_state.conversation.append({"role": "human", "content": user_input})

    # Prepare message history
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        *[
            HumanMessage(content=msg["content"])
            if msg["role"] == "human"
            else AIMessage(content=msg["content"])
            for msg in st.session_state.conversation
        ],
    ]

    # Call the agent
    agent_response = agent.invoke(
        {"messages": messages},
        config={"configurable": {"session_id": "unique_session_id"}},
    )

    # Add agent response to conversation
    st.session_state.conversation.append({"role": "assistant", "content": agent_response.content})
    st.chat_message("assistant").write(agent_response.content)
