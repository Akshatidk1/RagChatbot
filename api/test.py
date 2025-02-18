import json
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, AIMessage, HumanMessage
from langchain.schema.runnable import RunnableWithMessageHistory


# Load questions from JSON file
def load_questions():
    with open("questions.json", "r") as f:
        return json.load(f)


# Initialize the memory for conversation
memory = ConversationBufferMemory(return_messages=True)


# Create a LangChain Chat model (OpenAI)
llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)

# Create a simple agent function with RunnableWithMessageHistory
def create_agent_with_memory(llm, memory):
    def agent_logic(inputs):
        messages = inputs.get("messages", [])
        latest_message = messages[-1].content if messages else ""
        return llm.predict(latest_message)

    return RunnableWithMessageHistory(
        agent_logic,
        lambda session_id: memory,  # Use the same memory for all sessions
    )


agent = create_agent_with_memory(llm, memory)


# Streamlit UI
st.title("Streamlit Chatbot with LangChain Memory")

# Initialize session state for conversation and question progress
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "question_index" not in st.session_state:
    st.session_state.question_index = 0
if "answers" not in st.session_state:
    st.session_state.answers = {}

questions = load_questions()


# Function to handle asking questions and collecting answers
def ask_next_question():
    if st.session_state.question_index < len(questions):
        current_question = questions[st.session_state.question_index]
        st.session_state.conversation.append(
            {"role": "assistant", "content": current_question["question"]}
        )
        st.session_state.question_index += 1


# Display chat history
for msg in st.session_state.conversation:
    if msg["role"] == "assistant":
        st.chat_message("assistant").write(msg["content"])
    else:
        st.chat_message("human").write(msg["content"])


# Handle user input
user_input = st.chat_input("Your response...")
if user_input:
    # Add user input to conversation
    st.session_state.conversation.append({"role": "human", "content": user_input})

    # Store the answer
    current_question = questions[st.session_state.question_index - 1]
    st.session_state.answers[current_question["question"]] = user_input

    # Check if the question was required
    if current_question["priority"] == 1 and not user_input.strip():
        st.warning("This question is required. Please provide a valid answer.")
    else:
        # Move to the next question
        ask_next_question()

# After collecting all answers, call the agent with answers
if st.session_state.question_index >= len(questions):
    st.write("Thank you for providing all the required information!")
    st.write("Collected Answers:")
    st.write(st.session_state.answers)

    # Create a message history for the agent
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        *[
            HumanMessage(content=msg["content"])
            if msg["role"] == "human"
            else AIMessage(content=msg["content"])
            for msg in st.session_state.conversation
        ],
    ]

    # Call the agent with conversation history
    agent_response = agent.invoke(
        {"messages": messages},
        config={"configurable": {"session_id": "unique_session_id"}},
    )
    st.chat_message("assistant").write(agent_response)
