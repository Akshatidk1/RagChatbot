import streamlit as st  

# Initialize session state
if "current_task" not in st.session_state:
    st.session_state.current_task = ""
if "agent_answer" not in st.session_state:
    st.session_state.agent_answer = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# App Layout
st.set_page_config(page_title="Code Generator by Autogen", layout="wide")

# Main UI
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Code Generator by Autogen</h1>", unsafe_allow_html=True)

# Task Input Section
task_input = st.text_area("Enter Task:", placeholder="Describe the coding task here...", height=100)
submit = st.button("Submit Task")

# Handle Submission
if submit and task_input.strip():
    st.session_state.current_task = task_input
    st.session_state.agent_answer = f"Agent is generating code for: {task_input}"  # Placeholder for real agent response
    st.session_state.chat_history.append(f"User: {task_input}")
    st.session_state.chat_history.append(f"Agent: {st.session_state.agent_answer}")

# Display Current Task
st.subheader("📌 Current Task")
st.info(st.session_state.current_task if st.session_state.current_task else "No task submitted yet.")

# Display Agent's Answer
st.subheader("🤖 Agent's Answer")
st.success(st.session_state.agent_answer if st.session_state.agent_answer else "Waiting for task input...")

# Popover for Chat History
with st.sidebar:
    st.subheader("💬 Chat History")
    for chat in st.session_state.chat_history:
        st.write(chat)
