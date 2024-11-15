import streamlit as st
import requests
import streamlit.components.v1 as components

# CSS for chat input at the bottom
st.markdown("""
    <style>
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        border: 1px solid #ccc;
        padding: 10px;
        background-color: #f9f9f9;
        border-radius: 10px;
    }
    .stTextInput {
        position: fixed;
        bottom: 20px;
        width: calc(100% - 20px);
    }
    .message-row {
        display: flex;
        align-items: flex-start;
        margin-bottom: 10px;
    }
    .user-message {
        background-color: #DCF8C6;
        padding: 8px;
        border-radius: 8px;
        max-width: 80%;
    }
    .ai-message {
        background-color: #F1F0F0;
        padding: 8px;
        border-radius: 8px;
        max-width: 80%;
    }
    .profile-emoji {
        font-size: 2em;
        margin-right: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# JavaScript to scroll to the bottom of the chat
scroll_js = """
    <script>
    function scrollToBottom() {
        var chatDiv = document.getElementById('chat-container');
        chatDiv.scrollTop = chatDiv.scrollHeight;
    }
    setTimeout(scrollToBottom, 100);
    </script>
"""

# Page 2: Chat with AI
def page_chat():
    # Prompt for session name
    session_name = st.text_input("Enter your Session Name:", key="session_name_input")

    # Session state to store chat history
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []

    # Only show the chat interface if the session name is provided
    if session_name:
        chat_container = st.empty()  # Placeholder for chat messages
        
        # Function to display chat messages dynamically
        def render_chat():
            with chat_container.container():
                st.markdown("<div class='chat-container' id='chat-container'>", unsafe_allow_html=True)

                # Display the chat history with emojis as profile indicators
                for message in st.session_state['messages']:
                    if message['role'] == 'user':
                        st.markdown(f"""
                            <div class="message-row">
                                <span class="profile-emoji">👤</span>
                                <div class="user-message">{message['text']}</div>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                            <div class="message-row">
                                <span class="profile-emoji">🤖</span>
                                <div class="ai-message">{message['text']}</div>
                            </div>
                        """, unsafe_allow_html=True)

                st.markdown("", unsafe_allow_html=True)
                components.html(scroll_js, height=0)  

        # Initially render chat messages
        render_chat()

        # Input field for user message
        user_input = st.text_input("You: ", key="user_input_text")

        if user_input:
            # Save user input to session state
            st.session_state['messages'].append({"role": "user", "text": user_input})

            # Re-render the chat immediately to show user input
            render_chat()

            # Function to send message to FastAPI
            payload = {"session_id": session_name, "user_input": user_input}
            response = requests.post("http://127.0.0.1:8000/rag/chat", json=payload)

            if response.status_code == 200:
                response_data = response.json()
                ai_response = response_data.get("response", "")
            else:
                ai_response = "Error: Could not retrieve response from AI."

            # Save AI response to session state
            st.session_state['messages'].append({"role": "ai", "text": ai_response})

            # Re-render the chat after AI response
            render_chat()

    else:
        st.warning("Please enter a Session Name to start chatting.")

# Call the page chat function to render the page
page_chat()
