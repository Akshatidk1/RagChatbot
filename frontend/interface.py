import streamlit as st
import requests

# Page 1: Enter Web URLs
def page_urls():
    st.title("Enter Web URLs")

    # Section for entering web URLs
    url_list = []
    st.write("You can enter up to 5 URLs.")
    for i in range(5):
        url_input = st.text_input(f"Enter URL {i + 1}", key=f"url_{i}")
        if url_input:
            url_list.append(url_input)

    # Optional: Token for API
    token = st.text_input("Enter Token (Optional)", type="password", key="token_input")

    # Send URLs to the scraping API
    if st.button("Save"):
        if url_list:
            # Send request to scrapeWeb API
            payload = {"data": url_list, "token": token}
            response = requests.post("http://127.0.0.1:8000/web/scrapeWeb", json=payload)

            if response.status_code == 200:
                st.success("URLs have been saved and processed!")
                st.write("Entered URLs:", url_list)
                st.write("Response from API:", response.json())
            else:
                st.error(f"Error: Could not submit URLs. Status code: {response.status_code}")
        else:
            st.error("Please enter at least one URL.")

# Page 2: Chat with FastAPI
def page_chat():
    st.title("Chat with AI")

    # Prompt for session name
    session_name = st.text_input("Enter your Session Name:", key="session_name")

    # Session state to store chat history
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []

    # Only show the chat interface if the session name is provided
    if session_name:
        # Input field for user message
        user_input = st.text_input("You: ", key="user_input")

        if user_input:
            # Save user input to session state
            st.session_state['messages'].append({"role": "user", "text": user_input})

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

        # Display the chat history
        st.write("### Chat History")
        for message in st.session_state['messages']:
            if message['role'] == 'user':
                st.write(f"**You**: {message['text']}")
            else:
                st.write(f"**AI**: {message['text']}")
    else:
        st.warning("Please enter a Session Name to start chatting.")

# Page 3: Show Previous Chats
def page_previous_chats():
    st.title("Previous Chats")

    # Fetch previous chat data from API (Replace with your actual API endpoint)
    response = requests.get("http://127.0.0.1:8000/previous_chats")
    
    if response.status_code == 200:
        previous_chats = response.json()

        # Display previous chats grouped by session names
        for session_name, conversations in previous_chats.items():
            st.subheader(session_name)
            for message in conversations:
                if message['type'] == "human":
                    st.write(f"**You**: {message['content']}")
                else:
                    st.write(f"**AI**: {message['content']}")
            st.write("---")  # Separator for each session
    else:
        st.error("Could not retrieve previous chats.")

# Main navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Enter Web URLs", "Chat", "Previous Chats"])

if page == "Enter Web URLs":
    page_urls()
elif page == "Chat":
    page_chat()
else:
    page_previous_chats()
