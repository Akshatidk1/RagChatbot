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
            response = requests.post("https://akshatchatbot.ogesone.com/web/scrapeWeb", json=payload)

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
            response = requests.post("https://akshatchatbot.ogesone.com/rag/chat", json=payload)

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
    response = requests.get("https://akshatchatbot.ogesone.com/previous_chats")
    
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

# Page 4: Upload Document and Process
def page_upload_doc():
    st.title("Upload Document")

    # Upload file
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "txt", "csv", "xlsx"])

    if uploaded_file is not None:
        # Send file to upload API
        with st.spinner("Uploading file..."):
            files = {"file": uploaded_file}
            response = requests.post("https://akshatchatbot.ogesone.com/upload/uploadDoc", files=files)
            
            if response.status_code == 200:
                response_data = response.json()
                saved_path = response_data.get("file_path", "")
                st.success(f"File uploaded successfully! Saved at: {saved_path}")

                # Now process the document with the `/processDoc` endpoint
                with st.spinner("Processing document..."):
                    process_payload = {"data": saved_path}
                    process_response = requests.post("https://akshatchatbot.ogesone.com/upload/processDoc", json=process_payload)

                    if process_response.status_code == 200:
                        st.success("Document processed successfully!")
                        st.write("Response from API:", process_response.json())
                    else:
                        st.error(f"Error: Could not process the document. Status code: {process_response.status_code}")
            else:
                st.error(f"Error: Could not upload the file. Status code: {response.status_code}")

# Main navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Enter Web URLs", "Chat", "Previous Chats", "Upload Document"])

if page == "Enter Web URLs":
    page_urls()
elif page == "Chat":
    page_chat()
elif page == "Previous Chats":
    page_previous_chats()
else:
    page_upload_doc()
