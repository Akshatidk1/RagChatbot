import streamlit as st
import requests
from web_url import *
from chat_page import *
from previous_chat import *
from upload_page import *

# Apply custom styling with CSS
st.markdown("""
    <style>
    /* Main page styling */
    .main {
        background-color: #F0F2F6;
        padding: 20px;
    }
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #292b2c;
        color: white;
    }
    /* Title and header styles */
    h1, h2 {
        color: #1F4E79;
        font-family: 'Segoe UI', sans-serif;
    }
    /* Button styles */
    .stButton button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 8px;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    /* Expander styling */
    .streamlit-expanderHeader {
        font-size: 16px;
        font-weight: bold;
        color: #1F4E79;
    }
    /* Footer styling */
    footer {
        font-size: 12px;
        text-align: center;
        color: #888;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("Navigation")
st.sidebar.markdown("Choose a section to navigate to:")
page = st.sidebar.selectbox("Choose a page", ["Enter Web URLs", "Chat", "Previous Chats", "Upload Document"])

# Add a logo or image (optional) for a more polished UI
# st.sidebar.image("./assets/logo.png", use_column_width=True) 

# Display custom footer in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("Built 💻 by Akshat Nishad for katonic.ai")

# Enhanced Page 1: Enter Web URLs
if page == "Enter Web URLs":
    st.title("🌐 Enter Web URLs")
    st.subheader("Add up to 5 URLs and submit for processing")
    page_urls()

# Enhanced Page 2: Chat with AI
elif page == "Chat":
    st.title("💬 Chat with AI")
    st.subheader("Start a session and chat with the AI")
    with st.expander("Chat Instructions"):
        st.write("""
        - Enter a session name to start chatting.
        - Your chat history will be saved under the session name.
        """)
    page_chat()

# Enhanced Page 3: Previous Chats
elif page == "Previous Chats":
    st.title("📜 Previous Chats")
    st.subheader("Review your past conversations")
    st.markdown("Here you can browse through all previously saved chat sessions.")
    page_previous_chats()

# Enhanced Page 4: Upload Document
else:
    st.title("📄 Upload Document")
    st.subheader("Upload and process files (PDF, DOCX, TXT, XLSX, PPTX)")
    page_upload_doc()

