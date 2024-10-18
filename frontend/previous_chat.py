import streamlit as st
import requests

# Page 3: Show Previous Chats
def page_previous_chats():

    # Fetch previous chat data from API (Replace with your actual API endpoint)
    response = requests.get("https://ragchat.ogesone.com/previous_chats")
    
    if response.status_code == 200:
        previous_chats = response.json()

        # Display previous chats grouped by session names
        for session_name, conversations in previous_chats.items():
            st.subheader(f"💬 Session: {session_name}")
            for message in conversations:
                if message['type'] == "human":
                    # Emoji for User (You)
                    st.markdown(f"""
                        <div style='display: flex; align-items: center; margin-bottom: 10px;'>
                            <span style='font-size: 1.5em; margin-right: 10px;'>👤</span>
                            <strong>&nbsp; </strong> {message['content']}
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    # Emoji for AI
                    st.markdown(f"""
                        <div style='display: flex; margin-bottom: 10px;'>
                            <span style='font-size: 1.5em; margin-right: 10px;'>🤖</span>
                        <div>{message['content']}</div>
                        </div>
                    """, unsafe_allow_html=True)

            st.write("---")  # Separator for each session
    else:
        st.error("Could not retrieve previous chats.")
