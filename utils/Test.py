import streamlit as st
import pandas as pd

# Inject custom CSS for chat bubbles with avatars
st.markdown(
    """
    <style>
    /* Container for each chat message */
    .chat-container {
        margin: 10px 0;
        display: flex;
        flex-direction: row;
        align-items: flex-end;
    }
    /* For manager messages, reverse the flex direction so the avatar is on the right */
    .chat-container.manager {
        flex-direction: row-reverse;
    }
    /* Avatar styling */
    .chat-avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        overflow: hidden;
        margin: 0 10px;
    }
    .chat-avatar img {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }
    /* Content container for the chat bubble */
    .chat-content {
        max-width: 70%;
        display: flex;
        flex-direction: column;
    }
    /* Chat bubble styling */
    .chat-bubble {
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        word-wrap: break-word;
    }
    /* Styling for agent messages */
    .agent .chat-bubble {
        background-color: #DCF8C6;
    }
    /* Styling for manager messages */
    .manager .chat-bubble {
        background-color: #FFF3CD;
    }
    /* Sender label styling */
    .sender-label {
        font-weight: bold;
        margin-bottom: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Agent Chat Manager UI with Avatars")

# File uploader for the CSV file
uploaded_file = st.file_uploader("Upload your chat CSV", type=["csv"])

if uploaded_file is not None:
    # Read CSV into a DataFrame
    df = pd.read_csv(uploaded_file)
    
    # Validate required columns: 'name' and 'content'
    if "name" not in df.columns or "content" not in df.columns:
        st.error("CSV must contain 'name' and 'content' columns. Optionally, include 'avatar' column.")
    else:
        st.write("### Conversation:")
        
        # Iterate over each row (assuming CSV rows are in conversation order)
        for idx, row in df.iterrows():
            sender = row["name"]
            message = row["content"]
            # Use the 'avatar' column if present and not null; otherwise, use a default image.
            if "avatar" in df.columns and pd.notnull(row["avatar"]):
                avatar_url = row["avatar"]
            else:
                avatar_url = "https://via.placeholder.com/40"
            
            # Determine CSS class based on sender (manager messages will be styled differently)
            css_class = "manager" if sender.strip().lower() == "manager" else "agent"
            
            # Render the chat message as an HTML block
            st.markdown(
                f"""
                <div class="chat-container {css_class}">
                    <div class="chat-avatar">
                        <img src="{avatar_url}" alt="{sender} avatar">
                    </div>
                    <div class="chat-content">
                        <div class="sender-label">{sender}</div>
                        <div class="chat-bubble">{message}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
