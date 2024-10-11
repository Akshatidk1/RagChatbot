import streamlit as st
import requests

st.title("RAG Chatbot")
st.write("Upload files or enter URLs to ask questions.")

# File uploader for multiple file types
uploaded_files = st.file_uploader("Upload Files", type=['pdf', 'docx', 'xlsx', 'pptx', 'txt'], accept_multiple_files=True)

# Text area for URLs
web_urls = st.text_area("Enter URLs (comma-separated)")
# Input for user's question
query = st.text_input("Ask your question")

if st.button("Submit"):
    sources = []

    # Upload files to the backend
    for uploaded_file in uploaded_files:
        response = requests.post("http://127.0.0.1:8000/upload", files={"file": uploaded_file})

        # Check response status
        if response.status_code == 200:
            try:
                print("here at this ")
                print(response)
                # source = response.json()['content']
                sources.append(response.json()['content'])
            except KeyError:
                st.error("Error: 'content' not found in the response.")
        else:
            st.error(f"Failed to upload {uploaded_file.name}: {response.status_code}, {response.text}")
    
    # Scrape URLs if provided
    if web_urls:
        urls = [url.strip() for url in web_urls.split(',')]  # Clean up spaces
        for url in urls:
            response = requests.post("http://127.0.0.1:8000/scrape", data={"url": url})

            # Check response status
            if response.status_code == 200:
                try:
                    # source = response.json()['content']
                    sources.append(response.json()['content'][0])
                except KeyError:
                    st.error(f"Error: 'content' not found in the response for URL {url}.")
            else:
                st.error(f"Failed to scrape {url}: {response.status_code}, {response.text}")
    
    # Submit query if both query and sources are provided
    if query and sources:
        print(sources)
        response = requests.post("http://127.0.0.1:8000/query", json={"query": query, "sources": sources})
        
        # Check response status
        if response.status_code == 200:
            st.write(response.json()['response'])
        else:
            st.error(f"Failed to get response for the query: {response.status_code}, {response.text}")
    else:
        st.warning("Please enter a question and upload files or URLs to get a response.")
