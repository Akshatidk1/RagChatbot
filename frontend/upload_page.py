import streamlit as st
import requests

# Page 4: Upload Document and Process
def page_upload_doc():
    # Upload file
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "txt", "xlsx", "pptx"])

    if uploaded_file is not None:
        # Send file to upload API
        with st.spinner("Uploading file..."):
            files = {"file": uploaded_file}
            response = requests.post("http://127.0.0.1:8000/upload/uploadDoc", files=files)

            if response.status_code == 200:
                response_data = response.json()
                if response_data.get("error", True):
                    st.error("Error: Could not upload the file.")
                else:
                    saved_path = response_data.get("file_path", "")
                    st.success(f"File uploaded successfully! Saved at: {saved_path}")

                    # Now process the document with the `/processDoc` endpoint
                    with st.spinner("Processing document..."):
                        process_payload = {"data": saved_path}
                        process_response = requests.post("http://127.0.0.1:8000/upload/processDoc", json=process_payload)

                        if process_response.status_code == 200:
                            process_data = process_response.json()
                            if process_data.get("error", True):
                                st.error("Error: Could not process the document.")
                            else:
                                st.success("Document processed successfully!")
                                st.write("Response from API:", process_data)
                        else:
                            st.error(f"Error: Could not process the document. Status code: {process_response.status_code}")
            else:
                st.error(f"Error: Could not upload the file. Status code: {response.status_code}")