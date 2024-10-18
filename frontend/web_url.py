import streamlit as st
import requests

# Page 1: Enter Web URLs
def page_urls():
    # Section for entering web URLs
    url_list = []
    st.write("You can enter up to 5 URLs.")
    for i in range(5):
        url_input = st.text_input(f"Enter URL {i + 1}", key=f"url_{i}")
        if url_input:
            url_list.append(url_input)

    # Send URLs to the scraping API
    if st.button("Save"):
        if url_list:
            # Send request to scrapeWeb API
            payload = {"data": url_list}
            response = requests.post("https://ragchat.ogesone.com/web/scrapeWeb", json=payload)

            if response.status_code == 200:
                response_data = response.json()
                if response_data.get("error", True):
                    st.error("Error: Could not process the URLs.")
                else:
                    st.success("URLs have been saved and processed successfully!")
                    # st.write("Entered URLs:", url_list)
            else:
                st.error(f"Error: Could not submit URLs. Status code: {response.status_code}")
        else:
            st.error("Please enter at least one URL.")