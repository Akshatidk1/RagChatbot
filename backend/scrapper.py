import requests
from bs4 import BeautifulSoup

def scrape_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.get_text()
    except Exception as e:
        return f"Error retrieving URL content: {e}"
