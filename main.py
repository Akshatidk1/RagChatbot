from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
from router.scraper import router as scraprouter
from router.chat import router as chatrouter
from router.upload import router as uploadrouter
import json
import os
from typing import Dict, List
app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Welcome to the Conversational RAG Chatbot API!"}
# Path to the chat history JSON file
CHAT_HISTORY_FILE = "./chat_storage/chat_history.json"

# Function to load chat history from the JSON file
def load_chat_history() -> Dict[str, List[Dict[str, str]]]:
    if not os.path.exists(CHAT_HISTORY_FILE):
        raise HTTPException(status_code=404, detail="Chat history file not found.")
    
    with open(CHAT_HISTORY_FILE, "r") as file:
        return json.load(file)

# API endpoint to get previous chats
@app.get("/previous_chats", response_model=Dict[str, List[Dict[str, str]]])
async def get_previous_chats():
    return load_chat_history()
class URLRequest(BaseModel):
    url: str
app.include_router(scraprouter, prefix="/web")
app.include_router(uploadrouter,prefix="/upload")
app.include_router(chatrouter,prefix="/rag")

# To run the application, use:
# uvicorn main:app --reload