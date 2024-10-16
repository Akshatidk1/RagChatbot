from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from api.chat import ConversationalRAGChatbot  # Make sure to import the chatbot class
import logging

router = APIRouter()

class ChatRequest(BaseModel):
    session_id: str
    user_input: str


@router.post("/chat")
def chat_endpoint(request: ChatRequest):
    try:
        # Instantiate the chatbot
        chatbot = ConversationalRAGChatbot()
        response = chatbot.chat_with_llm(request.session_id, request.user_input)
        return {"response": response}  # Change this to response directly
    except Exception as e:
        logging.error(f"Error in chat_endpoint: {e}") 
        raise HTTPException(status_code=500, detail=str(e))
