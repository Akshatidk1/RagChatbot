from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from api.chat import ConversationalRAGChatbot  # Make sure to import the chatbot class
import logging

router = APIRouter()

# Define the request body using Pydantic
class ChatRequest(BaseModel):
    session_id: str
    user_input: str

# Define the chat endpoint
@router.post("/chat")
def chat_endpoint(request: ChatRequest):
    try:
        # Instantiate the chatbot
        chatbot = ConversationalRAGChatbot()
        # Call the chat_with_llm method and get the AI's response
        response = chatbot.chat_with_llm(request.session_id, request.user_input)
        
        # Return the response as a JSON object
        return {"response": response}  # Change this to response directly
    except Exception as e:
        logging.error(f"Error in chat_endpoint: {e}")  # Log the error for debugging
        raise HTTPException(status_code=500, detail=str(e))
