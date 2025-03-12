import os
import redis
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from langchain_openai import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentType, initialize_agent
from langchain.tools import Tool

# Environment Variables (Set your OpenAI API key)
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# Initialize Redis
redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

# FastAPI app
app = FastAPI()

# Define a model for chat request
class ChatRequest(BaseModel):
    session_id: str
    user_message: str

# Function to retrieve agent memory per session
def get_agent(session_id: str):
    """Fetch or create a LangChain agent with Redis-based memory for a given session."""
    
    memory_key = f"session:{session_id}:memory"

    # Retrieve previous conversation from Redis
    past_conversation = redis_client.get(memory_key)
    
    # Initialize memory with past conversation (if exists)
    memory = ConversationBufferMemory(memory_key=memory_key)
    if past_conversation:
        memory.chat_memory.add_message(past_conversation)

    # Define the LLM (GPT-4)
    llm = OpenAI(temperature=0)

    # Define tools (customize as needed)
    tools = [
        Tool(
            name="EchoTool",
            func=lambda query: f"You said: {query}",
            description="Echoes back user input"
        )
    ]

    # Initialize the agent
    agent = initialize_agent(
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        tools=tools,
        llm=llm,
        memory=memory,
        verbose=True
    )

    return agent, memory

# Chat endpoint
@app.post("/chat/")
async def chat(request: ChatRequest):
    """Handles user messages and returns chatbot responses."""
    
    session_id = request.session_id
    user_message = request.user_message

    # Get agent and memory for the session
    agent, memory = get_agent(session_id)

    try:
        # Generate response
        response = agent.run(user_message)

        # Store updated conversation in Redis
        redis_client.set(memory.memory_key, memory.chat_memory.messages)

        return {"session_id": session_id, "response": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run with: uvicorn script_name:app --reload
