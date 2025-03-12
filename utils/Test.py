from fastapi import FastAPI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.chat_models import ChatOpenAI
from langchain.memory import RedisChatMessageHistory, ConversationBufferMemory
import requests
import redis
import json

# FastAPI App
app = FastAPI()

# Redis Setup for Shared Memory
redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Function to generate payload using LLM
def generate_payload(document_type, user_answers):
    """LLM generates structured JSON payload."""
    prompt = f"""
    Generate a structured JSON payload for a {document_type.upper()} document 
    based on these answers: {json.dumps(user_answers, indent=2)}.
    """
    return llm.invoke(prompt).content

# Dummy API to Simulate S3 URL Generation
def send_to_api(payload):
    """Simulate sending payload to an API and returning a document URL."""
    response = requests.post("https://dummy-api.com/generate", json=payload)
    return response.json().get("document_url", "https://s3-bucket.com/sample_doc.pdf")

# Tool for Interface Specification
def interface_tool(user_answers):
    """Handles Interface FS creation."""
    payload = generate_payload("Interface", user_answers)
    document_url = send_to_api(payload)
    return f"Your Interface Specification document is ready: {document_url}"

# Tool for Enhancement Specification
def enhancement_tool(user_answers):
    """Handles Enhancement FS creation."""
    payload = generate_payload("Enhancement", user_answers)
    document_url = send_to_api(payload)
    return f"Your Enhancement Specification document is ready: {document_url}"

# Tool for Report Specification
def report_tool(user_answers):
    """Handles Report FS creation."""
    payload = generate_payload("Report", user_answers)
    document_url = send_to_api(payload)
    return f"Your Report Specification document is ready: {document_url}"

# Registering Tools
tools = [
    Tool(name="Interface Specification", func=interface_tool, description="Handles Interface FS generation."),
    Tool(name="Enhancement Specification", func=enhancement_tool, description="Handles Enhancement FS generation."),
    Tool(name="Report Specification", func=report_tool, description="Handles Report FS generation."),
]

# Initialize LangChain Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True),
    verbose=True
)

@app.post("/chat")
def chat_with_agent(session_id: str, user_input: str):
    """Handles user input and maintains conversation context."""
    # Retrieve stored answers from Redis
    stored_data = redis_client.get(session_id)
    user_answers = json.loads(stored_data) if stored_data else {}

    # Determine document type
    if "document_type" not in user_answers:
        if user_input.lower() in ["interface", "enhancement", "report"]:
            user_answers["document_type"] = user_input.lower()
            response = f"Great! Let's collect details for the {user_answers['document_type']} document."
        else:
            return {"response": "Which document do you want to generate? (Interface, Enhancement, Report)"}
    else:
        # Collect answers dynamically
        doc_type = user_answers["document_type"]
        sections = {
            "interface": ["Document Name", "Objective Overview", "Functional Requirement", "Processing Logic", "Transfer Method"],
            "enhancement": ["Document Title", "Use Case Benefit", "Functional Design", "Data Flow Diagram"],
            "report": ["Document Title", "Use Case Benefit", "Functional Design", "Report Input"]
        }

        pending_sections = [s for s in sections[doc_type] if s not in user_answers]

        if pending_sections:
            next_question = pending_sections[0]
            user_answers[next_question] = user_input
            response = f"Got it! Now, please provide {next_question}."
        else:
            # Call the appropriate tool dynamically
            tool_result = agent.run(f"Generate {doc_type} document with answers: {json.dumps(user_answers)}")
            response = tool_result

    # Store updated answers in Redis
    redis_client.set(session_id, json.dumps(user_answers))

    return {"response": response}
    
