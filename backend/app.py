from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
from backend.doc_processing import process_document
from backend.scrapper import scrape_url
# from backend.query_handler import handle_query
from backend.embedding import RagRetrival
# set api openai key
import os
os.environ["GOOGLE_API_KEY"] = "AIzaSyAIujOVkqnEXrz7Yj1ztWVfdw_RVWwwVGw"
app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    sources: list

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Endpoint to save the uploaded document."""
    try:
        # Read the file content once
        file_content = await file.read()

        # Define the save directory and create it if it doesn't exist
        save_directory = "./backend/uploaded_files"
        os.makedirs(save_directory, exist_ok=True)

        # Use file.filename to get the actual filename
        file_location = os.path.join(save_directory, file.filename)

        # Save the file content
        with open(file_location, "wb") as file_object:
            file_object.write(file_content)
        # Process the saved file (for example, extract text from a PDF)
        file_text = process_document(file_location)
        print(file_text)
        return {"status": "success", "content": file_text}
    except Exception as e:
        return {"error": True, "message": f"Some error occurred: {str(e)}"}
    

@app.post("/scrape")
async def scrape_web_url(url: str):
    content = scrape_url(url)
    return {"status": "success", "content": content}

@app.post("/query")
async def query_response(query_request: QueryRequest):
    print("here")
    print(query_request)
    rag = RagRetrival(query_request.sources,query_request.query)
    response = rag.ask_llm()
    return {"response": response}
