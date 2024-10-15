from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from api.test import testApiFunction

api = APIRouter()

# ... (Rest of your code from the previous response)

# Additional routes (if needed)
@api.get("/healthcheck")
def healthcheck():
    return {"status": "OK"}

@api.get("/about")
def about():
    return {"message": "This is a sample API"}

@api.get("/test")
def test():
    try:
        # Call the function and get the result
        result = testApiFunction()        
        if result:
            return result 
        else:
            raise HTTPException(status_code=500, detail="Invalid response from testApiFunction")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in /test route: {str(e)}")
