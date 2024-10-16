from fastapi import APIRouter, Request, Form, File, UploadFile
import warnings
from api.upload import *
import json
import os
import sys
import shutil  # For removing files in the directory

warnings.filterwarnings("ignore")
router = APIRouter()

# Define the directory to save uploaded files
UPLOAD_DIRECTORY = "./doc_storage"

# Function to break the file into chunks
def save_file_in_chunks(file, save_path, chunk_size=1024*1024):  # 1 MB chunk size by default
    with open(save_path, 'wb') as f:
        while chunk := file.read(chunk_size):
            f.write(chunk)

# Function to delete all files in a directory
def clear_upload_directory(directory):
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

@router.get("/")
def read_root():
    return {"Status": "Upload API working"}

@router.post("/processDoc", tags=['web'])
async def scrape_web(request: Request):
    '''Router for Vectorize Documents and store the embedding to pinecodesb'''
    try:
        request_body = await request.body()
        input_data = json.loads(request_body.decode('utf-8'))['data']
        data = doc_to_vectordb(input_data)
        return {"error": False, "message": "Data Updated Successfully"}
    except Exception as e:
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
        return {"error": True, "message": f"Some error occurred in: {str(e)}"}

@router.post("/uploadDoc", tags=['upload'])
async def upload_doc(file: UploadFile = File(...)):
    '''Upload any type of file and return saved directory path. File is broken down into chunks.'''
    try:
        if not os.path.exists(UPLOAD_DIRECTORY):
            os.makedirs(UPLOAD_DIRECTORY)
        clear_upload_directory(UPLOAD_DIRECTORY)
        save_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
        save_file_in_chunks(file.file, save_path)
        
        return {"error": False, "file_path": save_path}
    
    except Exception as e:
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
        return {"error": True, "message": f"Some error occurred in: {str(e)}"}
