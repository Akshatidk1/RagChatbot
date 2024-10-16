from bs4 import SoupStrainer
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader
from langchain.document_loaders import PyPDFDirectoryLoader 
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from utils.embedding import *
from utils.enviroment import *
from utils.pineconedb import *
import sys
import warnings
warnings.filterwarnings("ignore")

UPLOAD_DIRECTORY = "./doc_storage"

def doc_to_vectordb(data):
    try:
        # Ensure the directory exists
        if not os.path.exists(UPLOAD_DIRECTORY):
            return {"error": True, 'message': f'Directory not found: {UPLOAD_DIRECTORY}'}

        docs = []  # To store all documents
        
        # Loop through files in the directory
        for filename in os.listdir(UPLOAD_DIRECTORY):
            file_path = os.path.join(UPLOAD_DIRECTORY, filename)
            if os.path.isfile(file_path):
                # Determine file type and use the appropriate loader
                if filename.endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                elif filename.endswith(".docx"):
                    loader = Docx2txtLoader(file_path)
                elif filename.endswith(".txt"):
                    loader = TextLoader(file_path)
                elif filename.endswith(".csv"):
                    loader = CSVLoader(file_path)
                else:
                    print(f"Unsupported file type for file: {filename}")
                    continue

                # Load the document
                doc = loader.load()
                docs.extend(doc)  # Add the loaded document to the list of docs

        if not docs:
            return {"error": True, 'message': "No valid documents found in the directory."}

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        # Save the split documents into the vector store
        PineconeVectorStore.from_documents(splits, embeddings, index_name=index_name)
        
        return {"error": False, 'message': "Data Updated Successfully"}
    except Exception as e:
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
        return {"error": True, 'message': f'Error: {str(e)}'}