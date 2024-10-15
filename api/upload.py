from bs4 import SoupStrainer
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



def doc_to_vectordb(directory):
    try:
        file_loader=PyPDFDirectoryLoader(directory)
        doc = file_loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(doc)
        PineconeVectorStore.from_documents(splits,embeddings,index_name = index_name)
    except Exception as e:
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
        return {"error": True, 'message': f'Error: {str(e)}'}