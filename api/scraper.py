from bs4 import SoupStrainer 
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from utils.embedding import *
from utils.enviroment import *
from utils.pineconedb import *
import sys
import warnings
warnings.filterwarnings("ignore")
def web_to_vectordb(url_list):
    try:
        loader = WebBaseLoader(
            web_paths=url_list,
            bs_kwargs=dict(parse_only=SoupStrainer("p"))  # Target paragraph tags for main content
        )
        doc = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(doc)
        PineconeVectorStore.from_documents(splits,embeddings,index_name = index_name)
        return {"response":"Data Base Updated"}
    except Exception as e:
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
        return {"error": True, 'message': f'Error: {str(e)}'}