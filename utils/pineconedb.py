# Initialize Pinecone
from pinecone import Pinecone
import warnings
warnings.filterwarnings("ignore")
vectordb = Pinecone(api_key="4158a78f-e1ed-4f88-9ad6-923ee500456e", environment="us-east-1")

index_name = "ragchatbot"