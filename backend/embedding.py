from langchain.text_splitter import RecursiveCharacterTextSplitter
# The storage layer for the parent chunks
from langchain.storage import InMemoryStore
# load embedding model
from langchain.embeddings import HuggingFaceEmbeddings
# create vectorstore using Chromadb
from langchain.vectorstores import Chroma
# create retriever
from langchain.retrievers import ParentDocumentRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
# create document chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
# create retrieval chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.retrievers import ParentDocumentRetriever
# from langchain.prompts import ChatPromptTemplate
# from langchain.chains import create_stuff_documents_chain, create_retrieval_chain
# from langchain.document_loaders.base import InMemoryStore
# from langchain.text_splitters import RecursiveCharacterTextSplitter
# from langchain.llms import ChatGoogleGenerativeAI

class RagRetrival:
    def __init__(self, file_path, question):
        print(file_path)
        # Initialize an empty list to store all documents
        all_documents = []

        # Loop through each PDF file and load the documents
        for pdf_file in file_path:
            loader = PyPDFLoader(pdf_file)
            documents = loader.load()
            all_documents.extend(documents)
        self.documents = all_documents
        self.question = question
        
        # Create the parent and child document splitters
        self.parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
        self.child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

        # Create an in-memory store and vectorstore
        self.store = InMemoryStore()
        self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5", encode_kwargs={"normalize_embeddings": True})
        self.vectorstore = Chroma(collection_name="split_parents", embedding_function=self.embeddings)

        # Create the retriever
        self.retriever = ParentDocumentRetriever(
            vectorstore=self.vectorstore,
            docstore=self.store,
            child_splitter=self.child_splitter,
            parent_splitter=self.parent_splitter,
        )

        # Initialize the language model
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2
        )

        # Define the prompt template
        self.template = """
        You are an assistant for question-answering tasks.
        Use the provided context only to answer the following question:

        <context>
        {context}
        </context>

        Question: {input}
        """
        self.prompt = ChatPromptTemplate.from_template(self.template)
        self.doc_chain = create_stuff_documents_chain(self.llm, self.prompt)

    def format_documents(self):
        """Ensure documents are in the correct format with 'page_content' attribute."""
        formatted_documents = [{"page_content": doc} if isinstance(doc, str) else doc for doc in self.documents]
        return formatted_documents

    def adding_doc_vectorstore(self):
        # Ensure documents are correctly formatted
        formatted_documents = self.format_documents()

        # Add documents to the retriever
        self.retriever.add_documents(formatted_documents)

        # Create the full retrieval chain
        chain_all = create_retrieval_chain(self.retriever, self.doc_chain)
        return chain_all

    def ask_llm(self):
        # Add documents and create retrieval chain
        chain = self.adding_doc_vectorstore()

        # Ask the language model using the created chain
        response = chain.invoke({"input": f"{self.question}"})
        answer = response['answer']
        return answer


def search_embeddings():
    return
def store_embeddings():
    return