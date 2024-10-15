import json
import os
import logging
import warnings
from utils.embedding import *
from utils.pineconedb import *
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
import pinecone
from langchain.vectorstores import Pinecone
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)

class ConversationalRAGChatbot:
    def __init__(self, history_file="./chat_storage/chat_history.json"):
        logging.info("Initializing ConversationalRAGChatbot")

        # Initialize model and retriever
        self.vector_store = self.initialize_vector_store()
        self.retriever = self.initialize_retriever()
        self.model = self.initialize_model()

        # Set up prompts and chains
        self.setup_prompts_and_chains()

        # Chat history storage
        self.history_file = history_file
        self.store = self.load_chat_history()
        logging.info("Chat history loaded successfully")

    def initialize_vector_store(self):
        """Initialize the Pinecone vector store."""
        try:
            vector_store = Pinecone.from_existing_index(index_name, embeddings)
            logging.info("PineconeVectorStore initialized successfully")
            return vector_store
        except Exception as e:
            logging.error(f"Failed to initialize PineconeVectorStore: {e}")
            raise

    def initialize_retriever(self):
        """Create and return the retriever."""
        try:
            retriever = self.vector_store.as_retriever()
            logging.info(f"Retriever type: {type(retriever)}")  # Confirm it's a retriever
            logging.info("Retriever created successfully")
            return retriever
        except Exception as e:
            logging.error(f"Failed to create retriever: {e}")
            raise

    def initialize_model(self):
        """Initialize the ChatGoogleGenerativeAI model."""
        try:
            model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", convert_system_message_to_human=True)
            logging.info("ChatGoogleGenerativeAI model initialized successfully")
            return model
        except Exception as e:
            logging.error(f"Failed to initialize ChatGoogleGenerativeAI: {e}")
            raise

    def setup_prompts_and_chains(self):
        """Set up prompts and chains for the chatbot."""
        try:
            # System and retriever prompts
            self.system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer the question. "
                "If you don't know the answer, say that you don't know. "
                "Use three sentences maximum and keep the answer concise."
                "\n\n{context}"
            )

            self.chat_prompt = ChatPromptTemplate.from_messages(
                [("system", self.system_prompt), ("human", "{input}")]
            )
            logging.info("Chat prompts initialized successfully")

            self.retriever_prompt = (
                "Given a chat history and the latest user question which might reference context in the chat history, "
                "formulate a standalone question which can be understood without the chat history. "
                "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
            )

            self.contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [("system", self.retriever_prompt), MessagesPlaceholder(variable_name="chat_history"), ("human", "{input}")]
            )

            self.history_aware_retriever = create_history_aware_retriever(self.model, self.retriever, self.contextualize_q_prompt)

            self.qa_prompt = ChatPromptTemplate.from_messages(
                [("system", self.system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")]
            )

            self.question_answer_chain = create_stuff_documents_chain(self.model, self.qa_prompt)
            self.rag_chain = create_retrieval_chain(self.history_aware_retriever, self.question_answer_chain)

            logging.info("Retrieval and QA chains set up successfully")
        except Exception as e:
            logging.error(f"Error setting up prompts and chains: {e}")
            raise

    def load_chat_history(self):
        """Load chat history from a JSON file."""
        if os.path.exists(self.history_file):
            try:
                if os.stat(self.history_file).st_size == 0:
                    logging.info("Chat history file is empty, returning an empty dictionary")
                    return {}

                with open(self.history_file, "r") as f:
                    chat_history_data = json.load(f)
                    logging.info("Chat history loaded from file")
                    return self._convert_to_chat_message_history(chat_history_data)

            except json.JSONDecodeError:
                logging.error("Failed to decode chat history, file may be corrupted or improperly formatted.")
                return {}

        logging.info("No chat history file found, starting with an empty history")
        return {}

    def _convert_to_chat_message_history(self, data):
        """
        Convert a dictionary of serialized chat histories back into ChatMessageHistory objects.
        """
        history = {}
        for session_id, messages in data.items():
            chat_history = ChatMessageHistory()
            for message in messages:
                if message["type"] == "human":
                    chat_history.add_message(HumanMessage(content=message["content"]))
                elif message["type"] == "ai":
                    chat_history.add_message(AIMessage(content=message["content"]))
            history[session_id] = chat_history
        return history

    def save_chat_history(self):
        """Save chat history to a JSON file, handling potential serialization issues."""
        try:
            serializable_store = self._make_serializable(self.store)

            with open(self.history_file, "w") as f:
                json.dump(serializable_store, f, indent=4)
                logging.info("Chat history saved to file")
        except (TypeError, json.JSONDecodeError) as e:
            logging.error(f"Failed to save chat history: {e}")
            raise

    def _make_serializable(self, data):
        """
        Recursively process the data structure to ensure it's serializable.
        Convert objects like `ChatMessageHistory` to basic serializable types (e.g., lists of dictionaries).
        """
        if isinstance(data, ChatMessageHistory):
            return [self._message_to_dict(message) for message in data.messages]

        if isinstance(data, dict):
            return {key: self._make_serializable(value) for key, value in data.items()}

        if isinstance(data, list):
            return [self._make_serializable(item) for item in data]

        return data

    def _message_to_dict(self, message):
        """
        Convert a message (HumanMessage or AIMessage) to a serializable dictionary.
        """
        if isinstance(message, HumanMessage):
            return {"type": "human", "content": message.content}
        elif isinstance(message, AIMessage):
            return {"type": "ai", "content": message.content}
        else:
            logging.warning(f"Unexpected message type: {type(message)}")
            return {"type": "unknown", "content": str(message)}

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """Get or create chat history for a session."""
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
            logging.info(f"New session history created for session_id: {session_id}")
        return self.store[session_id]

    def chat_with_llm(self, session_id: str, user_input: str):
        """Run the chat conversation with the LLM, storing history."""
        logging.info(f"Processing chat with session_id: {session_id} and user_input: {user_input}")

        # Create a callable to return chat history
        history_callable = lambda: self.get_session_history(session_id)

        conversational_rag_chain = RunnableWithMessageHistory(
            self.rag_chain,
            history_callable,  # Pass callable for history
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        try:
            logging.info("conversational_rag_chain invoke started")
            response = conversational_rag_chain.invoke({"input": user_input},
                config={
                    "configurable": {"session_id": session_id}
                })
            logging.info(f"Response from LLM: {response['answer']}")

            # Ensure to correctly access the response content
            if isinstance(response, dict) and 'answer' in response:
                answer_content = response['answer']
            else:
                logging.error(f"Unexpected response format: {response}")
                answer_content = "Sorry, an error occurred while processing your request."
            logging.info(f"Response generated: {answer_content}")

            # Save chat history after each response
            self.save_chat_history()
            return answer_content

        except Exception as e:
            logging.error(f"Error during chat: {e}")
            return "Sorry, an error occurred while processing your request."
