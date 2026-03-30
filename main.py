from typing import List, Dict, Optional
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_classic.memory import ConversationBufferMemory
import config
from utils.embeddings import get_embeddings
from utils.llm import get_llm


# Custom prompt template for RAG
RAG_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based on the provided context.

Use the following pieces of context to answer the question at the end. If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer the question clearly and concisely based only on the context provided above:"""

RAG_PROMPT = PromptTemplate(
    template=RAG_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)


class RAGChatbot:

    def __init__(self, runtime_keys=None):
        """Initialize the RAG chatbot

        Args:
            runtime_keys: Optional dict with API keys from user input
        """
        self.vectorstore = None
        self.retriever = None
        self.llm = None
        self.qa_chain = None
        self.runtime_keys = runtime_keys or {}
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

    def load_vectorstore(self):
        embeddings = get_embeddings(self.runtime_keys)

        self.vectorstore = FAISS.load_local(
            config.VECTORSTORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": config.TOP_K}
        )

        print(f"[SUCCESS] Vector store loaded successfully")

    def initialize_llm(self):
        self.llm = get_llm(self.runtime_keys)
        print(f"LLM initialized successfully")

    def setup_qa_chain(self):
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            return_source_documents=True,
            verbose=False
        )

        print(f"[SUCCESS] QA Chain initialized")

    def initialize(self):
        print("\nInitializing RAG Chatbot...")
        self.load_vectorstore()
        self.initialize_llm()
        self.setup_qa_chain()
        print("[SUCCESS] RAG Chatbot ready!\n")

    def query(self, question: str) -> Dict:
        if not self.qa_chain:
            raise ValueError("QA chain not initialized. Call initialize() first.")

        # Get response
        response = self.qa_chain.invoke({"question": question})

        return {
            "answer": response["answer"],
            "source_documents": response.get("source_documents", [])
        }

    def query_simple(self, question: str) -> str:
        result = self.query(question)
        return result["answer"]

    def get_relevant_documents(self, query: str) -> List:
        if not self.retriever:
            raise ValueError("Retriever not initialized. Call initialize() first.")

        docs = self.retriever.invoke(query)
        return docs

    def clear_memory(self):
        self.memory.clear()
        print("Conversation memory cleared")


def test_rag_pipeline():
    try:
        config.validate_config()

        # Initialize chatbot
        bot = RAGChatbot()
        bot.initialize()

        # Test query
        test_question = "What is this document about?"
        print(f"\nTest Question: {test_question}")

        result = bot.query(test_question)
        print(f"\nAnswer: {result['answer']}")

        print(f"\nSource Documents: {len(result['source_documents'])}")
        for i, doc in enumerate(result['source_documents'], 1):
            print(f"\n[Source {i}]")
            print(f"Content: {doc.page_content[:200]}...")
            print(f"Metadata: {doc.metadata}")

        return True

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_rag_pipeline()
