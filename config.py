import os
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL_TYPE = os.getenv("EMBEDDING_MODEL_TYPE", "huggingface")

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

def get_api_keys(runtime_keys=None):
    """
    Get API keys from environment or runtime overrides

    Args:
        runtime_keys: Dict with keys like {'openai': '...', 'groq': '...', 'huggingface': '...'}

    Returns:
        Dict with current API keys
    """
    runtime_keys = runtime_keys or {}
    return {
        "openai": runtime_keys.get("openai") or OPENAI_API_KEY,
        "huggingface": runtime_keys.get("huggingface") or HUGGINGFACE_API_KEY,
        "groq": runtime_keys.get("groq") or GROQ_API_KEY,
    }

# OpenAI Models
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_LLM_MODEL = "gpt-4o-mini"

# HuggingFace Models
HUGGINGFACE_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HUGGINGFACE_LLM_MODEL = "microsoft/DialoGPT-medium"

# Groq Models (using sentence transformers locally for embeddings)
GROQ_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_LLM_MODEL = "llama-3.1-8b-instant"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

TOP_K = 4
SIMILARITY_THRESHOLD = 0.5

VECTORSTORE_PATH = "./vectorstore"
COLLECTION_NAME = "rag_documents"

DATA_DIR = "./data"

LLM_TEMPERATURE = 0.3

def validate_config():
    """Validate required configuration based on embedding model type"""
    if EMBEDDING_MODEL_TYPE == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings and LLM. Please set it in .env file")
    elif EMBEDDING_MODEL_TYPE == "huggingface":
        if not HUGGINGFACE_API_KEY:
            print("Warning: HUGGINGFACE_API_KEY not found. Will attempt to use local model.")
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is required for LLM when using HuggingFace embeddings. Please set it in .env file")
    elif EMBEDDING_MODEL_TYPE == "groq":
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is required for Groq LLM. Please set it in .env file")
    else:
        raise ValueError(f"Unsupported embedding model type: {EMBEDDING_MODEL_TYPE}")
    return True
