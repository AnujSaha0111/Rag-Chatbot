import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Embedding Model Configuration


# Data Directory
DATA_DIR = "./data"

# Temperature for LLM
LLM_TEMPERATURE = 0.3

def validate_config():
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
