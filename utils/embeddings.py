from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
import config


def get_embeddings():
    if config.EMBEDDING_MODEL_TYPE == "openai":
        print("Using OpenAI embeddings...")
        return OpenAIEmbeddings(
            model=config.OPENAI_EMBEDDING_MODEL,
            openai_api_key=config.OPENAI_API_KEY
        )

    elif config.EMBEDDING_MODEL_TYPE == "huggingface":
        print("Using HuggingFace embeddings...")
        try:
            # Try with API key first if available
            if config.HUGGINGFACE_API_KEY:
                from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
                return HuggingFaceInferenceAPIEmbeddings(
                    api_key=config.HUGGINGFACE_API_KEY,
                    model_name=config.HUGGINGFACE_EMBEDDING_MODEL
                )
            else:
                # Use local HuggingFace model
                print("No HuggingFace API key found. Using local model...")
                return HuggingFaceEmbeddings(
                    model_name=config.HUGGINGFACE_EMBEDDING_MODEL
                )
        except ImportError:
            print("HuggingFace dependencies not found. Using local model...")
            return HuggingFaceEmbeddings(
                model_name=config.HUGGINGFACE_EMBEDDING_MODEL
            )

    elif config.EMBEDDING_MODEL_TYPE == "groq":
        print("Using Groq with local embeddings...")
        # Groq doesn't have embeddings API, so we use local sentence transformers
        return HuggingFaceEmbeddings(
            model_name=config.GROQ_EMBEDDING_MODEL
        )

    else:
        raise ValueError(f"Unsupported embedding model type: {config.EMBEDDING_MODEL_TYPE}")


def test_embeddings():
    try:
        embeddings = get_embeddings()
        test_text = "This is a test document for embeddings."
        result = embeddings.embed_query(test_text)
        print(f"Embeddings working! Vector dimension: {len(result)}")
        return True
    except Exception as e:
        print(f"Embeddings test failed: {e}")
        return False
