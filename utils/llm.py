from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
import config


def get_llm():
    if config.EMBEDDING_MODEL_TYPE == "openai":
        print(f"Using OpenAI LLM: {config.OPENAI_LLM_MODEL}")
        return ChatOpenAI(
            model=config.OPENAI_LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            openai_api_key=config.OPENAI_API_KEY
        )

    elif config.EMBEDDING_MODEL_TYPE == "groq":
        print(f"Using Groq LLM: {config.GROQ_LLM_MODEL}")
        return ChatGroq(
            model=config.GROQ_LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            groq_api_key=config.GROQ_API_KEY
        )

    elif config.EMBEDDING_MODEL_TYPE == "huggingface":
        print(f"Using HuggingFace LLM: {config.HUGGINGFACE_LLM_MODEL}")
        # For simplicity, fallback to Groq for now since local HF LLMs are complex
        print("Note: Falling back to Groq LLM as local HuggingFace LLMs require more setup")
        return ChatGroq(
            model=config.GROQ_LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            groq_api_key=config.GROQ_API_KEY
        )

    else:
        raise ValueError(f"Unsupported embedding model type: {config.EMBEDDING_MODEL_TYPE}")


def test_llm():
    try:
        llm = get_llm()
        test_query = "Hello, please respond with 'LLM is working correctly!'"
        result = llm.invoke(test_query)
        print(f"LLM Response: {result.content}")
        return True
    except Exception as e:
        print(f"LLM test failed: {e}")
        return False


if __name__ == "__main__":
    test_llm()