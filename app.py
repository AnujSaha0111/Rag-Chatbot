import streamlit as st
import os
from main import RAGChatbot
from ingest import ingest_documents, check_vectorstore_exists, get_pdf_count
import config

# Page configuration
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        border-left: 3px solid #1E88E5;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = None
    st.session_state.chatbot_initialized = False

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'show_sources' not in st.session_state:
    st.session_state.show_sources = True


def initialize_chatbot():
    try:
        if not check_vectorstore_exists():
            st.error("⚠️ Vector store not found. Please run ingestion first!")
            return False

        with st.spinner("Initializing chatbot..."):
            st.session_state.chatbot = RAGChatbot()
            st.session_state.chatbot.initialize()
            st.session_state.chatbot_initialized = True

        st.success("✅ Chatbot initialized successfully!")
        return True

    except Exception as e:
        st.error(f"❌ Failed to initialize chatbot: {e}")
        return False


def run_ingestion():
    with st.spinner("Processing documents... This may take a few minutes."):
        success = ingest_documents()

    if success:
        st.success("✅ Documents ingested successfully!")
        st.session_state.chatbot_initialized = False
        st.rerun()
    else:
        st.error("❌ Ingestion failed. Check the logs for details.")

def main():
    # Header
    st.markdown('<div class="main-header">🤖 RAG Chatbot</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Chat with your documents using AI</div>',
                unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Display current settings
        st.info(f"""
        **Current Settings:**
        - Embedding Model: {config.EMBEDDING_MODEL_TYPE.upper()}
        - LLM Model: {config.GROQ_LLM_MODEL if config.EMBEDDING_MODEL_TYPE == 'groq' else (config.HUGGINGFACE_LLM_MODEL if config.EMBEDDING_MODEL_TYPE == 'huggingface' else config.OPENAI_LLM_MODEL)}
        - Chunk Size: {config.CHUNK_SIZE}
        - Top K Results: {config.TOP_K}
        """)

        st.markdown("---")

        # Document Management
        st.header("📚 Document Management")

        pdf_count = get_pdf_count(config.DATA_DIR)
        vectorstore_exists = check_vectorstore_exists()

        st.metric("PDF Files in /data", pdf_count)
        st.metric("Vector Store Status", "✅ Ready" if vectorstore_exists else "❌ Not Ready")

        st.markdown("---")

    # Main chat interface
    if not st.session_state.chatbot_initialized:
        st.warning("⚠️ Please initialize the chatbot from the sidebar to start chatting.")

        # Show instructions
        col1, col2, col3 = st.columns(3)

        with col1:
            st.info("**Step 1**\n\nAdd PDF files to the `data/` folder")

        with col2:
            st.info("**Step 2**\n\nClick 'Run Ingestion' in the sidebar")

        with col3:
            st.info("**Step 3**\n\nClick 'Initialize Chatbot' to start")

        return




if __name__ == "__main__":
    main()
