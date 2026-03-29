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

if 'chatbot' not in st.session_state:
    st.session_state.chatbot = None
    st.session_state.chatbot_initialized = False

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'show_sources' not in st.session_state:
    st.session_state.show_sources = True

if 'api_keys' not in st.session_state:
    st.session_state.api_keys = {
        'openai': '',
        'groq': '',
        'huggingface': ''
    }


def initialize_chatbot():
    try:
        if not check_vectorstore_exists():
            st.error("⚠️ Vector store not found. Please run ingestion first!")
            return False

        with st.spinner("Initializing chatbot..."):
            st.session_state.chatbot = RAGChatbot(runtime_keys=st.session_state.api_keys)
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


def display_sources(sources):
    if not sources:
        return

    with st.expander(f"📄 View {len(sources)} Source Document(s)", expanded=False):
        for i, doc in enumerate(sources, 1):
            st.markdown(f"**Source {i}:** {doc.metadata.get('source', 'Unknown')}")
            st.markdown(f"**Page:** {doc.metadata.get('page', 'N/A')}")
            st.markdown(f"**Content:**")

            content = doc.page_content if doc.page_content else "No content available"
            clean_content = ' '.join(content.split())[:300] + "..."

            import time
            unique_key = f"source_content_{i}_{int(time.time() * 1000000)}"
            st.text_area(
                label="Document Content",
                value=clean_content,
                height=100,
                disabled=True,
                key=unique_key,
                label_visibility="collapsed"
            )
            st.markdown("---")


def main():
    st.markdown('<div class="main-header">🤖 RAG Chatbot</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Chat with your documents using AI</div>',
                unsafe_allow_html=True)

    with st.sidebar:
        st.header("⚙️ Configuration")

        # API Keys Section
        with st.expander("🔑 API Keys", expanded=False):
            st.caption("Enter your API keys for this session. Keys are not stored permanently.")

            st.session_state.api_keys['openai'] = st.text_input(
                "OpenAI API Key",
                value=st.session_state.api_keys['openai'],
                type="password",
                key="openai_key_input",
                help="Your OpenAI API key for GPT models"
            )

            st.session_state.api_keys['groq'] = st.text_input(
                "Groq API Key",
                value=st.session_state.api_keys['groq'],
                type="password",
                key="groq_key_input",
                help="Your Groq API key for Llama models"
            )

            st.session_state.api_keys['huggingface'] = st.text_input(
                "HuggingFace API Key",
                value=st.session_state.api_keys['huggingface'],
                type="password",
                key="hf_key_input",
                help="Your HuggingFace API key for inference"
            )

            st.info("💡 Only provide keys for the provider you're using. Session keys override environment variables.")

        st.markdown("---")

        st.info(f"""
        **Current Settings:**
        - Embedding Model: {config.EMBEDDING_MODEL_TYPE.upper()}
        - LLM Model: {config.GROQ_LLM_MODEL if config.EMBEDDING_MODEL_TYPE == 'groq' else (config.HUGGINGFACE_LLM_MODEL if config.EMBEDDING_MODEL_TYPE == 'huggingface' else config.OPENAI_LLM_MODEL)}
        - Chunk Size: {config.CHUNK_SIZE}
        - Top K Results: {config.TOP_K}
        """)

        st.markdown("---")

        st.header("📚 Document Management")

        pdf_count = get_pdf_count(config.DATA_DIR)
        vectorstore_exists = check_vectorstore_exists()

        st.metric("PDF Files in /data", pdf_count)
        st.metric("Vector Store Status", "✅ Ready" if vectorstore_exists else "❌ Not Ready")

        st.markdown("---")

        st.subheader("📥 Ingest Documents")
        st.caption("Load PDFs from the /data folder")

        if st.button("🚀 Run Ingestion", type="primary", use_container_width=True):
            if pdf_count == 0:
                st.error("No PDF files found in /data directory!")
            else:
                run_ingestion()

        st.markdown("---")

        if not st.session_state.chatbot_initialized:
            if st.button("🔄 Initialize Chatbot", use_container_width=True):
                initialize_chatbot()
        else:
            st.success("✅ Chatbot Active")
            if st.button("🔄 Reinitialize", use_container_width=True):
                initialize_chatbot()

        st.markdown("---")

        st.subheader("🎛️ Options")
        st.session_state.show_sources = st.checkbox("Show source documents", value=True)

        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            if st.session_state.chatbot:
                st.session_state.chatbot.clear_memory()
            st.rerun()

        st.markdown("---")

        with st.expander("ℹ️ How to Use"):
            st.markdown("""
            1. **Add PDFs**: Place PDF files in the `data/` folder
            2. **Run Ingestion**: Click 'Run Ingestion' to process documents
            3. **Initialize**: Click 'Initialize Chatbot' to load the system
            4. **Chat**: Type your questions in the chat box
            5. **View Sources**: Check the source documents used for answers
            """)

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

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant" and "sources" in message and st.session_state.show_sources:
                display_sources(message["sources"])

    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.chatbot.query(prompt)
                    response = result["answer"]
                    sources = result.get("source_documents", [])

                    st.markdown(response)

                    if st.session_state.show_sources and sources:
                        display_sources(sources)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "sources": sources
                    })

                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })


if __name__ == "__main__":
    main()
