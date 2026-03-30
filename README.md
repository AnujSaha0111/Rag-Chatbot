# 🤖 Multi-Provider RAG Chatbot

A production-ready **Retrieval-Augmented Generation (RAG)** chatbot that supports multiple LLM and embedding providers. Chat with your documents using AI with a beautiful Streamlit interface.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-1.2+-green.svg)](https://langchain.com)

## 🌟 Features

- **🔄 Multi-Provider Support**: OpenAI, Groq, HuggingFace embeddings and LLMs
- **🔑 Session-Based API Keys**: Enter API keys directly in the app without storing them permanently
- **💰 Cost-Effective**: Local embeddings + Groq LLM (minimal API costs)
- **📄 PDF Document Support**: Advanced PDF processing and ingestion
- **🧠 Conversation Memory**: Maintains context across questions
- **📊 Source Attribution**: View relevant document excerpts for each answer
- **🎨 Professional UI**: Clean Streamlit web interface with real-time status
- **⚙️ Easy Configuration**: Environment-based provider switching
- **🚀 Production Ready**: Error handling, logging, and validation
- **🔧 Extensible**: Easy to add new providers and document types

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Documents     │───▶│   Embeddings     │───▶│  Vector Store   │
│   (PDF files)  │    │ (Local/API)      │    │    (FAISS)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌──────────────────┐            │
│  User Question  │───▶│   Retrieval      │◀───────────┘
└─────────────────┘    │   (Semantic)     │
                       └──────────────────┘
                                │
┌─────────────────┐    ┌──────────────────┐
│   AI Response   │◀───│      LLM         │
│  with Sources   │    │ (OpenAI/Groq)    │
└─────────────────┘    └──────────────────┘
```

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/AnujSaha0111/rag-chatbot.git
cd rag-chatbot
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment (Optional)

You have two options for providing API keys:

#### Option A: Environment Variables (Recommended for local development)

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your API keys and embedding model type
```

#### Option B: Session-Based Input (Recommended for deployed apps)

No need to set environment variables! Just run the app and enter your API keys directly in the Streamlit interface:

```bash
# Just run the app without setting environment variables
streamlit run app.py
```

Then in the app:
1. Look for **"🔑 API Keys"** section in the left sidebar
2. Expand it and enter your API keys
3. Click "Initialize Chatbot" to use them
4. Keys are only stored in the current session and cleared when you close the app

**Benefits of Session-based keys:**
- ✅ No API keys stored in environment files
- ✅ Perfect for deployed applications
- ✅ Each user can use their own keys
- ✅ Automatic cleanup when session ends
- ✅ Fallback to environment variables if needed

### 5. Add Your Documents

```bash
# Place PDF files in the data directory
mkdir -p data
cp your_documents.pdf data/
```

### 6. Run Document Ingestion

```bash
python ingest.py
```

### 7. Start the Chatbot

```bash
streamlit run app.py
```

Visit `http://localhost:8501` and start chatting with your documents!

## 📁 Project Structure

```
rag-chatbot/
├── 📄 app.py                 # Streamlit web application
├── 🔧 main.py                # RAG pipeline core logic
├── 📥 ingest.py              # Document ingestion script
├── ⚙️ config.py              # Configuration management
├── 📋 requirements.txt       # Python dependencies
├── 🔒 .env.example          # Environment variables template
├── 🔒 .env                  # Your environment variables (git-ignored)
├── 🚫 .gitignore            # Git ignore rules
├── 📖 README.md             # This file
├── 📁 data/                 # Your PDF documents (git-ignored)
├── 🗃️ vectorstore/          # Generated embeddings (git-ignored)
└── 🛠️ utils/
    ├── __init__.py
    ├── 🔤 embeddings.py     # Multi-provider embedding support
    ├── 🤖 llm.py           # Multi-provider LLM support
    ├── 📁 loader.py        # Document loading utilities
    └── ✂️ splitter.py      # Text chunking utilities
```

## 🔧 Configuration

### 🔑 API Keys Configuration

You can provide API keys in two ways:

#### Method 1: Environment Variables (.env file)

Create `.env` file with your preferred setup:

```bash
EMBEDDING_MODEL_TYPE=groq
GROQ_API_KEY=your_groq_api_key_here
```

#### Method 2: Session-Based Input (NEW!)

Skip the `.env` file entirely! Use the built-in UI:

1. Start the Streamlit app: `streamlit run app.py`
2. In the left sidebar, expand **"🔑 API Keys"**
3. Enter your API keys:
   - **OpenAI API Key** (optional, for GPT models)
   - **Groq API Key** (optional, for Llama models)
   - **HuggingFace API Key** (optional, for HF inference)
4. Click **"Initialize Chatbot"** - it automatically uses the provided keys
5. Keys are session-specific and cleared when you close the app

**Key Priority:** Session keys override environment variables if both are provided.

**Use Cases:**
- **Deployed Apps**: Users provide their own keys safely
- **Multi-User**: Each user can use different API keys
- **No Key Files**: Perfect when you can't store keys in `.env`
- **Temporary Testing**: Try different providers without changing files

### 🔑 Getting API Keys

| Provider | Sign Up | Free Tier | Note |
|----------|---------|-----------|------|
| **Groq** | [console.groq.com](https://console.groq.com) | ✅ Generous | Fast inference |
| **OpenAI** | [platform.openai.com](https://platform.openai.com) | 💳 Credit required | High quality |
| **HuggingFace** | [huggingface.co](https://huggingface.co/settings/tokens) | ✅ Free tier | Local + API options |

## 💡 Usage Examples

### Basic Exploration
```
What is this document about?
Summarize the main points.
What are the key findings?
```

### Detail-Oriented Questions
```
What evidence supports the claim about [specific topic]?
How was the methodology designed for [research area]?
What are the limitations mentioned?
```

### Comparative Analysis
```
Compare approach A and approach B mentioned in the document.
What are the pros and cons of the proposed solution?
How does this relate to previous research?
```

### Follow-up Conversations
```
Can you elaborate on that point?
What specific examples are provided?
How confident is this conclusion?
```

## 🎛️ Advanced Features

### Custom Document Processing

Add support for new file types in `utils/loader.py`:

```python
def load_documents(data_dir):
    # Add support for .txt, .docx, .html, etc.
    pass
```

### Advanced Retrieval

Implement hybrid search in `main.py`:

```python
# Combine semantic + keyword search
from langchain.retrievers import EnsembleRetriever
```

### Custom Prompts

Edit prompts in `main.py`:

```python
RAG_PROMPT_TEMPLATE = """You are a domain expert.
Based on the provided context, give detailed analysis...

Context: {context}
Question: {question}

Expert Analysis:"""
```

## 🚨 Troubleshooting

### Common Issues & Solutions

**❌ ImportError: No module named 'langchain'**
```bash
# Solution: Install dependencies
pip install -r requirements.txt
```

**❌ OpenAI API quota exceeded**
```bash
# Solution: Switch to Groq
# Edit .env:
EMBEDDING_MODEL_TYPE=groq
```

**❌ Empty source documents displaying**
```bash
# Solution: Fixed in latest version
# Restart Streamlit app
```

**❌ Vector store not found**
```bash
# Solution: Run ingestion
python ingest.py
```

**❌ Streamlit key errors**
```bash
# Solution: Clear browser cache
# Or restart with: streamlit run app.py --server.runOnSave=true
```

**❌ API keys not working in session**
```bash
# Solution: Make sure to expand "🔑 API Keys" in sidebar and enter keys BEFORE initializing
# Session keys override environment variables
# Check your API key format (should be pasted exactly from provider)
```

**❌ "API key is required" error during initialization**
```bash
# Solution: Check that you entered the API key for your configured provider:
# - EMBEDDING_MODEL_TYPE=openai → need OpenAI key
# - EMBEDDING_MODEL_TYPE=groq → need Groq key
# - EMBEDDING_MODEL_TYPE=huggingface → need Groq key (for LLM fallback)
```

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 📞 Support & Community

- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/AnujSaha0111/rag-chatbot/issues)
- 📧 **Email**: anujsahabest0111@gmail.com

🌟 **Star this repo if it helped you!**

**🚀 Built with ❤️ for the AI community**