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