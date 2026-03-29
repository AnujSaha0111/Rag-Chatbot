import os
from langchain_community.vectorstores import FAISS
import config
from utils.loader import load_documents_from_directory, get_pdf_count
from utils.splitter import split_documents, get_chunk_stats
from utils.embeddings import get_embeddings


def ingest_documents(data_dir: str = config.DATA_DIR) -> bool:
    try:
        # Validate configuration
        config.validate_config()

        # Check if PDFs exist
        pdf_count = get_pdf_count(data_dir)
        if pdf_count == 0:
            print(f"No PDF files found in {data_dir}")
            print("Please add PDF files to the data/ directory")
            return False

        print(f"\n{'='*50}")
        print(f"Starting Document Ingestion")
        print(f"{'='*50}\n")
        print(f"Found {pdf_count} PDF file(s)")

        # Step 1: Load documents
        print("\n[1/4] Loading documents...")
        documents = load_documents_from_directory(data_dir)

        if not documents:
            print("No documents loaded. Exiting.")
            return False

        # Step 2: Split documents
        print("\n[2/4] Splitting documents into chunks...")
        chunks = split_documents(documents)

        # Print chunk statistics
        stats = get_chunk_stats(chunks)
        print(f"Chunk Statistics:")
        print(f"  - Total chunks: {stats['total']}")
        print(f"  - Average length: {stats['avg_length']:.0f} chars")
        print(f"  - Min/Max length: {stats['min_length']}/{stats['max_length']} chars")

        # Step 3: Get embeddings model
        print("\n[3/4] Initializing embeddings model...")
        embeddings = get_embeddings()

        # Step 4: Create and persist vector store
        print("\n[4/4] Creating vector store and generating embeddings...")
        print("This may take a few minutes...")

        # Create new vectorstore
        vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=embeddings
        )

        # Save to disk
        vectorstore.save_local(config.VECTORSTORE_PATH)

        print(f"\n{'='*50}")
        print(f"[SUCCESS] Ingestion Complete!")
        print(f"{'='*50}")
        print(f"Stored {len(chunks)} chunks in vector database")
        print(f"Location: {config.VECTORSTORE_PATH}")

        return True

    except Exception as e:
        print(f"\n[ERROR] Ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_vectorstore_exists() -> bool:
    index_file = os.path.join(config.VECTORSTORE_PATH, "index.faiss")
    return os.path.exists(index_file)


if __name__ == "__main__":
    success = ingest_documents()
    exit(0 if success else 1)
