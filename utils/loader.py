import os
from typing import List
from pypdf import PdfReader
from langchain_core.documents import Document


def load_pdf(file_path: str) -> List[Document]:
    documents = []

    try:
        reader = PdfReader(file_path)

        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()

            # Create Document object with metadata
            doc = Document(
                page_content=text,
                metadata={
                    "source": os.path.basename(file_path),
                    "page": page_num,
                    "total_pages": len(reader.pages)
                }
            )
            documents.append(doc)

    except Exception as e:
        print(f"Error loading {file_path}: {e}")

    return documents


def load_documents_from_directory(directory_path: str) -> List[Document]:
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")
        return []

    all_documents = []
    pdf_files = [f for f in os.listdir(directory_path) if f.endswith('.pdf')]

    if not pdf_files:
        print(f"No PDF files found in {directory_path}")
        return []

    print(f"Loading {len(pdf_files)} PDF file(s)...")

    for pdf_file in pdf_files:
        file_path = os.path.join(directory_path, pdf_file)
        print(f"  Processing: {pdf_file}")
        documents = load_pdf(file_path)
        all_documents.extend(documents)

    print(f"Loaded {len(all_documents)} pages total")

    return all_documents


def get_pdf_count(directory_path: str) -> int:
    if not os.path.exists(directory_path):
        return 0

    pdf_files = [f for f in os.listdir(directory_path) if f.endswith('.pdf')]
    return len(pdf_files)
