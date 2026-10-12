# ingest.py
#############################################
# 1. Read PDFs
# 2. Extract text
# 3. Split into chunks
# 4. Generate embeddings using nomic-embed-text
# 5. Store them in ChromaDB
#############################################
import os
import uuid

import chromadb
import ollama
from pypdf import PdfReader

from config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


DOCUMENTS_DIR = "./documents"


def load_pdf(file_path):
    """Extract text from a PDF."""

    reader = PdfReader(file_path)

    pages = []

    for page_number, page in enumerate(reader.pages):

        text = page.extract_text()

        if text:
            pages.append({
                "page": page_number + 1,
                "text": text
            })

    return pages


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks."""

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks


def get_embedding(text):
    """Generate embedding using Ollama."""

    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=text
    )

    return response["embeddings"][0]


def ingest_pdf(file_path, collection):
    """Ingest one PDF into a collection."""

    filename = os.path.basename(file_path)
    total_chunks = 0

    for page_data in load_pdf(file_path):
        page_number = page_data["page"]
        chunks = chunk_text(page_data["text"])

        for chunk_index, chunk in enumerate(chunks):
            print(
                f"Embedding page {page_number}, "
                f"chunk {chunk_index + 1}"
            )

            embedding = get_embedding(chunk)
            document_id = str(uuid.uuid4())

            collection.add(
                ids=[document_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{
                    "source": filename,
                    "page": page_number,
                    "chunk": chunk_index
                }]
            )

            total_chunks += 1

    return total_chunks


def main():

    print("Starting ingestion...")

    # Connect to ChromaDB
    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    total_chunks = 0

    # Process every PDF
    for filename in os.listdir(DOCUMENTS_DIR):

        if not filename.lower().endswith(".pdf"):
            continue

        file_path = os.path.join(
            DOCUMENTS_DIR,
            filename
        )

        print(f"\nProcessing: {filename}")

        total_chunks += ingest_pdf(file_path, collection)

    print("\n-----------------------------")
    print("Ingestion completed")
    print(f"Total chunks: {total_chunks}")
    print("-----------------------------")


if __name__ == "__main__":
    main()