import os                                                               # for file handling
import fitz                                                             # for PDF text extraction
import chromadb                                                         # for vector database
from sentence_transformers import SentenceTransformer                   # for generating embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter     # for splitting text into chunks
from dotenv import load_dotenv                                          # for loading Anthropic API key from .env file

load_dotenv()

PDF_DIR = "data/raw_pdfs"
CHROMA_DIR = "data/chroma_db"

def extract_text(pdf_path):
    """
    Extracts text from a PDF file using PyMuPDF (fitz).
    Returns: the extracted text as a single string.
    """
    doc = fitz.open(pdf_path)
    return " ".join(page.get_text() for page in doc)

def ingest(chunk_size=500, chunk_overlap=50):
    """
    Ingests PDF documents from the specified directory, extracts text, splits it into chunks, generates embeddings,
    and stores them in a ChromaDB collection.
    Parameters:
        - chunk_size: the maximum number of characters in each text chunk (default: 500)
        - chunk_overlap: the number of overlapping characters between consecutive chunks (default: 50)
    Returns: the name of the ChromaDB collection created for this ingestion run.
    """
    print(f"Ingesting with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Collection name encodes the configuration for experiment tracking
    collection_name = f"atsb_cs{chunk_size}_co{chunk_overlap}"

    # Delete if exists so it can re-run cleanly
    try:
        client.delete_collection(collection_name)
    except Exception as e:
        print(f"Error occurred while deleting collection: {e}")

    collection = client.create_collection(collection_name)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
    total_chunks = 0

    for i, pdf_file in enumerate(pdf_files):
        pdf_path = os.path.join(PDF_DIR, pdf_file)
        print(f"  [{i+1}/{len(pdf_files)}] {pdf_file}")

        text = extract_text(pdf_path)

        if len(text.strip()) < 500:
            print("Skipping due to insufficient text")
            continue

        chunks = splitter.split_text(text)

        embeddings = model.encode(chunks, show_progress_bar=False).tolist()

        ids = [f"{pdf_file}::chunk_{j}" for j in range(len(chunks))]
        metadatas = [{"source": pdf_file, "chunk_index": j} for j in range(len(chunks))]

        collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )

        total_chunks += len(chunks)
        print(f"    {len(chunks)} chunks ingested")

    print(f"\nDone. Total chunks: {total_chunks}")
    print(f"Collection: {collection_name}")
    return collection_name

if __name__ == "__main__":
    ingest(chunk_size=500, chunk_overlap=0)
    ingest(chunk_size=500, chunk_overlap=100)