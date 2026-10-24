"""
Run this once (or whenever your PDF files change) to build the FAISS index.

Drop any number of PDF files into the data/ folder, then:

Usage:
    export OPENAI_API_KEY="your-key-here"
    python build_index.py
"""

from rag_utils import get_client, load_pdfs_from_directory, build_index, save_index

DATA_DIR = "data"


def main():
    print(f"Scanning '{DATA_DIR}/' for PDF files...")
    chunks = load_pdfs_from_directory(DATA_DIR)
    print(f"\nTotal chunks created: {len(chunks)}")

    client = get_client()

    print("Embedding chunks with OpenAI (this calls the API, may take a bit for many chunks)...")
    index = build_index(chunks, client)

    print("Saving index to disk...")
    save_index(index, chunks)

    print("Done! Index saved to index/faiss.index")


if __name__ == "__main__":
    main()