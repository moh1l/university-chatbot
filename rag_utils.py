"""
Shared helper functions for the university Q&A RAG chatbot.
Handles: loading data, chunking, embedding (OpenAI), and FAISS retrieval.
"""

import os
import re
import pickle
import numpy as np
import faiss
from openai import OpenAI

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

INDEX_PATH = "index/faiss.index"
CHUNKS_PATH = "index/chunks.pkl"


def get_client() -> OpenAI:
    """Create an OpenAI client. Reads OPENAI_API_KEY from the environment."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Run: export OPENAI_API_KEY='your-key-here'"
        )
    return OpenAI(api_key=api_key)


def load_and_chunk(filepath: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Load a plain text file and split it into overlapping chunks.

    For structured Q&A-style data, we chunk on blank-line boundaries so each
    Q/A pair stays together when possible. (Kept for .txt data — see
    load_pdfs_from_directory() for PDF sources.)
    """
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    # Split on blank lines first (keeps each Q/A block intact)
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]

    chunks = []
    current = ""
    for block in blocks:
        if len(current) + len(block) < chunk_size:
            current += ("\n\n" if current else "") + block
        else:
            if current:
                chunks.append(current)
            current = block
    if current:
        chunks.append(current)

    return chunks


def extract_pdf_text(filepath: str) -> str:
    """Extract all text from a PDF file, page by page, using pypdf."""
    from pypdf import PdfReader

    reader = PdfReader(filepath)
    pages_text = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        pages_text.append(page_text)
    return "\n".join(pages_text)


def chunk_text(text: str, chunk_size_words: int = 220, overlap_words: int = 40) -> list[str]:
    """
    Split text into overlapping chunks by word count (works well for text
    extracted from PDFs, which often lacks clean paragraph breaks).
    """
    # Collapse excess whitespace/line breaks left over from PDF extraction
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    words = text.split(" ")
    chunks = []
    step = max(chunk_size_words - overlap_words, 1)
    for i in range(0, len(words), step):
        chunk_words = words[i : i + chunk_size_words]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
        if i + chunk_size_words >= len(words):
            break
    return chunks


def load_pdfs_from_directory(dir_path: str, chunk_size_words: int = 220, overlap_words: int = 40) -> list[str]:
    """
    Read every .pdf file in dir_path, extract its text, and split it into
    chunks. Each chunk is tagged with its source filename (e.g. "[Source:
    handbook.pdf]") so answers can be traced back to the right document.
    """
    all_chunks = []
    pdf_files = sorted(f for f in os.listdir(dir_path) if f.lower().endswith(".pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in '{dir_path}'.")

    for filename in pdf_files:
        filepath = os.path.join(dir_path, filename)
        raw_text = extract_pdf_text(filepath)
        text_chunks = chunk_text(raw_text, chunk_size_words, overlap_words)

        if not text_chunks:
            print(f"  Warning: no extractable text found in {filename} (scanned image PDF?)")
            continue

        for chunk in text_chunks:
            all_chunks.append(f"[Source: {filename}]\n{chunk}")

        print(f"  {filename}: {len(text_chunks)} chunks")

    return all_chunks


def embed_texts(client: OpenAI, texts: list[str], batch_size: int = 100) -> np.ndarray:
    """
    Embed a list of texts using the OpenAI embeddings API, in batches.

    OpenAI caps each request at 300,000 tokens total. A large PDF dataset can
    easily exceed that in one call, so we send `texts` in smaller batches and
    stitch the results back together in order.
    """
    all_vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(model=EMBED_MODEL, input=batch)
        all_vectors.extend(item.embedding for item in response.data)
    return np.array(all_vectors, dtype="float32")


def build_index(chunks: list[str], client: OpenAI) -> faiss.IndexFlatIP:
    """Embed all chunks and build a FAISS inner-product (cosine-like) index."""
    vectors = embed_texts(client, chunks)
    # Normalize so inner product behaves like cosine similarity
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def save_index(index: faiss.IndexFlatIP, chunks: list[str]):
    os.makedirs("index", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)


def load_index() -> tuple[faiss.IndexFlatIP, list[str]]:
    if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError(
            "No index found. Run `python build_index.py` first to build one."
        )
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


# Minimum cosine similarity a top chunk must hit for the question to be
# considered "in the data". Vectors are normalized, so scores range -1..1;
# 0.35-0.45 is a reasonable starting cutoff for text-embedding-3-small.
# Lower it if too many valid questions get rejected; raise it if off-topic
# questions are still getting answered.
SIMILARITY_THRESHOLD = 0.35

NOT_FOUND_MESSAGE_EN = (
    "I don't have information about that in the university data I was given. "
    "Please contact your academic advisor or the registrar's office for help with this."
)

NOT_FOUND_MESSAGE_AR = (
    "لا تتوفر لدي معلومات حول هذا الموضوع ضمن بيانات الجامعة المتاحة لدي. "
    "يرجى التواصل مع مرشدك الأكاديمي أو مكتب القبول والتسجيل للمساعدة."
)

# Matches presence of Arabic-script characters
_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")


def _is_arabic(text: str) -> bool:
    return bool(_ARABIC_RE.search(text))


def retrieve(query: str, client: OpenAI, index: faiss.IndexFlatIP, chunks: list[str], k: int = 3) -> tuple[list[str], list[float]]:
    """Embed the query and return the top-k most relevant chunks with their similarity scores."""
    query_vec = embed_texts(client, [query])
    faiss.normalize_L2(query_vec)
    scores, indices = index.search(query_vec, k)
    result_chunks = [chunks[i] for i in indices[0] if i != -1]
    result_scores = [float(s) for s, i in zip(scores[0], indices[0]) if i != -1]
    return result_chunks, result_scores


def answer_question(
    query: str,
    client: OpenAI,
    index: faiss.IndexFlatIP,
    chunks: list[str],
    k: int = 3,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> tuple[str, list[str]]:
    """
    Full RAG step: retrieve context, then ask the LLM to answer using only that context.

    If the best-matching chunk's similarity score is below `similarity_threshold`,
    the question is treated as out-of-scope and answered with a fixed refusal
    message WITHOUT calling the LLM at all — this is a hard guardrail, not just
    a prompt instruction, so it can't be talked around by clever phrasing.
    """
    context_chunks, scores = retrieve(query, client, index, chunks, k=k)

    if not context_chunks or max(scores) < similarity_threshold:
        return (NOT_FOUND_MESSAGE_AR if _is_arabic(query) else NOT_FOUND_MESSAGE_EN), []

    context_text = "\n\n---\n\n".join(context_chunks)

    system_prompt = (
        "You are a helpful university assistant that answers student questions. "
        "Only use the information in the provided context to answer — never use "
        "outside knowledge, even if you know the answer. "
        "If the context does not fully answer the question, say clearly that you "
        "don't have that information and suggest the student contact the registrar "
        "or their academic advisor. Keep answers concise and friendly. "
        "Always respond in the same language the student used to ask the question — "
        "if they asked in Arabic, answer fully in Arabic; if they asked in English, "
        "answer in English. The context provided may be in a different language than "
        "the question; translate the relevant information naturally, don't just repeat it."
    )

    user_prompt = f"Context:\n{context_text}\n\nStudent question: {query}"

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    answer = response.choices[0].message.content
    return answer, context_chunks