# University Q&A Chatbot

A bilingual (Arabic/English) chatbot that answers student questions about registration, GPA rules, academic regulations, graduation requirements, and more, using the university's own PDF documents as its only source of truth.

It uses **RAG (Retrieval-Augmented Generation)**: PDFs are split into chunks, embedded with OpenAI, and searched with FAISS. The most relevant chunks are then passed to the language model, which answers only from them. If nothing relevant is found, the bot refuses instead of guessing.

## Features

- Answers in the same language as the question (Arabic or English)
- Works directly from PDF files placed in `data/`
- Hard guardrail: questions that don't match the data (similarity below a threshold) are rejected without calling the LLM
- Shows the sources used for each answer
- Two interfaces: Streamlit (quick UI) and FastAPI + custom HTML frontend
- Logs every question so you can see what students ask and what the bot couldn't answer

## How it works

1. `build_index.py` reads every PDF in `data/`, splits the text into chunks (~220 words, 40 overlap), embeds them with `text-embedding-3-small`, and saves a FAISS index to `index/`.
2. When a question arrives, it is embedded and the top 3 closest chunks are retrieved.
3. If the best similarity score is below `0.35`, a fixed "not found" message is returned.
4. Otherwise, the chunks and question are sent to `gpt-4o-mini` with instructions to answer only from the context.

## Project structure

| File | Purpose |
|---|---|
| `rag_utils.py` | Core logic: PDF loading, chunking, embedding, retrieval, answering |
| `build_index.py` | Builds the FAISS index from the PDFs in `data/` |
| `app.py` | Streamlit interface |
| `backend.py` | FastAPI backend (`/api/ask`, `/api/health`) and question logging |
| `static/index.html` | Frontend served by the FastAPI backend |
| `view_logs.py` | Prints a summary of logged questions |
| `data/` | University PDF documents |
| `requirements.txt` | Python dependencies |

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your OpenAI API key

Windows (PowerShell):

```powershell
$env:OPENAI_API_KEY="your-key-here"
```

Mac / Linux:

```bash
export OPENAI_API_KEY="your-key-here"
```

Never commit your API key to the repository.

### 3. Build the index

Make sure your PDFs are in `data/`, then run:

```bash
python build_index.py
```

Run this again whenever the PDFs change. The generated `index/` folder is not stored in the repo.

### 4. Run the app

Option A: Streamlit

```bash
streamlit run app.py
```

Option B: FastAPI backend with the HTML frontend

```bash
uvicorn backend:app --reload --port 8000
```

Then open http://localhost:8000 in your browser.

### 5. View logged questions

```bash
python view_logs.py
```

## Configuration

In `rag_utils.py`:

- `SIMILARITY_THRESHOLD` (default `0.35`): lower it if valid questions get rejected, raise it if off-topic questions still get answered.
- `CHAT_MODEL` and `EMBED_MODEL`: the OpenAI models used.
- `k` (in `answer_question`): number of chunks retrieved per question.

## Notes

- PDFs that are scanned images have no extractable text and will be skipped (a warning is printed).
- The bot only knows what is in the provided documents. For anything else, students are directed to the registrar or their academic advisor.
- Student questions are saved in `logs/questions.jsonl` (not included in the repo).

---

## نبذة بالعربي

شات بوت (chatbot) يجاوب الطلاب على أسئلتهم عن الجامعة اعتمادًا على ملفات PDF الرسمية فقط. يدعم العربية والإنجليزية، ويرفض الأسئلة الخارجة عن البيانات بدل ما يخمّن.
