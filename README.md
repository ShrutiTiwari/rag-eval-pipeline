# RAG ABRSM Exam 📄

Retrieval-Augmented Generation chatbot for the ABRSM piano exam syllabus — ask questions, get accurate answers with source attribution and measurable retrieval quality.

🔗 [Live Demo](https://rag-abrsm-exam.vercel.app) · [Parent Project](https://www.powerparent.co.uk)

---

## What it does

Upload any PDF (here: the official ABRSM Piano 2025–2026 syllabus) and ask it questions in natural language:

- *"What pieces are required for Grade 3 piano?"*
- *"How many scales does a Grade 5 candidate need to know?"*
- *"What are the sight-reading requirements for Grade 1?"*

The system retrieves the most relevant chunks from the document using semantic similarity, feeds them as context to an LLM, and returns an answer with source attribution — showing exactly which part of the syllabus it drew from.

What makes this more than a basic chatbot demo is the **evaluation layer**: every retrieval is measurable. The dashboard exposes Precision, Recall, F1, MRR, and NDCG scores broken down by query difficulty — so you can see not just that it answers, but how well it retrieves.

---

## Why I built this

I have two children working toward ABRSM piano grade exams. The official syllabus PDF is dense — parents and students spend a lot of time hunting through it for specific requirements. A natural language interface over that document is genuinely useful.

Beyond the practical use case, this project exists to showcase the full RAG engineering stack — not just the "ask a question, get an answer" surface, but the retrieval pipeline, the evaluation methodology, and the architectural decisions that affect quality. Those decisions are where the real engineering lives.

---

## How the RAG pipeline works

```
PDF Document
    │
    ▼
DocumentLoader        — extracts text, preserves metadata (pages, size)
    │
    ▼
Chunking              — splits into overlapping 1000-char chunks
    │
    ▼
VectorStore           — generates OpenAI embeddings, cosine similarity search
    │
    ▼
ChatService           — retrieves top-K chunks, injects as context, calls LLM
    │
    ▼
Response + Sources    — answer with chunk references and similarity scores
```

Conversation history is maintained across turns, so follow-up questions work naturally: *"What about Grade 4?"* after asking about Grade 3 resolves correctly.

---

## Retrieval evaluation

This is the part that goes beyond most RAG demos.

`RetrievalEvaluator` runs a test suite of queries with known expected sources across three difficulty levels (easy / medium / hard) and computes:

| Metric | What it measures |
|---|---|
| **Precision** | Of the chunks retrieved, how many were actually relevant |
| **Recall** | Of the relevant chunks, how many were retrieved |
| **F1** | Harmonic mean of precision and recall |
| **MRR** | Was the most relevant chunk ranked first? |
| **NDCG** | Full ranking quality — rewards putting better results higher |

The `/api/rag-docs/evaluate` endpoint renders this as an interactive HTML dashboard, with per-query drill-down showing similarity scores and retrieved content previews.

`/api/rag-docs/compare` lets you A/B test retrieval configurations — different chunk sizes, topK values, similarity thresholds — against the same query set to see which configuration performs better before deploying changes.

---

## Technical decisions

**Chunking with overlap** — 100-character overlap between chunks prevents answers from being split across a chunk boundary. Without overlap, a question about a piece that straddles two chunks returns incomplete results.

**In-memory vector store** — no external vector database (Pinecone, Weaviate, etc.). For a single-document use case this is simpler and faster. The trade-off is that embeddings are regenerated on cold start; caching (30-minute TTL) mitigates this in practice.

**Cost-controlled embedding** — limited to the first 10 chunks per document on the demo deployment to keep OpenAI costs predictable. The architecture supports full document embedding; it's a config change.

**TruLensClient** — stub interface to a TruLens Python evaluation service for LLM-level metrics (groundedness, answer relevance, context relevance). The JS RAG pipeline is instrumented to call it when available; the service itself is the next build milestone.

**No auth, no database** — deliberate. The RAG pipeline is stateless. Documents live in `public/pdf-sources/`, responses are generated on demand. This makes the project trivially deployable and easy to understand as a standalone showcase.

---

## Project structure

```
rag-abrsm-exam/
├── server.js                   # Express entry point (~30 lines)
├── public/
│   ├── index.html              # document dashboard UI
│   └── pdf-sources/
│       └── ABRSM_Piano_2025_2026_syllabus.pdf
└── src/
    ├── core/
    │   ├── DocumentLoader.js   # PDF loading + chunking
    │   ├── VectorStore.js      # embeddings + cosine similarity search
    │   ├── ChatService.js      # RAG orchestration + conversation history
    │   ├── RetrievalEvaluator.js  # Precision/Recall/F1/MRR/NDCG
    │   └── TruLensClient.js    # interface to Python evaluation service
    └── routes/
        └── documentViewerRoutes.js  # all API endpoints
```

---

## API endpoints

| Endpoint | What it does |
|---|---|
| `GET /` | Document dashboard |
| `GET /api/rag-docs` | Document list + stats + embedding info |
| `POST /api/rag-docs/chat` | RAG chat with conversation history |
| `GET /api/rag-docs/chat/test` | Interactive chat UI |
| `POST /api/rag-docs/vector-search` | Semantic search with similarity scores |
| `GET /api/rag-docs/vector-search/test` | Interactive search UI |
| `GET /api/rag-docs/evaluate` | Retrieval quality dashboard |
| `POST /api/rag-docs/compare` | A/B test retrieval configurations |
| `GET /api/rag-docs/:id/chunks` | Inspect chunking for any document |
| `GET /api/rag-docs/:id/view` | Full document viewer with text search |

---

## Running locally

```bash
git clone https://github.com/ShrutiTiwari/rag-abrsm-exam
cd rag-abrsm-exam
npm install
cp .env.example .env   # add your OPENAI_API_KEY
npm start              # http://localhost:3000
```

---

## What I'd build next

**TruLens Python service** — the groundedness and answer relevance metrics (does the answer actually follow from the retrieved context?) are more meaningful than retrieval metrics alone. The client is already instrumented; the service needs building.

**Chunk size experimentation UI** — the `/compare` endpoint supports A/B testing configurations programmatically; a UI to set parameters and visualise the metric differences interactively would make this a proper RAG tuning tool.

**Multi-document RAG** — the pipeline supports multiple PDFs already. The natural extension for the ABRSM use case is adding theory syllabuses, marking criteria, and examiner reports — so a student can ask cross-document questions like *"What theory topics appear in both the Grade 4 syllabus and the examiner feedback?"*

---

## About the builder

Built by Shruti Tiwari — independent AI product builder (2023–present), 20 years backend engineering, former VP at Goldman Sachs. I build AI-native tools for music education and parenting, rooted in my own experience as a piano learner and parent of two children working toward ABRSM piano exams.

This project is part of [PowerParent](https://www.powerparent.co.uk) — a platform for parents and teachers to track music practice, manage school events, and support children's learning with AI.
