# rag-eval-pipeline

Retrieval-Augmented Generation pipeline with swappable chunking, embedding and LLM providers — ask questions in natural language over curated documents, and measure retrieval quality with inline Precision, Recall, MRR and NDCG scores.

🔗 [Live Demo](https://rag-eval-pipeline.vercel.app) · [GitHub](https://github.com/ShrutiTiwari/rag-eval-pipeline) · [Parent Project](https://www.powerparent.co.uk)

---

## What it does

Ask questions in natural language over two curated documents:

- **ABRSM Piano 2025–2026 syllabus** — *"What pieces are required for Grade 3 piano?"*
- **St Paul's Juniors 11+ entrance syllabus** — *"What subjects are in the ISEB Common Pre-Test?"*

The system retrieves the most relevant chunks using semantic similarity, feeds them as context to an LLM, and returns an answer with source attribution — showing exactly which part of the document it drew from.

Select a document from the dashboard — chat and vector search are scoped to that document only. No cross-contamination between docs.

What makes this more than a basic chatbot demo is the **evaluation layer**: every retrieval is measurable. The dashboard exposes Precision, Recall, F1, MRR, and NDCG scores broken down by query difficulty — so you can see not just that it answers, but how well it retrieves.

---

## Why I built this

I have two children working toward ABRSM piano grade exams and preparing for 11+ entrance exams. Both involve dense PDFs that parents and students spend a lot of time hunting through. A natural language interface over those documents is genuinely useful.

Beyond the practical use case, this project exists to showcase the full RAG engineering stack — not just the "ask a question, get an answer" surface, but the retrieval pipeline, the evaluation methodology, and the architectural decisions that affect quality.

---

## How the RAG pipeline works

```
PDF Document
    │
    ▼
DocumentLoader        — extracts text, cleans, preserves metadata (pages, size)
    │
    ▼
ChunkingStrategy      — Fixed, SectionAware, or GradeBoundary (swappable via env var)
    │
    ▼
EmbeddingProvider     — generates vectors via local model or OpenAI (swappable)
    │
    ▼
VectorStore           — hybrid BM25 + cosine similarity search over in-memory embeddings
    │
    ▼
ChatService           — retrieves top-K chunks, injects as context, calls LLM
    │
    ▼
Response + Sources    — answer with chunk references and similarity scores
```

Conversation history is maintained across turns so follow-up questions work naturally.

---

## Provider abstraction — swap models via env vars

Embedding, LLM, and chunking backends are all abstracted behind provider interfaces. No code changes needed to switch:

```bash
# Embeddings (default: local, free, no API key needed)
EMBEDDING_PROVIDER=local        # Xenova/all-MiniLM-L6-v2 — runs entirely in Node.js
EMBEDDING_PROVIDER=openai       # text-embedding-3-small — requires OPENAI_API_KEY

# LLM chat (default: claude)
LLM_PROVIDER=claude             # Claude Haiku — requires ANTHROPIC_API_KEY
LLM_PROVIDER=openai             # gpt-4o-mini  — requires OPENAI_API_KEY

# Chunking strategy (default: fixed)
CHUNKING_STRATEGY=fixed         # overlapping fixed-size chunks (1000 chars, 100 overlap)
CHUNKING_STRATEGY=sectionaware  # splits on headings and section boundaries
CHUNKING_STRATEGY=gradeboundary # splits on grade/level boundaries (ABRSM, 11+ content)
```

The default embedding pipeline runs locally with no embedding API cost; answer generation uses Claude or OpenAI via the configured provider.

---

## Retrieval evaluation

`RetrievalEvaluator` runs a test suite of queries with known expected sources and computes:

| Metric | What it measures |
|---|---|
| **Precision** | Of the chunks retrieved, how many were actually relevant |
| **Recall** | Of the relevant chunks, how many were retrieved |
| **F1** | Harmonic mean of precision and recall |
| **MRR** | Was the most relevant chunk ranked first? |
| **NDCG** | Full ranking quality — rewards putting better results higher |

`ElevenPlusEvaluator` provides 12 content-specific queries for the 11+ syllabus across three difficulty levels (easy / medium / hard) and four query types (factual, section_detail, exclusion, late_content). The hard queries deliberately test content from the latter half of the document to catch chunk truncation issues.

`/api/rag-docs/compare` lets you A/B test configurations — chunk sizes, topK, similarity thresholds — against the same query set.

---

## Pipeline tests

133 unit tests across 6 files, all scoped to the 11+ document. Uses Node's built-in `node:test` — no test framework needed.

```
npm run test:loader      # 27 tests — DocumentLoader: load, cleanText, chunkDocument
npm run test:embedding   # 16 tests — EmbeddingProvider: dimensions, normalisation, determinism
npm run test:vector      # 17 tests — VectorStore: search, threshold, content relevance
npm run test:chat        # 20 tests — ChatService: init, context assembly, history, fallback + smoke test
npm run test:evaluator   # 18 tests — ElevenPlusEvaluator: query structure, metrics, recall by difficulty
npm run test:chunking    # 35 tests — ChunkingStrategy: Fixed, SectionAware, GradeBoundary comparison
npm test                 # all 133
```

Tests use the local embedding model (no API key needed) except the smoke test in `04-chat-service` which calls Claude.

---

## Technical decisions

**Local embeddings** — `Xenova/all-MiniLM-L6-v2` runs entirely in Node.js via `@xenova/transformers`. No OpenAI credits needed for embeddings. Downloads ~25MB on first run, cached thereafter.

**Per-document scoping** — the vector store is initialised per selected document. Clicking "Chat" on the 11+ card never embeds or searches the ABRSM doc, and vice versa. Each doc gets its own cached `ChatService` instance (1-hour TTL).

**Three chunking strategies** — `Fixed` uses overlapping 1000-char chunks (100-char overlap). `SectionAware` splits on headings and structural boundaries to keep sections intact. `GradeBoundary` splits on grade/level markers — purpose-built for ABRSM and 11+ content. Swap via `CHUNKING_STRATEGY` env var; the `/compare` endpoint lets you A/B test strategies against the same query set.

**Hybrid search** — `VectorStore` combines cosine similarity (semantic) with BM25 (keyword) using a weighted score. This improves recall for exact-term queries that pure semantic search can miss.

**In-memory vector store** — no external vector database. Simpler and fast for single-document use. Embeddings are regenerated on cold start; caching mitigates cost in practice.

**TruLensClient** — stub interface to a TruLens Python evaluation service for LLM-level metrics (groundedness, answer relevance). Instrumented and ready; the service itself is the next milestone.

---

## Project structure

```
rag-eval-pipeline/
├── server.js                        # Express entry point (~30 lines)
├── public/
│   ├── index.html                   # document dashboard UI
│   └── pdf-sources/
│       ├── piano-syllabus.pdf
│       └── 11-plus-syllabus.pdf
├── src/
│   ├── core/
│   │   ├── DocumentLoader.js        # PDF loading, cleaning, chunking
│   │   ├── VectorStore.js           # embeddings + hybrid BM25/cosine search
│   │   ├── ChatService.js           # RAG orchestration + conversation history
│   │   ├── RetrievalEvaluator.js    # generic Precision/Recall/F1/MRR/NDCG
│   │   ├── ElevenPlusEvaluator.js   # 11+ specific evaluator with content queries
│   │   └── TruLensClient.js         # interface to Python evaluation service
│   ├── providers/
│   │   ├── EmbeddingProvider.js     # Local + OpenAI embedding backends
│   │   ├── LLMProvider.js           # Claude + OpenAI LLM backends
│   │   └── ChunkingStrategy.js      # Fixed, SectionAware, GradeBoundary strategies
│   └── routes/
│       └── documentViewerRoutes.js  # all API endpoints + HTML UIs
└── test/
    └── 11-plus/
        ├── 01-document-loader.test.js
        ├── 02-embedding.test.js
        ├── 03-vector-store.test.js
        ├── 04-chat-service.test.js
        ├── 05-evaluator.test.js
        └── 06-chunking-strategy.test.js
```

---

## API endpoints

| Endpoint | What it does |
|---|---|
| `GET /` | Document dashboard — select a doc to scope chat/search |
| `GET /api/rag-docs` | Document list + per-doc stats |
| `POST /api/rag-docs/chat` | RAG chat (requires `doc` param) |
| `GET /api/rag-docs/chat/test?doc=` | Interactive chat UI scoped to a document |
| `POST /api/rag-docs/vector-search` | Semantic search (requires `doc` param) |
| `GET /api/rag-docs/vector-search/test?doc=` | Interactive search UI |
| `GET /api/rag-docs/evaluate?doc=` | Retrieval quality dashboard |
| `POST /api/rag-docs/compare` | A/B test retrieval configurations |
| `GET /api/rag-docs/:id/chunks` | Inspect chunking for any document |
| `GET /api/rag-docs/:id/view` | Full document viewer with text search |

---

## Running locally

```bash
git clone https://github.com/ShrutiTiwari/rag-eval-pipeline
cd rag-eval-pipeline
npm install
cp .env.example .env   # add ANTHROPIC_API_KEY (Claude) — no OpenAI key needed by default
npm run dev            # http://localhost:3001
```

On first request to chat or search, the local embedding model downloads (~25MB) and initialises. Subsequent requests are fast.

```bash
# Optional: override defaults via .env
CHUNKING_STRATEGY=gradeboundary   # fixed (default) | sectionaware | gradeboundary
EMBEDDING_PROVIDER=local          # local (default) | openai
LLM_PROVIDER=claude               # claude (default) | openai
```

```bash
npm test                  # run all 133 pipeline tests
npm run test:chunking     # chunking strategy comparison tests only
```

---

## What I'd build next

**TruLens Python service** — groundedness and answer relevance metrics (does the answer actually follow from the retrieved context?) are more meaningful than retrieval metrics alone. The client is already instrumented; the service needs building.

**Ollama LLM provider** — add a third provider backend using Ollama for fully offline chat (local embeddings + local LLM = zero external API calls).

**Chunk size experimentation UI** — the `/compare` endpoint supports A/B testing programmatically; a UI to set parameters and visualise metric differences interactively would make this a proper RAG tuning tool.

**Multi-document RAG** — cross-document questions like *"What topics appear in both the ABRSM Grade 4 syllabus and the 11+ maths syllabus?"*

---

## About the builder

Built by Shruti Tiwari — independent AI product builder (2023–present), 20 years backend engineering, former VP at Goldman Sachs. I build AI-native tools for music education and parenting, rooted in my own experience as a piano learner and parent of two children working toward ABRSM piano exams and 11+ entrance exams.

This project is part of [PowerParent](https://www.powerparent.co.uk) — a platform for parents and teachers to track music practice, manage school events, and support children's learning with AI.
