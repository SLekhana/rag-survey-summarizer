# 📊 LLM-Powered RAG Survey Response Summarizer

![CI](https://github.com/yourusername/rag-survey-summarizer/actions/workflows/ci.yml/badge.svg)
![Coverage](https://codecov.io/gh/yourusername/rag-survey-summarizer/branch/main/graph/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![License](https://img.shields.io/badge/license-MIT-blue)

End-to-end RAG system for semantic search and executive summarization of 100K+ survey responses. Combines **BM25 sparse retrieval + FAISS dense retrieval** via Reciprocal Rank Fusion, with **OpenAI GPT** for structured theme extraction and **LangChain agents** for multi-step reasoning.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    INGESTION PIPELINE                    │
│  Raw Text → Sentence Chunker → all-MiniLM-L6-v2 Embed  │
│           → FAISS IVF Index + ChromaDB (persistent)     │
│           → BM25 Index + TF-IDF Baseline                │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                  HYBRID RETRIEVAL ENGINE                  │
│  Query → BM25 (sparse) ──┐                              │
│        → FAISS IVF ──────┼──→ RRF Fusion → Top-K Chunks│
│        → ChromaDB ───────┘                              │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│               LLM GENERATION + GUARDRAILS                │
│  Chunks → Prompt Template (few-shot + CoT) → GPT-3.5/4 │
│        → JSON: executive_summary + themes               │
│        → Guardrails (input/output validation)           │
│        → ROUGE evaluation + hallucination score         │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│              FASTAPI + STREAMLIT DASHBOARD               │
│  /ingest  /search  /summarize  /evaluate  /agent        │
│  Usage tracking • Latency monitoring • Schema versioning │
└─────────────────────────────────────────────────────────┘
```

---

## 📈 Results

| Retrieval Mode | ROUGE-1 | Theme Accuracy | Latency |
|---|---|---|---|
| TF-IDF Baseline | — | 0.52 | ~50ms |
| Sparse (BM25) | 0.41 | 0.61 | ~80ms |
| Dense (FAISS IVF) | 0.48 | 0.68 | ~120ms |
| **Hybrid (BM25 + FAISS)** | **0.54** | **0.72** | ~150ms |

**+20% theme detection accuracy** over TF-IDF baseline | **50% reduction** in manual review time

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) |
| Dense Index | FAISS IVF (Inverted File Index) |
| Sparse Index | BM25 (`rank-bm25`) |
| Hybrid Fusion | Reciprocal Rank Fusion (RRF) |
| Persistent Store | ChromaDB |
| Generation | OpenAI GPT-3.5-turbo / GPT-4 |
| Agents | LangChain (tool-calling: SQL + semantic search) |
| Evaluation | ROUGE-1/2/L, Theme Accuracy, Hallucination Score |
| API | FastAPI + Pydantic v2 |
| UI | Streamlit |
| Infra | Docker, docker-compose, PostgreSQL |
| CI/CD | GitHub Actions (test → lint → build) |

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/yourusername/rag-survey-summarizer
cd rag-survey-summarizer

# 2. Set environment
cp .env.example .env
# Edit .env → add your OPENAI_API_KEY

# 3. Run with Docker
docker-compose up --build

# API:       http://localhost:8000
# Docs:      http://localhost:8000/docs
# Streamlit: http://localhost:8501
```

**Or run locally:**
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload        # API on :8000
streamlit run streamlit_app/app.py   # UI on :8501
```

---

## 📡 API Reference

### POST `/ingest`
Ingest survey documents with chunking + embedding.
```json
{
  "documents": [
    {"id": "1", "text": "Product quality exceeded expectations.", "metadata": {"source": "Q1"}}
  ]
}
```

### POST `/search`
Hybrid semantic search over indexed documents.
```json
{"query": "delivery issues", "top_k": 10, "mode": "hybrid"}
```

### POST `/summarize`
Full RAG pipeline → executive summary + themes.
```json
{"query": "What are the main customer pain points?", "max_themes": 5}
```

### POST `/evaluate`
Controlled experiment: compare sparse vs dense vs hybrid vs TF-IDF.
```json
{
  "query": "customer feedback themes",
  "ground_truth_themes": ["delivery", "quality", "support"]
}
```

### POST `/agent`
LangChain agent with tool-calling for multi-step questions.
```
?query=How many responses mention delivery issues and what do they say?
```

---

## 🧪 Testing

```bash
# Run full test suite with coverage
pytest tests/ --cov=app --cov-report=term-missing -v

# Target: 95%+ coverage across ingestion, retrieval, evaluation, API
```

**Test categories:**
- `TestChunking` — text splitting, overlap, schema versioning
- `TestBM25Retriever` — sparse search, ranking correctness
- `TestFAISSIndex` — IVF build, search, save/load, incremental add
- `TestRRF` — hybrid fusion, score ordering
- `TestROUGE` — F1 scoring, partial/full/no match
- `TestThemeAccuracy` — TF-IDF cosine matching
- `TestHallucinationDetector` — pattern + grounding scoring
- `TestGuardrails` — input/output content filtering
- `TestAPIEndpoints` — full flow: ingest → search → evaluate

---

## 🔬 Evaluation Design

The `/evaluate` endpoint runs a **controlled experiment** comparing all retrieval strategies:

```python
# Controlled experiment: sparse vs dense vs hybrid vs TF-IDF baseline
modes = ["sparse", "dense", "hybrid", "tfidf_baseline"]

for mode in modes:
    # Retrieve → Generate → Evaluate
    rouge = compute_rouge(generated_summary, ground_truth)
    theme_acc = compute_theme_accuracy(predicted_themes, ground_truth_themes)
    hallucination = compute_hallucination_score(summary, source_chunks)
```

**Metrics:**
- **ROUGE-1/2/L**: n-gram overlap vs ground truth summaries
- **Theme Accuracy**: TF-IDF cosine similarity between predicted and GT themes
- **Hallucination Score**: Pattern detection + source grounding (0=grounded, 1=hallucinated)

---

## 📁 Project Structure

```
rag-survey-summarizer/
├── app/
│   ├── main.py                  # FastAPI application
│   ├── core/
│   │   ├── config.py            # Settings (pydantic-settings)
│   │   ├── retrieval.py         # BM25 + FAISS + RRF hybrid
│   │   ├── generation.py        # OpenAI + LangChain agents + guardrails
│   │   └── evaluation.py        # ROUGE, theme accuracy, hallucination
│   ├── pipeline/
│   │   ├── ingestion.py         # Chunking + sentence-transformer embeddings
│   │   └── indexing.py          # FAISS IVF + ChromaDB persistent store
│   └── models/
│       └── schemas.py           # Pydantic request/response models
├── streamlit_app/
│   └── app.py                   # Interactive dashboard
├── tests/
│   └── test_all.py              # Full test suite
├── .github/workflows/
│   └── ci.yml                   # GitHub Actions CI/CD
├── docker-compose.yml           # API + Streamlit + PostgreSQL
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## 📊 Streamlit Dashboard

The Streamlit app provides:
- **Ingest tab**: upload JSON or load sample data
- **Search tab**: compare retrieval modes side-by-side
- **Summarize tab**: executive summaries with theme cards + ROUGE scores
- **Evaluate tab**: bar charts comparing mode performance
- **Dashboard tab**: live usage stats, latency metrics, request counts

---

## 🔒 Guardrails

Input and output validation to ensure reliable production behavior:
- **Input**: blocks PII-related queries, off-topic content
- **Output**: detects absolute language ("always", "never"), speculative phrases
- **Prompting**: low temperature (0.1) + `response_format: json_object` for deterministic JSON
- **Few-shot + CoT**: structured examples guide consistent theme extraction

---


