# 📊 LLM-Powered RAG Survey Response Summarizer

![CI](https://github.com/SLekhana/rag-survey-summarizer/actions/workflows/ci.yml/badge.svg)
![Coverage](https://codecov.io/gh/SLekhana/rag-survey-summarizer/branch/main/graph/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![License](https://img.shields.io/badge/license-MIT-blue)

End-to-end RAG system for semantic search and executive summarization of 100K+ survey responses. Combines **BM25 sparse retrieval + FAISS dense retrieval** via Reciprocal Rank Fusion, with a **cross-encoder reranker**, **OpenAI GPT** for structured theme extraction, and **LangChain agents** for multi-step reasoning.

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
│        → ChromaDB ───────┘         │                    │
│                                    ▼                    │
│              Cross-Encoder Reranker (ms-marco)          │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│               LLM GENERATION + GUARDRAILS                │
│  Chunks → Prompt Template (few-shot + CoT) → GPT-4o-mini│
│        → JSON: executive_summary + themes               │
│        → Guardrails (input/output validation)           │
│        → ROUGE + IR metrics + hallucination score       │
│        → Cost tracking (per-query, by model)            │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│              FASTAPI + STREAMLIT DASHBOARD               │
│  /ingest  /search  /summarize  /evaluate  /agent        │
│  /costs   /rerank                                       │
│  Usage tracking • Latency monitoring • Schema versioning │
└─────────────────────────────────────────────────────────┘
```

---

## 📈 Results

| Retrieval Mode | ROUGE-1 | Recall@10 | MRR | Theme Accuracy | Latency |
|---|---|---|---|---|---|
| TF-IDF Baseline | — | — | — | 0.52 | ~50ms |
| Sparse (BM25) | 0.41 | 0.71 | 0.63 | 0.61 | ~80ms |
| Dense (FAISS IVF) | 0.48 | 0.83 | 0.74 | 0.68 | ~120ms |
| **Hybrid (BM25 + FAISS)** | **0.54** | **0.91** | **0.82** | **0.72** | ~150ms |
| Hybrid + Reranker | 0.57 | **0.96** | **0.89** | 0.75 | ~280ms |

**FAISS Tuning:** nlist=64, nprobe=8 → 96% Recall@10 at 3.4ms  
**+20% theme detection accuracy** over TF-IDF baseline | **50% reduction** in manual review time

---

## 💰 Cost Analysis

| Model | $/query | $/1K queries | $/100K queries |
|---|---|---|---|
| gpt-3.5-turbo | $0.001085 | $1.09 | $108.50 |
| **gpt-4o-mini** ✅ | **$0.000378** | **$0.38** | **$37.80** |
| gpt-4o | $0.006300 | $6.30 | $630.00 |
| gpt-4-turbo | $0.021700 | $21.70 | $2,170.00 |

**With 40% Redis cache hit rate:** gpt-4o-mini effective cost = **$0.23/1K queries**  
**Weekly batch (100K NPS responses):** $22.68/week = **$1,179/year**

> Recommendation: `gpt-4o-mini` — best cost/quality ratio, JSON mode supported, <2s latency

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) |
| Dense Index | FAISS IVF (nlist=64, nprobe=8) |
| Sparse Index | BM25 (`rank-bm25`) |
| Hybrid Fusion | Reciprocal Rank Fusion (RRF) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Persistent Store | ChromaDB |
| Generation | OpenAI GPT-4o-mini / GPT-3.5 / GPT-4 |
| Agents | LangChain (tool-calling: SQL + semantic search) |
| Evaluation | ROUGE-1/2/L, Recall@k, MRR, nDCG, Theme Accuracy, Hallucination Score |
| Cost Tracking | Per-query token accounting + Redis cache simulation |
| API | FastAPI + Pydantic v2 |
| UI | Streamlit |
| Infra | Docker, docker-compose, PostgreSQL |
| CI/CD | GitHub Actions (lint → test → build → coverage) |

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/SLekhana/rag-survey-summarizer
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

**Or run locally (Python 3.11 recommended):**
```bash
conda create -n rag-env python=3.11 -y
conda activate rag-env
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

### POST `/rerank`
Cross-encoder reranking of retrieved chunks.
```json
{"query": "delivery issues", "top_k": 10, "rerank_top_k": 5}
```

### POST `/summarize`
Full RAG pipeline → executive summary + themes.
```json
{"query": "What are the main customer pain points?", "max_themes": 5}
```

### POST `/evaluate`
Controlled experiment: compare sparse vs dense vs hybrid vs TF-IDF.
Returns ROUGE, Recall@k, MRR, nDCG, theme accuracy, and hallucination scores.
```json
{
  "query": "customer feedback themes",
  "ground_truth_themes": ["delivery", "quality", "support"]
}
```

### GET `/costs`
Per-model cost breakdown for all queries in the current session.

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
```

**59 tests | 76%+ coverage | All passing**

| Test Class | What It Covers |
|---|---|
| `TestChunking` | Text splitting, overlap, schema versioning |
| `TestBM25Retriever` | Sparse search, ranking correctness |
| `TestFAISSIndex` | IVF build, search, save/load, incremental add |
| `TestRRF` | Hybrid fusion, score ordering |
| `TestROUGE` | F1 scoring, partial/full/no match |
| `TestThemeAccuracy` | TF-IDF cosine similarity matching |
| `TestHallucinationDetector` | Pattern + grounding scoring |
| `TestGuardrails` | Input/output content filtering |
| `TestAPIEndpoints` | Full flow: ingest → search → evaluate |
| `TestIRMetrics` | Recall@k, MRR, nDCG correctness |
| `TestCostTracker` | Per-query token accounting, model breakdown |
| `TestCostEndpoint` | `/costs` endpoint schema validation |

---

## 🔬 Evaluation Design

The `/evaluate` endpoint runs a **controlled experiment** across all retrieval strategies:

```python
modes = ["sparse", "dense", "hybrid", "tfidf_baseline"]

for mode in modes:
    results = retrieve(query, mode=mode)
    reranked = cross_encoder_rerank(query, results)
    summary = generate(reranked)

    metrics = {
        "rouge":         compute_rouge(summary, ground_truth),
        "recall_at_k":   compute_recall_at_k(retrieved_ids, relevant_ids, k=10),
        "mrr":           compute_mrr(retrieved_ids, relevant_ids),
        "ndcg":          compute_ndcg(retrieved_ids, relevant_ids),
        "theme_acc":     compute_theme_accuracy(predicted_themes, gt_themes),
        "hallucination": compute_hallucination_score(summary, source_chunks),
    }
```

---

## 📊 Benchmarks

Run standalone benchmark scripts:

```bash
# FAISS index tuning (nlist, nprobe sweep)
python benchmarks/faiss_tuning.py

# Cost analysis across models
python benchmarks/cost_analysis.py
```

Results are saved to `benchmarks/cost_analysis_results.json`.

---

## 📁 Project Structure

```
rag-survey-summarizer/
├── app/
│   ├── main.py                  # FastAPI application + all endpoints
│   ├── core/
│   │   ├── config.py            # Settings (pydantic-settings)
│   │   ├── retrieval.py         # BM25 + FAISS + RRF hybrid
│   │   ├── reranker.py          # Cross-encoder reranking (ms-marco)
│   │   ├── generation.py        # OpenAI + LangChain agents + guardrails
│   │   └── evaluation.py        # ROUGE, Recall@k, MRR, nDCG, cost tracking
│   ├── pipeline/
│   │   ├── ingestion.py         # Chunking + sentence-transformer embeddings
│   │   └── indexing.py          # FAISS IVF + ChromaDB persistent store
│   └── models/
│       └── schemas.py           # Pydantic request/response models
├── benchmarks/
│   ├── faiss_tuning.py          # FAISS nlist/nprobe parameter sweep
│   └── cost_analysis.py         # Per-model cost projection
├── streamlit_app/
│   └── app.py                   # Interactive dashboard
├── tests/
│   └── test_all.py              # 59-test suite (all passing)
├── .github/workflows/
│   └── ci.yml                   # Lint → test → build → coverage (≥75%)
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## 🔒 Guardrails

- **Input**: blocks PII-related queries, off-topic content
- **Output**: detects absolute language ("always", "never"), speculative phrases
- **Prompting**: low temperature (0.1) + `response_format: json_object` for deterministic JSON
- **Few-shot + CoT**: structured examples guide consistent theme extraction

---

## 🔁 CI/CD Pipeline

GitHub Actions runs on every push to `main`:

1. **Lint** — `ruff` (unused imports) + `black==24.10.0` (formatting)
2. **Test** — `pytest` with PostgreSQL service container
3. **Coverage** — enforced ≥ 75% via `pytest-cov`
4. **Build** — Docker image build + inspect
