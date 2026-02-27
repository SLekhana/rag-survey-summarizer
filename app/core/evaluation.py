"""
Evaluation Module — Production-Grade RAG Metrics
─────────────────────────────────────────────────
Classic NLP:
  - ROUGE-1, ROUGE-2, ROUGE-L

Information Retrieval (new):
  - Recall@k     : fraction of relevant docs retrieved in top-k
  - MRR          : Mean Reciprocal Rank of first relevant result
  - nDCG@k       : Normalized Discounted Cumulative Gain

LLM-as-Judge (new):
  - Faithfulness : is the summary grounded in the source?
  - Relevance    : does it answer the query?
  - Coherence    : is it well-structured and readable?

Other:
  - Theme detection accuracy (TF-IDF cosine matching)
  - Hallucination score (pattern + overlap heuristic)
  - Cost tracking per query
  - TF-IDF baseline comparison
"""

import time
import math
import logging
import re
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────
# ROUGE EVALUATION
# ──────────────────────────────────────────────────

def compute_rouge(hypothesis: str, references: List[str]) -> Dict[str, float]:
    """
    Compute ROUGE-1, ROUGE-2, ROUGE-L F1 scores.
    hypothesis: generated summary
    references: ground truth summaries (best match used)
    """
    if not hypothesis or not references:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    best_scores = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for ref in references:
        scores = scorer.score(ref, hypothesis)
        for key in best_scores:
            best_scores[key] = max(best_scores[key], scores[key].fmeasure)

    return best_scores


# ──────────────────────────────────────────────────
# IR METRICS: Recall@k, MRR, nDCG@k
# ──────────────────────────────────────────────────

def compute_recall_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int
) -> float:
    """
    Recall@k = |retrieved[:k] ∩ relevant| / |relevant|

    Measures: what fraction of relevant docs did we find in top-k?
    Perfect score = 1.0 means all relevant docs are in top-k results.
    """
    if not relevant_ids:
        return 0.0
    retrieved_top_k = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    return len(retrieved_top_k & relevant_set) / len(relevant_set)


def compute_mrr(
    retrieved_ids: List[str],
    relevant_ids: List[str]
) -> float:
    """
    Mean Reciprocal Rank = 1 / rank_of_first_relevant_result

    MRR = 1.0 means the first result is always relevant.
    MRR = 0.5 means the first relevant result is typically at rank 2.
    """
    relevant_set = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def compute_ndcg_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int
) -> float:
    """
    nDCG@k = DCG@k / IDCG@k

    Discounted Cumulative Gain rewards retrieving relevant docs early.
    nDCG = 1.0 means perfect ranking (all relevant docs at top positions).
    Uses binary relevance: 1 if doc is relevant, 0 otherwise.
    """
    relevant_set = set(relevant_ids)

    def dcg(ranked_ids: List[str], cutoff: int) -> float:
        score = 0.0
        for i, doc_id in enumerate(ranked_ids[:cutoff], start=1):
            rel = 1 if doc_id in relevant_set else 0
            score += rel / math.log2(i + 1)
        return score

    actual_dcg = dcg(retrieved_ids, k)
    ideal_order = list(relevant_set) + [d for d in retrieved_ids if d not in relevant_set]
    ideal_dcg = dcg(ideal_order, k)

    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def compute_ir_metrics(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k_values: List[int] = None
) -> Dict[str, float]:
    """
    Compute full IR metric suite: Recall@k, MRR, nDCG@k for multiple k values.

    Args:
        retrieved_ids: ordered list of retrieved document IDs (ranked)
        relevant_ids: list of ground-truth relevant document IDs
        k_values: list of cutoff values (default: [1, 3, 5, 10])
    """
    k_values = k_values or [1, 3, 5, 10]
    metrics: Dict[str, float] = {}

    for k in k_values:
        metrics[f"recall@{k}"] = round(compute_recall_at_k(retrieved_ids, relevant_ids, k), 4)
        metrics[f"ndcg@{k}"] = round(compute_ndcg_at_k(retrieved_ids, relevant_ids, k), 4)

    metrics["mrr"] = round(compute_mrr(retrieved_ids, relevant_ids), 4)
    return metrics


# ──────────────────────────────────────────────────
# LLM-AS-JUDGE EVALUATION
# ──────────────────────────────────────────────────

LLM_JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for AI-generated summaries.
Score a summary on three dimensions. Be rigorous and objective.
Respond ONLY with valid JSON, no preamble."""

LLM_JUDGE_PROMPT = """Evaluate this AI-generated summary of survey responses.

QUERY: {query}

SOURCE RESPONSES (retrieved context):
{context}

GENERATED SUMMARY:
{summary}

Score each dimension from 0.0 to 1.0:

1. faithfulness: Is every claim in the summary grounded in the source responses?
   1.0 = all claims traceable to sources, 0.0 = significant hallucinations

2. relevance: Does the summary directly address the query?
   1.0 = perfectly addresses the query, 0.0 = misses the point

3. coherence: Is the summary well-structured and readable?
   1.0 = clear, concise, professional executive language, 0.0 = confusing

Respond ONLY with JSON:
{{
  "faithfulness": 0.0,
  "relevance": 0.0,
  "coherence": 0.0,
  "reasoning": "brief explanation"
}}"""


@dataclass
class LLMJudgeResult:
    faithfulness: float = 0.0
    relevance: float = 0.0
    coherence: float = 0.0
    overall: float = 0.0
    reasoning: str = ""
    cost_usd: float = 0.0
    model: str = ""
    error: Optional[str] = None


def run_llm_judge(
    query: str,
    summary: str,
    source_texts: List[str],
    openai_client=None,
    model: str = "gpt-4o-mini",
) -> LLMJudgeResult:
    """
    Use GPT-4o-mini as an impartial judge to score summary quality.

    Evaluates:
    - Faithfulness: grounded in source? (anti-hallucination)
    - Relevance: answers the query?
    - Coherence: professional, readable?

    Cost: ~$0.001-0.003 per call with gpt-4o-mini.
    Falls back gracefully if no API key / network issue.
    """
    if openai_client is None:
        try:
            from openai import OpenAI
            from app.core.config import settings
            openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
        except Exception as e:
            return LLMJudgeResult(error=f"No OpenAI client: {e}")

    context = "\n".join([f"[{i+1}] {t}" for i, t in enumerate(source_texts[:5])])
    prompt = LLM_JUDGE_PROMPT.format(query=query, context=context, summary=summary)

    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": LLM_JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=300,
            response_format={"type": "json_object"}
        )

        raw = response.choices[0].message.content
        data = json.loads(raw)

        faithfulness = float(data.get("faithfulness", 0.0))
        relevance = float(data.get("relevance", 0.0))
        coherence = float(data.get("coherence", 0.0))
        overall = round((faithfulness + relevance + coherence) / 3, 4)

        # Cost estimation (gpt-4o-mini pricing)
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        cost_usd = (input_tokens * 0.00000015) + (output_tokens * 0.0000006)

        return LLMJudgeResult(
            faithfulness=round(faithfulness, 4),
            relevance=round(relevance, 4),
            coherence=round(coherence, 4),
            overall=overall,
            reasoning=data.get("reasoning", ""),
            cost_usd=round(cost_usd, 6),
            model=model
        )

    except Exception as e:
        logger.error(f"LLM judge failed: {e}")
        return LLMJudgeResult(error=str(e))


# ──────────────────────────────────────────────────
# COST TRACKING
# ──────────────────────────────────────────────────

COST_PER_TOKEN = {
    "gpt-3.5-turbo": {"input": 0.0000005, "output": 0.0000015},
    "gpt-4o-mini":   {"input": 0.00000015, "output": 0.0000006},
    "gpt-4o":        {"input": 0.0000025, "output": 0.00001},
    "gpt-4-turbo":   {"input": 0.00001, "output": 0.00003},
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate USD cost for an OpenAI API call."""
    pricing = COST_PER_TOKEN.get(model, COST_PER_TOKEN["gpt-4o-mini"])
    return (input_tokens * pricing["input"]) + (output_tokens * pricing["output"])


@dataclass
class QueryCostRecord:
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    endpoint: str
    timestamp: float = field(default_factory=time.time)


class CostTracker:
    """
    Tracks token usage and cost per query.
    Answers: 'What is your cost per 1K summaries?'

    Typical gpt-4o-mini usage:
      - Input: ~800 tokens, Output: ~300 tokens
      - Cost per query: ~$0.0003
      - Cost per 1K queries: ~$0.30
    """

    def __init__(self):
        self.records: List[QueryCostRecord] = []

    def record(self, model: str, input_tokens: int, output_tokens: int, endpoint: str = "summarize"):
        cost = estimate_cost(model, input_tokens, output_tokens)
        self.records.append(QueryCostRecord(
            model=model, input_tokens=input_tokens,
            output_tokens=output_tokens, cost_usd=cost, endpoint=endpoint
        ))
        return cost

    def summary(self) -> Dict[str, Any]:
        if not self.records:
            return {"total_queries": 0, "total_cost_usd": 0.0}

        total_cost = sum(r.cost_usd for r in self.records)
        total_queries = len(self.records)
        avg_cost = total_cost / total_queries

        by_model: Dict[str, Dict] = {}
        for r in self.records:
            if r.model not in by_model:
                by_model[r.model] = {"queries": 0, "total_cost": 0.0, "total_tokens": 0}
            by_model[r.model]["queries"] += 1
            by_model[r.model]["total_cost"] += r.cost_usd
            by_model[r.model]["total_tokens"] += r.input_tokens + r.output_tokens

        return {
            "total_queries": total_queries,
            "total_cost_usd": round(total_cost, 6),
            "avg_cost_per_query_usd": round(avg_cost, 6),
            "estimated_cost_per_1k_queries_usd": round(avg_cost * 1000, 4),
            "by_model": {
                m: {
                    "queries": v["queries"],
                    "total_cost_usd": round(v["total_cost"], 6),
                    "avg_tokens_per_query": round(v["total_tokens"] / v["queries"])
                }
                for m, v in by_model.items()
            }
        }


# Global cost tracker
cost_tracker = CostTracker()


# ──────────────────────────────────────────────────
# THEME DETECTION ACCURACY
# ──────────────────────────────────────────────────

def compute_theme_accuracy(
    predicted_themes: List[str],
    ground_truth_themes: List[str],
    threshold: float = 0.5
) -> float:
    """
    Compute theme detection accuracy via TF-IDF cosine similarity matching.
    Returns: recall-like accuracy (% of ground truth themes detected)
    """
    if not predicted_themes or not ground_truth_themes:
        return 0.0

    all_themes = predicted_themes + ground_truth_themes
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    try:
        tfidf_matrix = vectorizer.fit_transform(all_themes)
    except Exception:
        return 0.0

    pred_matrix = tfidf_matrix[:len(predicted_themes)]
    gt_matrix = tfidf_matrix[len(predicted_themes):]

    sim_matrix = cosine_similarity(gt_matrix, pred_matrix)
    matched = sum(1 for row in sim_matrix if row.max() >= threshold)

    return matched / len(ground_truth_themes)


# ──────────────────────────────────────────────────
# HALLUCINATION DETECTION
# ──────────────────────────────────────────────────

HALLUCINATION_PATTERNS = [
    r'\b(always|never|all|every|none|everyone|nobody)\b',
    r'\b(definitely|certainly|absolutely|guaranteed)\b',
    r'\b(\d+%)\b',
    r'\b(research shows|studies indicate|experts say)\b',
]


def compute_hallucination_score(generated_text: str, source_texts: List[str]) -> float:
    """
    Estimate hallucination likelihood. Score 0.0=grounded, 1.0=hallucinated.
    Combine with LLM-as-judge faithfulness for production use.
    """
    pattern_flags = sum(
        1 for p in HALLUCINATION_PATTERNS
        if re.search(p, generated_text, re.IGNORECASE)
    )
    gen_words = set(generated_text.lower().split())
    source_words = set(" ".join(source_texts).lower().split())
    overlap = len(gen_words & source_words) / len(gen_words) if gen_words else 0.0

    pattern_score = min(pattern_flags / len(HALLUCINATION_PATTERNS), 1.0)
    hallucination_score = 0.4 * pattern_score + 0.6 * (1 - overlap)
    return round(hallucination_score, 3)


# ──────────────────────────────────────────────────
# TF-IDF BASELINE
# ──────────────────────────────────────────────────

class TFIDFBaseline:
    """TF-IDF retrieval baseline for comparison against hybrid RAG."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), stop_words="english", max_features=50000
        )
        self.tfidf_matrix = None
        self.corpus_ids: List[str] = []
        self.corpus_texts: List[str] = []

    def fit(self, chunk_ids: List[str], texts: List[str]):
        self.corpus_ids = chunk_ids
        self.corpus_texts = texts
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        logger.info(f"TF-IDF baseline fitted on {len(texts)} documents")

    def search(self, query: str, top_k: int = 10) -> List[tuple]:
        if self.tfidf_matrix is None:
            return []
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.corpus_ids[i], float(scores[i])) for i in top_indices if scores[i] > 0]


# ──────────────────────────────────────────────────
# EXPERIMENT RUNNER
# ──────────────────────────────────────────────────

@dataclass
class ExperimentResult:
    mode: str
    rouge_1: float
    rouge_2: float
    rouge_l: float
    theme_accuracy: float
    hallucination_score: float
    latency_ms: float
    retrieved_count: int
    # IR metrics
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr: float = 0.0
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    # LLM judge
    llm_faithfulness: Optional[float] = None
    llm_relevance: Optional[float] = None
    llm_coherence: Optional[float] = None
    llm_overall: Optional[float] = None
    llm_cost_usd: float = 0.0


class ExperimentRunner:
    """
    Runs controlled retrieval experiments comparing:
    sparse | dense | hybrid | tfidf_baseline

    Reports ROUGE, Recall@k, MRR, nDCG, LLM-judge scores, latency, and cost.
    Key metric: ≥20% theme accuracy improvement over TF-IDF baseline.
    """

    def __init__(self, retriever, generator, tfidf_baseline: TFIDFBaseline,
                 openai_client=None, enable_llm_judge: bool = False):
        self.retriever = retriever
        self.generator = generator
        self.tfidf = tfidf_baseline
        self.openai_client = openai_client
        self.enable_llm_judge = enable_llm_judge

    def run_single(
        self,
        query: str,
        ground_truth_themes: List[str],
        ground_truth_summary: Optional[str] = None,
        relevant_chunk_ids: Optional[List[str]] = None,
        top_k: int = 10,
        mode: str = "hybrid"
    ) -> ExperimentResult:
        start = time.perf_counter()

        chunks, retrieval_latency = self.retriever.retrieve(query, top_k=top_k, mode=mode)
        source_texts = [c.text for c in chunks]
        retrieved_ids = [c.id for c in chunks]

        gen_result = self.generator.generate(query, chunks)
        executive_summary = gen_result.get("executive_summary", "")
        predicted_themes = [t["theme"] for t in gen_result.get("themes", [])]

        latency_ms = (time.perf_counter() - start) * 1000

        rouge = compute_rouge(
            executive_summary,
            [ground_truth_summary] if ground_truth_summary else ground_truth_themes
        )
        theme_acc = compute_theme_accuracy(predicted_themes, ground_truth_themes)
        hallucination = compute_hallucination_score(executive_summary, source_texts)

        ir_metrics = {}
        if relevant_chunk_ids:
            ir_metrics = compute_ir_metrics(retrieved_ids, relevant_chunk_ids, k_values=[1, 3, 5, 10])

        judge_result = None
        if self.enable_llm_judge and executive_summary:
            judge_result = run_llm_judge(
                query=query, summary=executive_summary,
                source_texts=source_texts, openai_client=self.openai_client
            )

        return ExperimentResult(
            mode=mode,
            rouge_1=rouge["rouge1"], rouge_2=rouge["rouge2"], rouge_l=rouge["rougeL"],
            theme_accuracy=theme_acc,
            hallucination_score=hallucination,
            latency_ms=latency_ms,
            retrieved_count=len(chunks),
            recall_at_1=ir_metrics.get("recall@1", 0.0),
            recall_at_3=ir_metrics.get("recall@3", 0.0),
            recall_at_5=ir_metrics.get("recall@5", 0.0),
            recall_at_10=ir_metrics.get("recall@10", 0.0),
            mrr=ir_metrics.get("mrr", 0.0),
            ndcg_at_5=ir_metrics.get("ndcg@5", 0.0),
            ndcg_at_10=ir_metrics.get("ndcg@10", 0.0),
            llm_faithfulness=judge_result.faithfulness if judge_result else None,
            llm_relevance=judge_result.relevance if judge_result else None,
            llm_coherence=judge_result.coherence if judge_result else None,
            llm_overall=judge_result.overall if judge_result else None,
            llm_cost_usd=judge_result.cost_usd if judge_result else 0.0,
        )

    def run_all_modes(
        self,
        query: str,
        ground_truth_themes: List[str],
        ground_truth_summary: Optional[str] = None,
        relevant_chunk_ids: Optional[List[str]] = None,
        top_k: int = 10
    ) -> Dict[str, Any]:
        modes = ["sparse", "dense", "hybrid"]
        results = {}

        for mode in modes:
            try:
                result = self.run_single(
                    query, ground_truth_themes, ground_truth_summary,
                    relevant_chunk_ids, top_k, mode
                )
                results[mode] = result
                logger.info(
                    f"[{mode}] ROUGE-1: {result.rouge_1:.3f}, "
                    f"Theme Acc: {result.theme_accuracy:.3f}, "
                    f"MRR: {result.mrr:.3f}, nDCG@5: {result.ndcg_at_5:.3f}, "
                    f"Latency: {result.latency_ms:.0f}ms"
                )
            except Exception as e:
                logger.error(f"Experiment failed for mode {mode}: {e}")

        # TF-IDF baseline
        try:
            tfidf_results = self.tfidf.search(query, top_k=top_k)
            tfidf_texts = [
                self.tfidf.corpus_texts[self.tfidf.corpus_ids.index(cid)]
                for cid, _ in tfidf_results if cid in self.tfidf.corpus_ids
            ]
            tfidf_ids = [cid for cid, _ in tfidf_results]
            tfidf_theme_acc = compute_theme_accuracy(
                [" ".join(tfidf_texts[:3])[:200]], ground_truth_themes
            )
            tfidf_ir = {}
            if relevant_chunk_ids:
                tfidf_ir = compute_ir_metrics(tfidf_ids, relevant_chunk_ids, k_values=[1, 3, 5, 10])

            results["tfidf_baseline"] = ExperimentResult(
                mode="tfidf_baseline",
                rouge_1=0.0, rouge_2=0.0, rouge_l=0.0,
                theme_accuracy=tfidf_theme_acc,
                hallucination_score=0.0, latency_ms=0.0,
                retrieved_count=len(tfidf_results),
                recall_at_1=tfidf_ir.get("recall@1", 0.0),
                recall_at_5=tfidf_ir.get("recall@5", 0.0),
                mrr=tfidf_ir.get("mrr", 0.0),
                ndcg_at_5=tfidf_ir.get("ndcg@5", 0.0),
            )
        except Exception as e:
            logger.error(f"TF-IDF baseline failed: {e}")

        best_mode = max(
            [m for m in modes if m in results],
            key=lambda m: results[m].theme_accuracy,
            default="hybrid"
        )

        baseline_acc = results.get("tfidf_baseline", ExperimentResult("", 0, 0, 0, 0, 0, 0, 0)).theme_accuracy
        hybrid_acc = results.get("hybrid", ExperimentResult("", 0, 0, 0, 0, 0, 0, 0)).theme_accuracy
        improvement = ((hybrid_acc - baseline_acc) / max(baseline_acc, 0.01)) * 100

        return {
            "query": query,
            "results": {k: vars(v) for k, v in results.items()},
            "best_mode": best_mode,
            "improvement_over_tfidf_pct": round(improvement, 1),
            "target_met": improvement >= 20.0
        }
