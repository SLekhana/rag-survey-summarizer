"""
Evaluation Module
- ROUGE-1, ROUGE-2, ROUGE-L scoring
- Held-out theme detection accuracy
- Hallucination detection score
- Controlled experiments: sparse vs dense vs hybrid
- TF-IDF baseline comparison
"""

import time
import logging
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import numpy as np
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# ROUGE EVALUATION
# ──────────────────────────────────────────────

def compute_rouge(hypothesis: str, references: List[str]) -> Dict[str, float]:
    """
    Compute ROUGE-1, ROUGE-2, ROUGE-L F1 scores.
    hypothesis: generated summary
    references: ground truth summaries (best match is used)
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


# ──────────────────────────────────────────────
# THEME DETECTION ACCURACY
# ──────────────────────────────────────────────

def compute_theme_accuracy(
    predicted_themes: List[str],
    ground_truth_themes: List[str],
    threshold: float = 0.5
) -> float:
    """
    Compute theme detection accuracy via TF-IDF cosine similarity matching.
    A predicted theme is "correct" if it has cosine similarity >= threshold
    with any ground truth theme.

    Returns: precision-like accuracy (% of ground truth themes detected)
    """
    if not predicted_themes or not ground_truth_themes:
        return 0.0

    # Vectorize all themes together
    all_themes = predicted_themes + ground_truth_themes
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    try:
        tfidf_matrix = vectorizer.fit_transform(all_themes)
    except Exception:
        return 0.0

    pred_matrix = tfidf_matrix[:len(predicted_themes)]
    gt_matrix = tfidf_matrix[len(predicted_themes):]

    # For each GT theme, check if any predicted theme matches
    sim_matrix = cosine_similarity(gt_matrix, pred_matrix)
    matched = sum(1 for row in sim_matrix if row.max() >= threshold)

    return matched / len(ground_truth_themes)


# ──────────────────────────────────────────────
# HALLUCINATION DETECTION
# ──────────────────────────────────────────────

HALLUCINATION_PATTERNS = [
    r'\b(always|never|all|every|none|everyone|nobody)\b',
    r'\b(definitely|certainly|absolutely|guaranteed)\b',
    r'\b(\d+%)\b',  # Specific percentages not in source
    r'\b(research shows|studies indicate|experts say)\b',
]


def compute_hallucination_score(
    generated_text: str,
    source_texts: List[str]
) -> float:
    """
    Estimate hallucination likelihood.
    Score 0.0 = likely grounded, 1.0 = likely hallucinated.

    Method:
    1. Check for absolute/speculative language patterns
    2. Check n-gram overlap with source texts (low overlap = potential hallucination)
    """
    pattern_flags = 0
    for pattern in HALLUCINATION_PATTERNS:
        if re.search(pattern, generated_text, re.IGNORECASE):
            pattern_flags += 1

    # N-gram overlap
    gen_words = set(generated_text.lower().split())
    source_words = set(" ".join(source_texts).lower().split())
    if gen_words:
        overlap = len(gen_words & source_words) / len(gen_words)
    else:
        overlap = 0.0

    # Combine: high pattern flags + low overlap = high hallucination risk
    pattern_score = min(pattern_flags / len(HALLUCINATION_PATTERNS), 1.0)
    grounding_score = 1 - overlap

    hallucination_score = 0.4 * pattern_score + 0.6 * grounding_score
    return round(hallucination_score, 3)


# ──────────────────────────────────────────────
# TF-IDF BASELINE
# ──────────────────────────────────────────────

class TFIDFBaseline:
    """
    TF-IDF retrieval baseline for comparison against hybrid RAG.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            max_features=50000
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


# ──────────────────────────────────────────────
# CONTROLLED EXPERIMENT RUNNER
# ──────────────────────────────────────────────

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


class ExperimentRunner:
    """
    Runs controlled retrieval experiments comparing:
    - sparse (BM25)
    - dense (FAISS)
    - hybrid (BM25 + FAISS)
    - baseline (TF-IDF)

    Key metric: theme_accuracy improvement over TF-IDF baseline.
    Target: ≥ 20% improvement (per resume claims).
    """

    def __init__(self, retriever, generator, tfidf_baseline: TFIDFBaseline):
        self.retriever = retriever
        self.generator = generator
        self.tfidf = tfidf_baseline

    def run_single(
        self,
        query: str,
        ground_truth_themes: List[str],
        ground_truth_summary: Optional[str] = None,
        top_k: int = 10,
        mode: str = "hybrid"
    ) -> ExperimentResult:
        """Run a single experiment for one mode."""
        start = time.perf_counter()

        chunks, retrieval_latency = self.retriever.retrieve(query, top_k=top_k, mode=mode)
        source_texts = [c.text for c in chunks]

        gen_result = self.generator.generate(query, chunks)
        executive_summary = gen_result.get("executive_summary", "")
        predicted_themes = [t["theme"] for t in gen_result.get("themes", [])]

        latency_ms = (time.perf_counter() - start) * 1000

        rouge = compute_rouge(executive_summary, [ground_truth_summary] if ground_truth_summary else ground_truth_themes)
        theme_acc = compute_theme_accuracy(predicted_themes, ground_truth_themes)
        hallucination = compute_hallucination_score(executive_summary, source_texts)

        return ExperimentResult(
            mode=mode,
            rouge_1=rouge["rouge1"],
            rouge_2=rouge["rouge2"],
            rouge_l=rouge["rougeL"],
            theme_accuracy=theme_acc,
            hallucination_score=hallucination,
            latency_ms=latency_ms,
            retrieved_count=len(chunks)
        )

    def run_all_modes(
        self,
        query: str,
        ground_truth_themes: List[str],
        ground_truth_summary: Optional[str] = None,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Compare all retrieval modes + TF-IDF baseline.
        Returns results dict with best_mode and improvement metrics.
        """
        modes = ["sparse", "dense", "hybrid"]
        results = {}

        for mode in modes:
            try:
                result = self.run_single(query, ground_truth_themes, ground_truth_summary, top_k, mode)
                results[mode] = result
                logger.info(f"[{mode}] ROUGE-1: {result.rouge_1:.3f}, Theme Acc: {result.theme_accuracy:.3f}, Latency: {result.latency_ms:.0f}ms")
            except Exception as e:
                logger.error(f"Experiment failed for mode {mode}: {e}")

        # TF-IDF baseline
        try:
            tfidf_results = self.tfidf.search(query, top_k=top_k)
            tfidf_texts = [self.tfidf.corpus_texts[self.tfidf.corpus_ids.index(cid)]
                          for cid, _ in tfidf_results if cid in self.tfidf.corpus_ids]
            tfidf_themes_raw = " ".join(tfidf_texts[:3])
            tfidf_theme_acc = compute_theme_accuracy([tfidf_themes_raw[:200]], ground_truth_themes)
            results["tfidf_baseline"] = ExperimentResult(
                mode="tfidf_baseline",
                rouge_1=0.0, rouge_2=0.0, rouge_l=0.0,
                theme_accuracy=tfidf_theme_acc,
                hallucination_score=0.0,
                latency_ms=0.0,
                retrieved_count=len(tfidf_results)
            )
            logger.info(f"[TF-IDF baseline] Theme Acc: {tfidf_theme_acc:.3f}")
        except Exception as e:
            logger.error(f"TF-IDF baseline failed: {e}")

        # Determine best mode
        best_mode = max(
            [m for m in modes if m in results],
            key=lambda m: results[m].theme_accuracy,
            default="hybrid"
        )

        # Compute improvement over TF-IDF
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
