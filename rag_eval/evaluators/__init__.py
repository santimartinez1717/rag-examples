"""
Evaluadores RAG — tres niveles de especialización:

  base       → evaluate_rag() / evaluate_graphrag()        (rag_evaluator.py original)
  graphrag   → evaluate_graphrag_advanced()                 (14 evaluadores Neo4j-específicos)
  universal  → evaluate_rag_universal()                     (funciona para cualquier RAG)
"""

from rag_eval.evaluators.universal import (
    evaluate_rag_universal,
    faithfulness_nli,
    hallucination_rate,
    atomic_fact_precision,
    context_precision_at_k,
    context_recall,
    context_relevance,
    answer_relevance_universal,
    correctness_universal,
    correctness_continuous,
    negative_rejection,
    confidence_score_universal,
    train_confidence_weights,
    predict_confidence_score,
    confidence_score_learned_factory,
    compute_ece,
    temperature_scaling,
    find_optimal_temperature,
    compute_calibration_report,
    DEFAULT_EVALUATORS,
    FULL_EVALUATORS,
    NLI_ONLY_EVALUATORS,
    DISCRIMINATIVE_EVALUATORS,
)

__all__ = [
    "evaluate_rag_universal",
    "faithfulness_nli",
    "hallucination_rate",
    "atomic_fact_precision",
    "context_precision_at_k",
    "context_recall",
    "context_relevance",
    "answer_relevance_universal",
    "correctness_universal",
    "correctness_continuous",
    "negative_rejection",
    "confidence_score_universal",
    "train_confidence_weights",
    "predict_confidence_score",
    "confidence_score_learned_factory",
    "compute_ece",
    "temperature_scaling",
    "find_optimal_temperature",
    "compute_calibration_report",
    "DEFAULT_EVALUATORS",
    "FULL_EVALUATORS",
    "NLI_ONLY_EVALUATORS",
    "DISCRIMINATIVE_EVALUATORS",
]
