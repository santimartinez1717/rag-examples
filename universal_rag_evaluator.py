# Compatibility shim — importa desde el paquete rag_eval
# Los notebooks existentes pueden seguir haciendo: from universal_rag_evaluator import ...
from rag_eval.evaluators.universal import *  # noqa: F401, F403
from rag_eval.evaluators.universal import (
    evaluate_rag_universal, faithfulness_nli, hallucination_rate,
    atomic_fact_precision, context_precision_at_k, context_recall,
    context_relevance, answer_relevance_universal, correctness_universal,
    negative_rejection, confidence_score_universal,
    compute_ece, temperature_scaling, find_optimal_temperature,
    compute_calibration_report, print_universal_summary,
    DEFAULT_EVALUATORS, FULL_EVALUATORS, NLI_ONLY_EVALUATORS,
    mrr, negative_rejection_rate,
)
