"""
Datasets de evaluación para el framework RAG.

Formato estándar LangSmith:
    [{"inputs": {"question": str}, "outputs": {"answer": str}}, ...]
"""

from rag_eval.datasets.northwind import DATASET_NORTHWIND
from rag_eval.datasets.recommendations import (
    DATASET_RECOMMENDATIONS,
    load_recommendations_dataset,
)
from rag_eval.datasets.metaqa import (
    DATASET_METAQA,
    load_metaqa_dataset,
)

__all__ = [
    "DATASET_NORTHWIND",
    "DATASET_RECOMMENDATIONS",
    "load_recommendations_dataset",
    "DATASET_METAQA",
    "load_metaqa_dataset",
]
