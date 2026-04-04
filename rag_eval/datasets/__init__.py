"""
Datasets de evaluación para el framework RAG.

Formato estándar LangSmith:
    [{"inputs": {"question": str}, "outputs": {"answer": str}}, ...]
"""

from rag_eval.datasets.northwind import DATASET_NORTHWIND

__all__ = ["DATASET_NORTHWIND"]
