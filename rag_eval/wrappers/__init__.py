"""
Wrappers RAG — adaptadores para distintas arquitecturas.

Todos los wrappers implementan la firma estándar:
    def rag_fn(inputs: dict) -> dict

Donde inputs contiene al menos {"question": str}
y outputs contiene al menos {"answer": str, "context": str | list}.
"""

from rag_eval.wrappers.graphrag_neo4j import neo4j_graphrag_wrapper_standalone
from rag_eval.wrappers.graphrag_naive import neo4j_graphrag_naive

__all__ = [
    "neo4j_graphrag_wrapper_standalone",
    "neo4j_graphrag_naive",
]
