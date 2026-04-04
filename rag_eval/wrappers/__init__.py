"""
Wrappers RAG — adaptadores para distintas arquitecturas.

Todos los wrappers implementan la firma estándar:
    def rag_fn(inputs: dict) -> dict

Donde inputs contiene al menos {"question": str}
y outputs contiene al menos {"answer": str, "context": str | list}.
"""

from rag_eval.wrappers.graphrag_neo4j import neo4j_graphrag_wrapper_standalone
from rag_eval.wrappers.graphrag_naive import neo4j_graphrag_naive
from rag_eval.wrappers.graphrag_no_context import graphrag_no_context
from rag_eval.wrappers.graphrag_always_refuse import graphrag_always_refuse

__all__ = [
    "neo4j_graphrag_wrapper_standalone",   # main: validate_cypher=True, GPT-3.5
    "neo4j_graphrag_naive",                # naive: validate_cypher=False, minimal prompt
    "graphrag_no_context",                 # US-G1: ignores Neo4j, answers from LLM knowledge only
    "graphrag_always_refuse",              # US-G2: always refuses, validates negative_rejection
]
