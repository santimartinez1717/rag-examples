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
from rag_eval.wrappers.graphrag_langgraph import graphrag_langgraph

__all__ = [
    "neo4j_graphrag_wrapper_standalone",   # main: validate_cypher=True (GraphCypherQAChain)
    "neo4j_graphrag_naive",                # naive: validate_cypher=False, minimal prompt
    "graphrag_no_context",                 # US-G1: ignores Neo4j, validates hallucination metrics
    "graphrag_always_refuse",              # US-G2: always refuses, validates negative_rejection
    "graphrag_langgraph",                  # US-G4: LangGraph agent with explicit nodes + retry
    # graphrag_neo4j_native               # US-G3: neo4j-graphrag-python (requires neo4j-graphrag)
    # graphrag_llamaindex                 # US-G5: LlamaIndex + Neo4j (requires llama-index)
    # graphrag_lightrag                   # US-G6: LightRAG text-graph (requires lightrag-hku)
]

# Optional wrappers — import explicitly when needed:
#   from rag_eval.wrappers.graphrag_neo4j_native import graphrag_neo4j_native
#   from rag_eval.wrappers.graphrag_llamaindex import graphrag_llamaindex
#   from rag_eval.wrappers.graphrag_lightrag import graphrag_lightrag
