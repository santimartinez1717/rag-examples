"""
US-G5: graphrag_llamaindex
Wrapper basado en LlamaIndex con Neo4j PropertyGraphStore.

Arquitectura:
  - Conecta al grafo Northwind existente en Neo4j (sin reindexar)
  - Usa TextToCypherRetriever para convertir NL → Cypher
  - Alternativa a LangChain GraphCypherQAChain: misma tarea, otra librería

Diferencias vs GraphCypherQAChain (LangChain):
  - LlamaIndex maneja el contexto con PropertyGraphIndex
  - TextToCypherRetriever incluye retry logic interno
  - LLMSynonymRetriever puede expandir entidades en la query
  - API más modular: los retrievers se pueden combinar

Metadata LangSmith:
  {"architecture": "GraphRAG-LlamaIndex", "library": "llama_index"}
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── Cache del índice ───────────────────────────────────────────────────────────

_index_cache = None
_graph_store_cache = None


def _get_graph_store():
    global _graph_store_cache
    if _graph_store_cache is None:
        from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
        _graph_store_cache = Neo4jPropertyGraphStore(
            username=os.getenv("NEO4J_USERNAME", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", ""),
            url=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        )
    return _graph_store_cache


def _get_index():
    global _index_cache
    if _index_cache is None:
        from llama_index.core import PropertyGraphIndex, Settings
        from llama_index.llms.openai import OpenAI
        from llama_index.embeddings.openai import OpenAIEmbedding

        Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

        graph_store = _get_graph_store()

        # Cargar índice sobre el grafo Northwind existente (sin reindexar)
        _index_cache = PropertyGraphIndex.from_existing(
            property_graph_store=graph_store,
        )
    return _index_cache


def graphrag_llamaindex(inputs: dict) -> dict:
    """
    Wrapper LlamaIndex GraphRAG (US-G5).
    Conecta al grafo Northwind en Neo4j y responde usando TextToCypher.
    """
    question = inputs["question"]

    try:
        from llama_index.core.indices.property_graph import TextToCypherRetriever
        from llama_index.llms.openai import OpenAI

        index = _get_index()

        # TextToCypherRetriever: genera Cypher desde la pregunta
        llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
        retriever = index.as_retriever(
            sub_retrievers=[
                TextToCypherRetriever(
                    index.property_graph_store,
                    llm=llm,
                )
            ]
        )

        nodes = retriever.retrieve(question)
        context = "\n".join(node.get_content() for node in nodes) if nodes else ""

        # Sintetizar respuesta final
        if context:
            synthesis_prompt = f"""You are a helpful assistant. Answer the question using the database results below.
If the results are empty or irrelevant, say "I don't have information about that."

Database results:
{context}

Question: {question}

Answer:"""
            response = llm.complete(synthesis_prompt)
            answer = response.text.strip()
        else:
            answer = "I don't have information about that."

        return {
            "answer": answer,
            "context": context,
            "architecture": "GraphRAG-LlamaIndex",
            "retriever": "TextToCypher",
        }

    except Exception as e:
        import traceback
        return {
            "answer": f"ERROR: {str(e)}",
            "context": "",
            "architecture": "GraphRAG-LlamaIndex",
            "error": traceback.format_exc()[:500],
        }


if __name__ == "__main__":
    test_q = {"question": "How many employees are in the company?"}
    result = graphrag_llamaindex(test_q)
    print(f"Answer: {result['answer']}")
    print(f"Context: {result['context'][:200]}")
