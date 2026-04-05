"""
US-G3: graphrag_neo4j_native
Wrapper usando neo4j-graphrag-python (librería oficial de Neo4j),
distinto del enfoque LangChain GraphCypherQAChain.

Diferencias clave vs graphrag_neo4j.py (LangChain):
  - Usa neo4j-graphrag SDK (neo4j_graphrag.generation.GraphRAG)
  - Pipeline: embed query → vector search → contexto → LLM
  - Es un RAG sobre texto vectorizado, NO genera Cypher
  - Requiere un vector index en Neo4j ("chunk_vector" o similar)

Metadata LangSmith:
  {"architecture": "GraphRAG-Neo4j-native", "library": "neo4j-graphrag-python"}

Prerequisitos:
    pip install neo4j-graphrag

    # En Neo4j, crear vector index:
    CREATE VECTOR INDEX chunk_vector IF NOT EXISTS
    FOR (n:Chunk) ON (n.embedding)
    OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}}

Nota: Si no hay datos vectorizados en Neo4j, este wrapper retorna un error graceful
y puede usarse como "arquitectura alternativa" incluso si no está completamente
configurada — el framework evalúa lo que tiene.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Cache del pipeline
_rag_cache = None


def _build_pipeline():
    """Construye el pipeline neo4j-graphrag-python."""
    try:
        from neo4j import GraphDatabase
        from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
        from neo4j_graphrag.retrievers import VectorRetriever
        from neo4j_graphrag.generation import GraphRAG
        from langchain_openai import ChatOpenAI
        from neo4j_graphrag.llm import LangChainLLMInterface
    except ImportError as e:
        raise ImportError(
            f"neo4j-graphrag no está instalado: {e}\n"
            "Instalar con: pip install neo4j-graphrag"
        ) from e

    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(
            os.getenv("NEO4J_USERNAME", "neo4j"),
            os.getenv("NEO4J_PASSWORD", ""),
        ),
    )

    embedder = OpenAIEmbeddings(model="text-embedding-ada-002")

    retriever = VectorRetriever(
        driver=driver,
        index_name=os.getenv("NEO4J_VECTOR_INDEX", "chunk_vector"),
        embedder=embedder,
        return_properties=["text", "source"],
    )

    # Adaptar ChatOpenAI de LangChain a la interfaz neo4j-graphrag
    lc_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    llm = LangChainLLMInterface(lc_llm)

    pipeline = GraphRAG(retriever=retriever, llm=llm)
    return pipeline


def graphrag_neo4j_native(inputs: dict) -> dict:
    """
    Wrapper neo4j-graphrag-python (US-G3).
    Arquitectura: vector search + LLM (sin Cypher).
    """
    global _rag_cache

    question = inputs["question"]

    try:
        if _rag_cache is None:
            _rag_cache = _build_pipeline()

        result = _rag_cache.search(query_text=question, retriever_config={"top_k": 5})

        # neo4j-graphrag devuelve un objeto RagResultModel
        answer = result.answer if hasattr(result, "answer") else str(result)
        items = result.retriever_result.items if hasattr(result, "retriever_result") else []
        context_chunks = [item.content for item in items if hasattr(item, "content")]
        context = "\n\n".join(context_chunks) if context_chunks else ""

        return {
            "answer": answer,
            "context": context,
            "cypher_query": "",          # no usa Cypher
            "db_results": context_chunks,
            "architecture": "GraphRAG-Neo4j-native",
        }

    except ImportError as e:
        return {
            "answer": f"ERROR: neo4j-graphrag no está instalado. {e}",
            "context": "",
            "cypher_query": "",
            "db_results": [],
            "architecture": "GraphRAG-Neo4j-native",
        }
    except Exception as e:
        import traceback
        return {
            "answer": f"ERROR: {str(e)}\n{traceback.format_exc()[:500]}",
            "context": "",
            "cypher_query": "",
            "db_results": [],
            "architecture": "GraphRAG-Neo4j-native",
        }


if __name__ == "__main__":
    test_q = {"question": "How many employees are in the company?"}
    result = graphrag_neo4j_native(test_q)
    print(f"Answer: {result['answer'][:200]}")
    print(f"Context chunks: {len(result.get('db_results', []))}")
