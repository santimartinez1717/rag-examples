"""
US-G3 (revisado): graphrag_neo4j_native — Hybrid 3-Retriever con neo4j-graphrag SDK.

Arquitectura híbrida con 3 estrategias de recuperación combinadas:
  1. VectorRetriever    — búsqueda semántica sobre embeddings de nodos (EmbeddedNode)
  2. Text2CypherRetriever — NL → Cypher con schema injection (librería oficial Neo4j)
  3. Context fusion     — combina los resultados de ambos para el LLM

Por qué es diferente a los wrappers LangChain (graphrag_main/naive/langgraph):
  - Usa neo4j-graphrag SDK (oficial de Neo4j) en lugar de LangChain GraphCypherQAChain
  - VectorRetriever: recupera por similitud semántica (útil cuando el Cypher falla)
  - Text2CypherRetriever: genera Cypher con un sistema de prompts diferente (sin validate_cypher)
  - Resultado: más robusto ante preguntas ambiguas (vector como fallback de Cypher)

Multi-database:
  - Northwind (neo4j): index "northwind_embeddings"
  - Movies (db1):      index "movie_embeddings"
  - GoT (db2):         index "got_embeddings"
  - El dataset incluye inputs["database"] para seleccionar la base de datos

Metadata LangSmith:
  {"architecture": "GraphRAG-Neo4j-native-hybrid", "library": "neo4j-graphrag-python"}

Prerequisitos:
    pip install neo4j-graphrag
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── Cache por database ────────────────────────────────────────────────────────

_pipeline_cache: dict = {}

# Mapeo database → vector index name
VECTOR_INDEX_MAP = {
    "neo4j": "northwind_embeddings",
    "db1":   "movie_embeddings",
    "db2":   "got_embeddings",
}

# Schema textual por database (usado por Text2CypherRetriever)
NEO4J_SCHEMA = {
    "neo4j": """
Node labels and properties:
- Employee: employeeID, firstName, lastName, title, city, country, region, homePhone
- Product: productID, productName, unitPrice, unitsInStock, unitsOnOrder, discontinued
- Category: categoryID, categoryName, description
- Customer: customerID, companyName, contactName, city, country
- Order: orderID, orderDate, shipCountry, shipCity
- Supplier: supplierID, companyName, country, city
- Shipper: shipperID, companyName, phone

Relationships:
- (Employee)-[:REPORTS_TO]->(Employee)
- (Employee)-[:SOLD]->(Order)
- (Order)-[:ORDERS]->(Product)
- (Product)-[:PART_OF]->(Category)
- (Product)-[:SUPPLIED_BY]->(Supplier)
- (Order)-[:PURCHASED_BY]->(Customer)
- (Order)-[:SHIPPED_VIA]->(Shipper)
""",
    "db1": """
Node labels and properties:
- Movie: title, released, tagline
- Person: name, born

Relationships:
- (Person)-[:ACTED_IN {roles}]->(Movie)
- (Person)-[:DIRECTED]->(Movie)
- (Person)-[:WROTE]->(Movie)
- (Person)-[:PRODUCED]->(Movie)
""",
    "db2": """
Node labels and properties:
- Person: name, nickname, status, title
- House: name, words, seat, region
- Location: name, type, region

Relationships:
- (Person)-[:MEMBER_OF]->(House)
- (House)-[:CONTROLS]->(Location)
- (Person)-[:PARENT_OF]->(Person)
- (Person)-[:SIBLING_OF]->(Person)
- (Person)-[:MARRIED_TO]->(Person)
- (Person)-[:KILLED]->(Person)
- (Person)-[:SERVES]->(House)
""",
}


def _build_pipeline(database: str):
    """Construye el pipeline híbrido para una base de datos específica."""
    from neo4j import GraphDatabase
    from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
    from neo4j_graphrag.retrievers import VectorRetriever, Text2CypherRetriever
    from neo4j_graphrag.llm import OpenAILLM

    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(
            os.getenv("NEO4J_USERNAME", "neo4j"),
            os.getenv("NEO4J_PASSWORD", ""),
        ),
    )

    embedder = OpenAIEmbeddings(model="text-embedding-ada-002")
    llm = OpenAILLM(model_name="gpt-3.5-turbo", model_params={"temperature": 0})

    vector_index = VECTOR_INDEX_MAP.get(database, "northwind_embeddings")
    schema = NEO4J_SCHEMA.get(database, "")

    # Retriever 1: Vector (semántico)
    vector_retriever = VectorRetriever(
        driver=driver,
        index_name=vector_index,
        embedder=embedder,
        return_properties=["text", "source_label", "source_id"],
        neo4j_database=database,
    )

    # Retriever 2: Text2Cypher (simbólico)
    t2c_retriever = Text2CypherRetriever(
        driver=driver,
        llm=llm,
        neo4j_schema=schema,
        neo4j_database=database,
    )

    return {
        "driver": driver,
        "vector_retriever": vector_retriever,
        "t2c_retriever": t2c_retriever,
        "llm": llm,
        "embedder": embedder,
        "database": database,
    }


def _get_pipeline(database: str):
    global _pipeline_cache
    if database not in _pipeline_cache:
        _pipeline_cache[database] = _build_pipeline(database)
    return _pipeline_cache[database]


def graphrag_neo4j_native(inputs: dict) -> dict:
    """
    Wrapper neo4j-graphrag hybrid (US-G3).
    Arquitectura: VectorRetriever + Text2CypherRetriever → fusión de contexto → LLM.

    El dataset puede incluir inputs["database"] para seleccionar la base de datos:
        "neo4j" → Northwind
        "db1"   → Movies
        "db2"   → Game of Thrones

    Por defecto usa "neo4j" (Northwind) para compatibilidad con datasets anteriores.
    """
    question = inputs["question"]
    database = inputs.get("database", "neo4j")

    try:
        pipeline = _get_pipeline(database)
        vector_retriever = pipeline["vector_retriever"]
        t2c_retriever = pipeline["t2c_retriever"]
        llm = pipeline["llm"]

        context_parts = []
        retrieval_log = {}

        # ── Retriever 1: Vector search ────────────────────────────────────────
        try:
            vector_result = vector_retriever.search(query_text=question, top_k=5)
            vector_chunks = []
            for item in (vector_result.items if hasattr(vector_result, "items") else []):
                content = item.content if hasattr(item, "content") else str(item)
                if content:
                    vector_chunks.append(content)
            if vector_chunks:
                context_parts.append("=== Vector Search Results ===\n" + "\n".join(vector_chunks))
            retrieval_log["vector_chunks"] = len(vector_chunks)
        except Exception as e:
            retrieval_log["vector_error"] = str(e)[:200]

        # ── Retriever 2: Text2Cypher ──────────────────────────────────────────
        cypher_query = ""
        try:
            t2c_result = t2c_retriever.search(query_text=question)
            t2c_chunks = []
            for item in (t2c_result.items if hasattr(t2c_result, "items") else []):
                content = item.content if hasattr(item, "content") else str(item)
                if content:
                    t2c_chunks.append(content)
            if t2c_chunks:
                context_parts.append("=== Graph Query Results ===\n" + "\n".join(t2c_chunks))
            # Extract the generated Cypher if available
            if hasattr(t2c_result, "metadata") and t2c_result.metadata:
                cypher_query = t2c_result.metadata.get("cypher", "")
            retrieval_log["t2c_chunks"] = len(t2c_chunks)
        except Exception as e:
            retrieval_log["t2c_error"] = str(e)[:200]

        # ── Fusion: generate answer from combined context ─────────────────────
        combined_context = "\n\n".join(context_parts) if context_parts else "No relevant information found."

        prompt = f"""You are a helpful assistant. Use the following retrieved information to answer the question.
If the information is insufficient, say so clearly without hallucinating.

Retrieved Information:
{combined_context}

Question: {question}

Answer:"""

        response = llm.invoke(prompt)
        # neo4j-graphrag LLM returns LLMResponse with .content
        if hasattr(response, "content"):
            answer = response.content
        else:
            answer = str(response)

        return {
            "answer": answer,
            "context": combined_context[:1000],
            "cypher_query": cypher_query,
            "db_results": context_parts,
            "architecture": "GraphRAG-Neo4j-native-hybrid",
            "retrieval_log": retrieval_log,
            "database": database,
        }

    except Exception as e:
        import traceback
        return {
            "answer": f"ERROR: {str(e)}",
            "context": "",
            "cypher_query": "",
            "db_results": [],
            "architecture": "GraphRAG-Neo4j-native-hybrid",
            "error": traceback.format_exc()[:500],
            "database": database,
        }


if __name__ == "__main__":
    # Test with each database
    tests = [
        {"question": "How many employees does the company have?", "database": "neo4j"},
        {"question": "What movies did Tom Hanks act in?", "database": "db1"},
        {"question": "What is House Stark's motto?", "database": "db2"},
    ]
    for t in tests:
        print(f"\nQ [{t['database']}]: {t['question']}")
        result = graphrag_neo4j_native(t)
        print(f"A: {result['answer'][:300]}")
        print(f"Log: {result.get('retrieval_log', {})}")
