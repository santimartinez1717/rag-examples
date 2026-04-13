"""
US-G4: graphrag_langgraph
Wrapper basado en LangGraph (agente con nodos explícitos):
  Nodo 1: generate_cypher — genera Cypher a partir de la pregunta
  Nodo 2: execute_cypher  — ejecuta en Neo4j y maneja errores
  Nodo 3: synthesize      — genera respuesta final con el contexto real

Diferencias vs GraphCypherQAChain (LangChain):
  - Control de flujo explícito → posibilidad de reintentos, branching
  - Estado persistente entre nodos (TypedDict)
  - Más interpretable: cada nodo tiene su propia traza en LangSmith
  - Puede añadir nodos de validación/reflexión sin cambiar la interfaz

Metadata LangSmith:
  {"architecture": "GraphRAG-LangGraph", "library": "langgraph"}
"""
import os
from typing import TypedDict, Optional, List
from dotenv import load_dotenv

load_dotenv()

# ── Estado del grafo ──────────────────────────────────────────────────────────

class GraphRAGState(TypedDict):
    question: str
    schema: str
    cypher_query: str
    db_results: List
    answer: str
    error: Optional[str]
    attempts: int


# ── Cache del grafo compilado ─────────────────────────────────────────────────

_graph_cache: dict = {}
_neo4j_graph_cache: dict = {}


def _get_neo4j_graph(database: str = "neo4j"):
    global _neo4j_graph_cache
    if database not in _neo4j_graph_cache:
        from langchain_neo4j import Neo4jGraph
        g = Neo4jGraph(
            url=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            username=os.getenv("NEO4J_USERNAME", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", ""),
            database=database,
            enhanced_schema=False,
        )
        g.refresh_schema()
        _neo4j_graph_cache[database] = g
    return _neo4j_graph_cache[database]


def _build_langgraph(database: str = "neo4j"):
    """Construye el grafo LangGraph para GraphRAG."""
    from langgraph.graph import StateGraph, END
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import PromptTemplate

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    neo4j_graph = _get_neo4j_graph(database)

    # ── Nodo 1: Generar Cypher ─────────────────────────────────────────────
    cypher_prompt = PromptTemplate(
        input_variables=["schema", "question"],
        template="""Generate a Cypher query for a Neo4j database.
Use ONLY the relationships and properties defined in the schema.
Return ONLY the Cypher query — no explanations, no markdown.

Schema:
{schema}

Question: {question}

Cypher:""",
    )

    def generate_cypher(state: GraphRAGState) -> GraphRAGState:
        schema = neo4j_graph.schema
        prompt = cypher_prompt.format(schema=schema, question=state["question"])
        response = llm.invoke(prompt)
        cypher = response.content.strip()
        # Limpiar posible markdown
        if cypher.startswith("```"):
            lines = cypher.split("\n")
            cypher = "\n".join(
                l for l in lines if not l.strip().startswith("```")
            ).strip()
        return {**state, "cypher_query": cypher, "schema": schema}

    # ── Nodo 2: Ejecutar Cypher ────────────────────────────────────────────
    def execute_cypher(state: GraphRAGState) -> GraphRAGState:
        cypher = state["cypher_query"]
        if not cypher:
            return {**state, "db_results": [], "error": "No Cypher query generated"}
        try:
            results = neo4j_graph.query(cypher)
            return {**state, "db_results": results or [], "error": None}
        except Exception as e:
            # Si falla, intentar con una versión simplificada
            attempts = state.get("attempts", 0) + 1
            return {**state, "db_results": [], "error": str(e), "attempts": attempts}

    # ── Nodo 3: Sintetizar respuesta ───────────────────────────────────────
    synthesis_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a helpful assistant. Answer the question using the database results below.
If the results are empty or there was an error, say "I don't have information about that."

Database results:
{context}

Question: {question}

Answer:""",
    )

    def synthesize(state: GraphRAGState) -> GraphRAGState:
        context = str(state["db_results"]) if state["db_results"] else "No results found."
        if state.get("error"):
            context = f"Query error: {state['error']}"
        prompt = synthesis_prompt.format(context=context, question=state["question"])
        response = llm.invoke(prompt)
        return {**state, "answer": response.content.strip()}

    # ── Nodo de reintento (si Cypher falla) ───────────────────────────────
    def retry_cypher(state: GraphRAGState) -> GraphRAGState:
        """Intenta generar un Cypher más simple si el primero falló."""
        simple_prompt = f"""The previous Cypher query failed with error: {state.get('error')}
Generate a simpler Cypher query for: {state['question']}
Schema: {state.get('schema', '')}
Use MATCH/RETURN only, no complex patterns.
Cypher:"""
        response = llm.invoke(simple_prompt)
        cypher = response.content.strip()
        if cypher.startswith("```"):
            lines = cypher.split("\n")
            cypher = "\n".join(l for l in lines if not l.strip().startswith("```")).strip()
        return {**state, "cypher_query": cypher}

    # ── Función de routing ─────────────────────────────────────────────────
    def should_retry(state: GraphRAGState) -> str:
        if state.get("error") and state.get("attempts", 0) < 2:
            return "retry"
        return "synthesize"

    # ── Construir grafo ────────────────────────────────────────────────────
    builder = StateGraph(GraphRAGState)

    builder.add_node("generate_cypher", generate_cypher)
    builder.add_node("execute_cypher", execute_cypher)
    builder.add_node("retry_cypher", retry_cypher)
    builder.add_node("synthesize", synthesize)

    builder.set_entry_point("generate_cypher")
    builder.add_edge("generate_cypher", "execute_cypher")
    builder.add_conditional_edges(
        "execute_cypher",
        should_retry,
        {"retry": "retry_cypher", "synthesize": "synthesize"},
    )
    builder.add_edge("retry_cypher", "execute_cypher")
    builder.add_edge("synthesize", END)

    return builder.compile()


def graphrag_langgraph(inputs: dict) -> dict:
    """
    Wrapper LangGraph GraphRAG (US-G4).
    Agente con nodos: generate_cypher → execute_cypher → synthesize.
    Incluye retry automático si el Cypher falla.

    inputs["database"] selecciona la base de datos:
      "neo4j" (Northwind, default), "db1" (Movies), "db2" (GoT)
    """
    global _graph_cache

    question = inputs["question"]
    database = inputs.get("database", "neo4j")

    try:
        if database not in _graph_cache:
            _graph_cache[database] = _build_langgraph(database)

        initial_state: GraphRAGState = {
            "question": question,
            "schema": "",
            "cypher_query": "",
            "db_results": [],
            "answer": "",
            "error": None,
            "attempts": 0,
        }

        final_state = _graph_cache[database].invoke(initial_state)

        answer = final_state.get("answer", "")
        cypher = final_state.get("cypher_query", "")
        db_results = final_state.get("db_results", [])
        context = f"Cypher: {cypher}\n\nResults: {db_results}"

        return {
            "answer": answer,
            "context": context,
            "cypher_query": cypher,
            "db_results": db_results,
            "architecture": "GraphRAG-LangGraph",
            "attempts": final_state.get("attempts", 0),
        }

    except Exception as e:
        import traceback
        return {
            "answer": f"ERROR: {str(e)}",
            "context": "",
            "cypher_query": "",
            "db_results": [],
            "architecture": "GraphRAG-LangGraph",
            "error": traceback.format_exc()[:500],
        }


if __name__ == "__main__":
    test_q = {"question": "How many employees are in the company?"}
    result = graphrag_langgraph(test_q)
    print(f"Answer: {result['answer']}")
    print(f"Cypher: {result['cypher_query']}")
    print(f"Attempts: {result.get('attempts', 0)}")
