"""
US-G1: graphrag_no_context
Wrapper que ejecuta el pipeline Neo4j normalmente pero IGNORA los resultados
y responde solo con conocimiento paramétrico del LLM.

Propósito de validación:
- hallucination_rate debe ser ALTO (≥ 0.7) → el LLM alucina sin contexto real
- faithfulness_nli debe ser BAJO (≤ 0.3) → las respuestas no están soportadas por el contexto
- correctness puede variar (el LLM conoce algunos hechos generales)

Criterio de aceptación (PLANNING.md US-G1):
  hallucination_rate ≥ 0.7 en este wrapper vs ≤ 0.3 en graphrag_neo4j (principal)
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Cache del chain Neo4j (solo para extraer schema/contexto como referencia)
_chain_cache = None


def _get_chain():
    global _chain_cache
    if _chain_cache is None:
        from langchain_neo4j import Neo4jGraph
        from langchain_neo4j.chains.graph_qa.cypher import GraphCypherQAChain
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import PromptTemplate

        graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            username=os.getenv("NEO4J_USERNAME", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", ""),
            enhanced_schema=False,
        )
        graph.refresh_schema()

        cypher_prompt = PromptTemplate(
            input_variables=["schema", "question"],
            template="Schema:\n{schema}\n\nQuestion: {question}\n\nCypher:",
        )
        qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="Context: {context}\nQuestion: {question}\nAnswer:",
        )

        _chain_cache = GraphCypherQAChain.from_llm(
            top_k=100,
            graph=graph,
            verbose=False,
            validate_cypher=False,
            return_intermediate_steps=True,
            qa_prompt=qa_prompt,
            cypher_prompt=cypher_prompt,
            qa_llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
            cypher_llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
            allow_dangerous_requests=True,
        )
    return _chain_cache


def graphrag_no_context(inputs: dict) -> dict:
    """
    Wrapper sin contexto: genera Cypher y lo ejecuta para obtener resultados reales,
    pero los DESCARTA y responde directamente con el LLM sin ningún contexto.

    Esto simula un LLM que "alucina" porque no tiene acceso a los datos reales.
    """
    from langchain_openai import ChatOpenAI

    question = inputs["question"]

    # Intentar ejecutar Neo4j de fondo para tener los resultados reales (no usados en respuesta)
    cypher_query = ""
    db_results = []
    try:
        chain = _get_chain()
        result = chain.invoke({"query": question})
        intermediate_steps = result.get("intermediate_steps", []) if isinstance(result, dict) else []
        if intermediate_steps:
            cypher_query = intermediate_steps[0].get("query", "") if len(intermediate_steps) > 0 else ""
            db_results = intermediate_steps[1].get("context", []) if len(intermediate_steps) > 1 else []
    except Exception:
        pass  # Si Neo4j falla, seguimos igual — el punto es ignorar el contexto

    # Responder SIN contexto — solo conocimiento paramétrico del LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    prompt = f"""You are a business data assistant. Answer the following question using only your general knowledge.
Do NOT say you don't have access to a database. Provide a specific answer based on what you know.

Question: {question}

Answer:"""

    llm_response = llm.invoke(prompt)
    answer = llm_response.content if hasattr(llm_response, "content") else str(llm_response)

    # El "contexto" que reportamos es vacío / fabricado — esto es lo que faithfulness_nli evaluará
    # Usamos un contexto mínimo para que el evaluador tenga algo contra lo que comparar
    context = "No database context was used. Answer generated from LLM parametric knowledge only."

    return {
        "answer": answer,
        "context": context,
        "cypher_query": cypher_query,      # generado pero ignorado
        "db_results": db_results,           # obtenidos pero ignorados
        "architecture": "graphrag_no_context",
    }


if __name__ == "__main__":
    test_q = {"question": "How many employees are in the company?"}
    result = graphrag_no_context(test_q)
    print(f"Answer: {result['answer']}")
    print(f"Context: {result['context']}")
