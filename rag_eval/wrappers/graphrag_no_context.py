"""
US-G1: graphrag_no_context
Wrapper que responde solo con conocimiento paramétrico del LLM, sin usar el grafo.

Propósito de validación:
- hallucination_rate debe ser ALTO (≥ 0.7)
- faithfulness_nli debe ser BAJO (≤ 0.3)
- Sirve como piso de referencia en la comparativa discriminativa

Multi-database: acepta inputs["database"] pero no lo usa (el punto es ignorar el contexto).
"""
import os
from dotenv import load_dotenv

load_dotenv()


def graphrag_no_context(inputs: dict) -> dict:
    """
    Wrapper sin contexto: responde directamente con el LLM sin ningún contexto de base de datos.
    Simula un LLM que alucina porque no tiene acceso a los datos reales.
    """
    from langchain_openai import ChatOpenAI

    question = inputs["question"]
    database = inputs.get("database", "neo4j")

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    prompt = f"""You are a data assistant. Answer the following question using only your general knowledge.
Do NOT say you don't have access to a database. Provide a specific answer based on what you know.

Question: {question}

Answer:"""

    llm_response = llm.invoke(prompt)
    answer = llm_response.content if hasattr(llm_response, "content") else str(llm_response)

    return {
        "answer": answer,
        "context": "No database context was used. Answer generated from LLM parametric knowledge only.",
        "cypher_query": "",
        "db_results": [],
        "schema_labels": [],
        "architecture": "graphrag_no_context",
        "database": database,
    }


if __name__ == "__main__":
    for q in [
        {"question": "How many employees are in the company?", "database": "neo4j"},
        {"question": "What movies did Tom Hanks act in?", "database": "db1"},
        {"question": "What is House Stark's motto?", "database": "db2"},
    ]:
        result = graphrag_no_context(q)
        print(f"[{q['database']}] Q: {q['question']}")
        print(f"  A: {result['answer'][:100]}")
