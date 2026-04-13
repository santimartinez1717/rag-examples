"""
US-S3: sqlrag_no_context — Baseline SQL sin acceso a base de datos.

Propósito de validación:
- correctness ≈ 0.0 → respuestas basadas en conocimiento paramétrico del LLM
- hallucination_rate ALTO → afirma cosas sin datos reales
- faithfulness_nli BAJO → no hay contexto real que soporte las respuestas

Análogo a graphrag_no_context pero para comparativa SQL vs GraphRAG.
Mismo dataset Northwind — permite comparación directa.
"""
import os
from dotenv import load_dotenv

load_dotenv()


def sqlrag_no_context(inputs: dict) -> dict:
    """
    Wrapper SQL sin contexto: responde con conocimiento paramétrico del LLM.
    No genera SQL, no consulta la BD. Simula hallucination.
    """
    from langchain_openai import ChatOpenAI

    question = inputs["question"]

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    prompt = f"""You are a data assistant for a company database. Answer the following question \
using only your general knowledge. Do NOT say you don't have access to a database. \
Provide a specific answer based on what you know about the Northwind trading company.

Question: {question}

Answer:"""

    llm_response = llm.invoke(prompt)
    answer = llm_response.content if hasattr(llm_response, "content") else str(llm_response)

    return {
        "answer": answer,
        "context": "No database context was used. Answer generated from LLM parametric knowledge only.",
        "sql_query": "",
        "db_results": [],
        "schema_tables": [],
        "architecture": "SQLrag-NoContext",
    }


if __name__ == "__main__":
    tests = [
        {"question": "How many employees does the company have?"},
        {"question": "What is the most expensive product?"},
        {"question": "Who is the CEO of the company?"},
    ]
    for t in tests:
        print(f"\nQ: {t['question']}")
        r = sqlrag_no_context(t)
        print(f"A:   {r['answer'][:200]}")
