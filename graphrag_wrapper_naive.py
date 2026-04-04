"""
GraphRAG "naive" — implementación alternativa sin validación de Cypher.
Usa un prompt más simple, sin few-shot examples, sin validate_cypher=True.
Sirve para comparar contra la implementación principal (graphrag_wrapper_standalone.py).

Diferencias vs wrapper principal:
- validate_cypher=False  → no valida antes de ejecutar
- Prompt más simple, sin instrucciones de schema explícitas
- Sin return_intermediate_steps en algunos casos → prueba la robustez del parser
- Permite ver qué errores aparecen sin validación
"""

import os
import re
import ast
from dotenv import load_dotenv

load_dotenv()

_chain_cache_naive = None


def create_neo4j_graphrag_naive():
    """Crea el chain naive — versión simplificada sin validación."""
    from langchain_neo4j import Neo4jGraph
    from langchain_neo4j.chains.graph_qa.cypher import GraphCypherQAChain
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import PromptTemplate

    graph = Neo4jGraph(
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
        enhanced_schema=False
    )
    graph.refresh_schema()

    # Prompt Cypher más simple — sin instrucciones detalladas
    cypher_template = """Generate a Cypher query for Neo4j to answer the question.
Use the schema: {schema}
Question: {question}
Cypher Query:"""

    # Prompt QA igual
    qa_template = """Use the following database results to answer the question.
Results: {context}
Question: {question}
Answer:"""

    cypher_chain = GraphCypherQAChain.from_llm(
        top_k=100,
        graph=graph,
        verbose=True,
        validate_cypher=False,          # diferencia clave
        return_intermediate_steps=True,
        qa_prompt=PromptTemplate(input_variables=["context", "question"], template=qa_template),
        cypher_prompt=PromptTemplate(input_variables=["schema", "question"], template=cypher_template),
        qa_llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
        cypher_llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
        allow_dangerous_requests=True,
    )
    return cypher_chain, graph


def neo4j_graphrag_naive(inputs: dict) -> dict:
    """Wrapper naive para GraphRAG — sin validación de Cypher."""
    global _chain_cache_naive

    try:
        if _chain_cache_naive is None:
            _chain_cache_naive = create_neo4j_graphrag_naive()

        chain, graph = _chain_cache_naive
        question = inputs["question"]
        result = chain.invoke({"query": question})

        # Extraer answer
        answer = None
        if isinstance(result, dict):
            for key in ["result", "output", "answer"]:
                if key in result and result[key]:
                    answer = result[key]
                    break
            if not answer:
                answer = str(result)
        else:
            answer = str(result) if result is not None else ""

        # Extraer intermediate steps
        cypher_query = ""
        db_results = []
        intermediate_steps = result.get("intermediate_steps", []) if isinstance(result, dict) else []
        if intermediate_steps:
            cypher_query = intermediate_steps[0].get("query", "") if len(intermediate_steps) > 0 else ""
            db_results = intermediate_steps[1].get("context", []) if len(intermediate_steps) > 1 else []

        context = f"Cypher Query: {cypher_query}\n\nDatabase Results: {db_results}"

        # Schema labels
        schema_labels = []
        try:
            schema = graph.structured_schema
            schema_labels = list(schema.get("node_props", {}).keys())
        except Exception:
            schema_labels = []

        return {
            "answer": answer if answer else "",
            "context": context,
            "cypher_query": cypher_query,
            "db_results": db_results if isinstance(db_results, list) else [],
            "schema_labels": schema_labels,
        }

    except Exception as e:
        import traceback
        return {
            "answer": f"ERROR: {str(e)}\n\n{traceback.format_exc()}",
            "context": "", "cypher_query": "", "db_results": [], "schema_labels": []
        }


if __name__ == "__main__":
    test_q = {"question": "How many employees are in the company?"}
    print(f"Test naive: {test_q['question']}")
    result = neo4j_graphrag_naive(test_q)
    print(f"Answer: {result['answer'][:200]}")
    print(f"Cypher: {result['cypher_query']}")
