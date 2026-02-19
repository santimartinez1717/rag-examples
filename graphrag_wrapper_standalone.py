"""
Wrapper standalone para GraphRAG Neo4j - SIN importar el código del tutorial
Evita conflictos de dependencias
"""
import os
from dotenv import load_dotenv

# Cargar variables
load_dotenv()

def create_neo4j_graphrag():
    """Crea el chain de GraphRAG directamente sin importar el tutorial"""
    from langchain_neo4j import Neo4jGraph
    from langchain_neo4j.chains.graph_qa.cypher import GraphCypherQAChain
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import PromptTemplate

    # Conectar a Neo4j (sin APOC)
    graph = Neo4jGraph(
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
        enhanced_schema=False  # Deshabilitar APOC
    )
    graph.refresh_schema()

    # Prompt para generar Cypher
    cypher_template = """Task:
Generate Cypher query for a Neo4j graph database.

Instructions:
Use only the provided relationship types and properties in the schema.
Do not include any explanations or apologies in your responses.
Do not respond to any questions that ask anything other than constructing a Cypher statement.

Schema:
{schema}

The question is:
{question}
"""

    cypher_prompt = PromptTemplate(
        input_variables=["schema", "question"],
        template=cypher_template
    )

    # Prompt para generar respuesta
    qa_template = """You are an assistant that takes the results from a Neo4j Cypher query and forms a human-readable response.

Query Results:
{context}

Question:
{question}

If the provided information is empty, respond by stating that you don't know the answer.
If the information is not empty, you must provide an answer using the results.

Answer:"""

    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=qa_template
    )

    # Crear el chain
    cypher_chain = GraphCypherQAChain.from_llm(
        top_k=100,
        graph=graph,
        verbose=True,
        validate_cypher=True,
        qa_prompt=qa_prompt,
        cypher_prompt=cypher_prompt,
        qa_llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
        cypher_llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
        allow_dangerous_requests=True,
    )

    return cypher_chain


# Cache del chain para no recrearlo en cada llamada
_chain_cache = None


def neo4j_graphrag_wrapper_standalone(inputs: dict) -> dict:
    """Wrapper standalone para GraphRAG"""
    global _chain_cache

    try:
        # Crear o reutilizar el chain
        if _chain_cache is None:
            _chain_cache = create_neo4j_graphrag()

        # Ejecutar
        question = inputs["question"]
        result = _chain_cache.invoke({"query": question})

        # Procesar resultado
        answer = None
        if isinstance(result, dict):
            for key in ["result", "output", "answer", "Answer"]:
                if key in result and result[key]:
                    answer = result[key]
                    break
            if not answer and len(result) == 1:
                answer = list(result.values())[0]
            elif not answer:
                answer = str(result)
        elif isinstance(result, str):
            answer = result
        else:
            answer = str(result) if result is not None else ""

        # Extraer contexto de Neo4j para evaluadores de groundedness
        context = result.get("context", result.get("intermediate_steps", "")) if isinstance(result, dict) else ""

        return {"answer": answer if answer else "", "context": str(context)}

    except Exception as e:
        import traceback
        return {"answer": f"ERROR: {str(e)}\n\n{traceback.format_exc()}"}


if __name__ == "__main__":
    # Test
    test_q = {"question": "How many employees are in the company?"}
    print(f"Test: {test_q['question']}")
    result = neo4j_graphrag_wrapper_standalone(test_q)
    print(f"Answer: {result['answer'][:200]}")
