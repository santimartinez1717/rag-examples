"""
US-S1: sqlrag_langchain — LangChain Text2SQL con SQLite (Northwind).

Arquitectura equivalente a graphrag_main pero sobre SQL en lugar de Cypher:
  - LangChain create_sql_query_chain genera el SQL desde lenguaje natural
  - SQLDatabase ejecuta el SQL sobre northwind.db (SQLite)
  - LLM sintetiza la respuesta final con el contexto real de la BD

Diferencias clave vs graphrag_main (GraphCypherQAChain):
  - Base de datos: SQLite (relacional) vs Neo4j (grafo)
  - Query language: SQL vs Cypher
  - Schema: tabular (JOIN) vs grafo (relaciones)
  - Mismo dataset Northwind: permite comparación directa cross-arquitectura

El framework de evaluación es idéntico:
  - Mismas preguntas Northwind (31 Q&A)
  - Mismos evaluadores (faithfulness_nli, hallucination_rate, correctness_continuous, negative_rejection)
  - Mismo confidence_score_universal

Prerequisito: data/northwind.db debe existir.
  → generado con scripts/build_northwind_sqlite.py

Metadata LangSmith:
  {"architecture": "SQLrag-LangChain", "library": "langchain-community"}
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent.parent
DB_PATH = ROOT / "data" / "northwind.db"

# Cache por db_path
_cache: dict = {}


def _build_chain(db_path: str):
    """Construye el chain Text2SQL con LangChain."""
    from langchain_community.utilities import SQLDatabase
    from langchain_classic.chains import create_sql_query_chain
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough

    db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # Chain: NL → SQL
    sql_chain = create_sql_query_chain(llm, db)

    # Chain: SQL results → final answer
    answer_prompt = PromptTemplate.from_template("""Given the following SQL query and its results, answer the question.
If the results are empty, say you don't have information about that.

SQL Query: {query}
SQL Results: {result}
Question: {question}

Answer:""")

    answer_chain = answer_prompt | llm | StrOutputParser()

    return db, sql_chain, answer_chain


def _get_chain(db_path: str):
    if db_path not in _cache:
        _cache[db_path] = _build_chain(db_path)
    return _cache[db_path]


def sqlrag_langchain(inputs: dict) -> dict:
    """
    Wrapper SQL RAG con LangChain (US-S1).
    Text2SQL: NL → SQL → DB results → respuesta.

    Compatible con el mismo dataset Northwind que graphrag_main.
    Ignora inputs["database"] — solo opera sobre northwind.db SQLite.
    """
    question = inputs["question"]
    db_path = str(DB_PATH)

    if not Path(db_path).exists():
        return {
            "answer": f"ERROR: northwind.db not found at {db_path}. Run scripts/build_northwind_sqlite.py first.",
            "context": "",
            "sql_query": "",
            "db_results": [],
            "schema_tables": [],
            "architecture": "SQLrag-LangChain",
        }

    try:
        db, sql_chain, answer_chain = _get_chain(db_path)

        # Step 1: Generate SQL
        sql_query = sql_chain.invoke({"question": question})
        # LangChain sometimes wraps in markdown — strip it
        sql_query = sql_query.strip()
        if "```" in sql_query:
            lines = sql_query.split("\n")
            sql_query = "\n".join(l for l in lines if not l.strip().startswith("```")).strip()

        # Step 2: Execute SQL
        try:
            db_result = db.run(sql_query)
        except Exception as e:
            db_result = f"SQL Error: {str(e)}"

        # Step 3: Synthesize answer
        answer = answer_chain.invoke({
            "query": sql_query,
            "result": db_result,
            "question": question,
        })

        schema_tables = db.get_usable_table_names()
        context = f"SQL Query: {sql_query}\n\nDatabase Results: {db_result}"

        return {
            "answer": answer,
            "context": context,
            "sql_query": sql_query,
            "db_results": [db_result] if isinstance(db_result, str) else db_result,
            "schema_tables": list(schema_tables),
            "architecture": "SQLrag-LangChain",
        }

    except Exception as e:
        import traceback
        return {
            "answer": f"ERROR: {str(e)}",
            "context": "",
            "sql_query": "",
            "db_results": [],
            "schema_tables": [],
            "architecture": "SQLrag-LangChain",
            "error": traceback.format_exc()[:500],
        }


if __name__ == "__main__":
    tests = [
        {"question": "How many employees does the company have?"},
        {"question": "What is the most expensive product?"},
        {"question": "Which employees work in London?"},
        {"question": "Which employee has processed the most orders?"},
        {"question": "Who is the CEO of the company?"},
    ]
    for t in tests:
        print(f"\nQ: {t['question']}")
        r = sqlrag_langchain(t)
        print(f"SQL: {r['sql_query']}")
        print(f"A:   {r['answer'][:200]}")
