"""
US-S2: sqlrag_agent — LangGraph SQL Agent con retry y self-correction.

Arquitectura diferente a sqlrag_langchain (US-S1):
  - LangGraph create_react_agent en lugar de create_sql_query_chain
  - SQLDatabaseToolkit provee herramientas: sql_db_query, sql_db_schema, sql_db_list_tables
  - El agente puede hacer múltiples intentos y corregir errores SQL
  - Retry implícito: el agente intenta de nuevo si falla la query

Ventajas vs sqlrag_langchain:
  - Auto-corrección de SQL erróneo
  - Puede explorar el schema antes de generar la query
  - Maneja mejor queries complejas con múltiples JOINs

Prerequisito: data/northwind.db debe existir.
  → generado con scripts/build_northwind_sqlite.py

Metadata LangSmith:
  {"architecture": "SQLrag-Agent", "library": "langgraph"}
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent.parent
DB_PATH = ROOT / "data" / "northwind.db"

# Cache por db_path
_cache: dict = {}


def _build_agent(db_path: str):
    """Construye el SQL Agent con LangGraph."""
    from langchain_community.utilities import SQLDatabase
    from langchain_community.agent_toolkits import SQLDatabaseToolkit
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent

    db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()

    system_message = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct SQLite query to run, \
then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, \
always limit your query to at most 10 results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the given tools. Only use information returned by the tools to formulate your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, \
rewrite the query and try again.
If the query returns empty results, say you don't have information about that."""

    agent = create_react_agent(llm, tools, prompt=system_message)
    return db, agent


def _get_agent(db_path: str):
    if db_path not in _cache:
        _cache[db_path] = _build_agent(db_path)
    return _cache[db_path]


def sqlrag_agent(inputs: dict) -> dict:
    """
    Wrapper SQL Agent con LangGraph (US-S2).
    Agente ReAct con herramientas SQL: puede explorar schema y auto-corregir.

    Compatible con el mismo dataset Northwind que sqlrag_langchain.
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
            "architecture": "SQLrag-Agent",
        }

    try:
        db, agent = _get_agent(db_path)

        # Run the agent
        result = agent.invoke({"messages": [{"role": "user", "content": question}]})

        # Extract the final answer from the last AI message
        messages = result.get("messages", [])
        answer = ""
        sql_query = ""
        tool_outputs = []

        for msg in messages:
            msg_type = type(msg).__name__
            if msg_type == "AIMessage":
                content = msg.content if hasattr(msg, "content") else str(msg)
                if content:
                    answer = content
            elif msg_type == "ToolMessage":
                content = msg.content if hasattr(msg, "content") else str(msg)
                # Try to extract the SQL query from tool calls
                if hasattr(msg, "name") and "query" in msg.name:
                    sql_query = content
                tool_outputs.append(content)

        # Also check for tool_calls in AI messages to get the actual SQL
        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc.get("name") == "sql_db_query":
                        sql_query = tc.get("args", {}).get("query", sql_query)

        schema_tables = db.get_usable_table_names()
        context = f"SQL Query: {sql_query}\n\nTool outputs: {'; '.join(str(t)[:200] for t in tool_outputs)}"

        return {
            "answer": answer,
            "context": context,
            "sql_query": sql_query,
            "db_results": tool_outputs,
            "schema_tables": list(schema_tables),
            "architecture": "SQLrag-Agent",
        }

    except Exception as e:
        import traceback
        return {
            "answer": f"ERROR: {str(e)}",
            "context": "",
            "sql_query": "",
            "db_results": [],
            "schema_tables": [],
            "architecture": "SQLrag-Agent",
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
        r = sqlrag_agent(t)
        print(f"SQL: {r['sql_query'][:100]}")
        print(f"A:   {r['answer'][:200]}")
