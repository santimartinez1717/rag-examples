"""
US-G6: graphrag_lightrag
Wrapper basado en LightRAG (HKUDS/LightRAG) — grafo de conocimiento incremental.

Arquitectura (diferente a todos los demás):
  - Extrae entidades y relaciones de documentos de texto (no usa Cypher)
  - Construye un knowledge graph propio con LLM
  - Dual-level retrieval:
    * Low-level: entidades específicas y sus conexiones directas
    * High-level: temas y patrones globales
  - Modo "hybrid" combina ambos niveles

Diferencias vs GraphCypherQAChain:
  - No usa Cypher: construye su propio grafo desde texto
  - Texto → entidades/relaciones → grafo propio → respuesta
  - Más adecuado para texto no estructurado (vs esquemas relacionales como Northwind)
  - Incremental: puede añadir documentos sin re-indexar todo

Setup (primera llamada):
  - Indexa rag_eval/datasets/northwind_text.txt (texto plano de Northwind)
  - Guarda el índice en data/lightrag_workdir/
  - Llamadas posteriores usan el índice cacheado

Metadata LangSmith:
  {"architecture": "GraphRAG-LightRAG", "library": "lightrag-hku"}
"""
import os
import asyncio
import threading
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Rutas ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent.parent
WORKDIR = ROOT / "data" / "lightrag_workdir"
NORTHWIND_TEXT = ROOT / "rag_eval" / "datasets" / "northwind_text.txt"

# ── Cache y event loop dedicado ───────────────────────────────────────────────

_rag_cache = None
_loop = None
_loop_thread = None


def _get_or_create_loop():
    """Crea un event loop persistente en un thread dedicado."""
    global _loop, _loop_thread
    if _loop is None or not _loop.is_running():
        _loop = asyncio.new_event_loop()

        def run_loop(loop):
            asyncio.set_event_loop(loop)
            loop.run_forever()

        _loop_thread = threading.Thread(target=run_loop, args=(_loop,), daemon=True)
        _loop_thread.start()

    return _loop


def _run_async(coro):
    """Ejecuta una coroutine en el event loop dedicado de LightRAG."""
    loop = _get_or_create_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=300)


async def _init_lightrag_async():
    """Inicializa LightRAG con backend local y lo indexa si es necesario."""
    from lightrag import LightRAG
    from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

    WORKDIR.mkdir(parents=True, exist_ok=True)

    rag = LightRAG(
        working_dir=str(WORKDIR),
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=openai_embed,
    )
    await rag.initialize_storages()

    # Indexar si no hay índice previo
    index_marker = WORKDIR / ".indexed"
    if not index_marker.exists() and NORTHWIND_TEXT.exists():
        print("  [LightRAG] Indexando Northwind text... (primera vez, ~30-60s)")
        text = NORTHWIND_TEXT.read_text(encoding="utf-8")
        await rag.ainsert(text)
        index_marker.touch()
        print("  [LightRAG] Indexación completa.")

    return rag


def _get_lightrag():
    """Wrapper síncrono: inicializa LightRAG en el event loop dedicado."""
    global _rag_cache
    if _rag_cache is None:
        _rag_cache = _run_async(_init_lightrag_async())
    return _rag_cache


def graphrag_lightrag(inputs: dict) -> dict:
    """
    Wrapper LightRAG GraphRAG (US-G6).
    Responde preguntas usando un knowledge graph construido desde texto Northwind.
    Modo hybrid: combina low-level (entidades) + high-level (temas).
    """
    question = inputs["question"]

    try:
        from lightrag import QueryParam

        rag = _get_lightrag()

        answer = _run_async(rag.aquery(
            question,
            param=QueryParam(mode="hybrid", enable_rerank=False),
        ))

        # LightRAG devuelve el answer directamente como string
        if not answer or answer.strip() == "":
            answer = "I don't have information about that."

        return {
            "answer": answer,
            "context": "LightRAG hybrid graph retrieval (entity + topic level)",
            "architecture": "GraphRAG-LightRAG",
            "retrieval_mode": "hybrid",
        }

    except Exception as e:
        import traceback
        return {
            "answer": f"ERROR: {str(e)}",
            "context": "",
            "architecture": "GraphRAG-LightRAG",
            "error": traceback.format_exc()[:500],
        }


if __name__ == "__main__":
    test_q = {"question": "How many employees does the company have?"}
    result = graphrag_lightrag(test_q)
    print(f"Answer: {result['answer']}")
    print(f"Architecture: {result['architecture']}")
