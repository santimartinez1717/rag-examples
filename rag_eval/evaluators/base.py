"""
RAG Evaluator — soporte universal para RAG vectorial y GraphRAG (Neo4j/Cypher).

Uso básico (RAG vectorial):
    evaluate_rag(rag_fn, dataset, dataset_name="...", project="...")

Uso para GraphRAG:
    evaluate_graphrag(graphrag_fn, dataset, dataset_name="...", project="...")

El wrapper GraphRAG debe retornar:
    {
        "answer": str,
        "context": str,           # texto libre (para backward compat)
        "cypher_query": str,      # Cypher generado
        "db_results": list,       # resultados raw de Neo4j
        "schema_labels": list,    # labels de nodos disponibles en el grafo
    }
"""

import re
import ast
from typing import Callable, List, Dict, Optional
from langsmith import Client
from langsmith.utils import LangSmithConflictError
from langchain_openai import ChatOpenAI
from typing_extensions import Annotated, TypedDict


# ─────────────────────────────────────────────
# TypedDicts para outputs estructurados
# ─────────────────────────────────────────────

class CorrectnessGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    correct: Annotated[bool, ..., "True if the answer is correct, False otherwise."]

class RelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[bool, ..., "True if the answer addresses the question"]

class GroundedGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    grounded: Annotated[bool, ..., "True if the answer is grounded in the facts"]

class CypherGrade(TypedDict):
    explanation: Annotated[str, ..., "Reasoning about Cypher semantic correctness"]
    score: Annotated[float, ..., "Score between 0.0 and 1.0"]


# ─────────────────────────────────────────────
# LLMs compartidos
# ─────────────────────────────────────────────

_llm = ChatOpenAI(model="gpt-4.1", temperature=0)
_correctness_llm = _llm.with_structured_output(CorrectnessGrade, method="json_schema", strict=True)
_relevance_llm   = _llm.with_structured_output(RelevanceGrade,   method="json_schema", strict=True)
_grounded_llm    = _llm.with_structured_output(GroundedGrade,    method="json_schema", strict=True)
_cypher_llm      = _llm.with_structured_output(CypherGrade,      method="json_schema", strict=True)


# ─────────────────────────────────────────────
# Helpers de parseo de contexto
# ─────────────────────────────────────────────

def _extract_cypher(context: str) -> str:
    """Extrae el Cypher query del campo context string."""
    if not context:
        return ""
    match = re.search(r'Cypher Query:\s*(.*?)\n\nDatabase Results:', context, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def _extract_db_results(context: str) -> list:
    """Extrae la lista de resultados de BD del campo context string."""
    if not context:
        return []
    match = re.search(r'Database Results:\s*(.*)', context, re.DOTALL)
    if not match:
        return []
    results_str = match.group(1).strip()
    try:
        results = ast.literal_eval(results_str)
        return results if isinstance(results, list) else []
    except (ValueError, SyntaxError):
        return [] if results_str == "[]" else [results_str]


def _is_no_answer_response(answer: str) -> bool:
    """Detecta si el answer explícitamente reconoce que no tiene información."""
    phrases = [
        "don't know", "do not know", "no information", "no data",
        "cannot find", "not found", "no results", "unable to find",
        "i don't have", "no records", "nothing found", "couldn't find"
    ]
    return any(p in answer.lower() for p in phrases)


def _format_db_results(db_results: list, max_records: int = 10) -> str:
    """Convierte resultados de Neo4j a texto legible para LLM-judges."""
    if not db_results:
        return "No results returned from database."
    lines = []
    skip_keys = {'photo', 'photoPath', 'notes'}
    for i, record in enumerate(db_results[:max_records]):
        if isinstance(record, dict):
            for alias, node in record.items():
                if isinstance(node, dict):
                    props = {k: v for k, v in node.items()
                             if k not in skip_keys and v is not None}
                    lines.append(f"[{i+1}] {alias}: {props}")
                else:
                    lines.append(f"[{i+1}] {alias}: {node}")
        else:
            lines.append(f"[{i+1}] {record}")
    if len(db_results) > max_records:
        lines.append(f"... and {len(db_results) - max_records} more records")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# EVALUADORES DETERMINISTAS (sin LLM)
# ─────────────────────────────────────────────

def cypher_generated(inputs: dict, outputs: dict) -> dict:
    """¿Se generó algún Cypher query válido?"""
    context = outputs.get("context", "")
    cypher = outputs.get("cypher_query", "") or _extract_cypher(context)
    has_cypher = bool(cypher) and any(
        kw in cypher.upper() for kw in ["MATCH", "RETURN", "CALL", "CREATE", "MERGE"]
    )
    return {"key": "cypher_generated", "score": int(has_cypher)}


def cypher_result_nonempty(inputs: dict, outputs: dict) -> dict:
    """¿El Cypher retornó al menos 1 resultado de la BD?"""
    db_results = outputs.get("db_results")
    if db_results is None:
        db_results = _extract_db_results(outputs.get("context", ""))
    return {"key": "cypher_result_nonempty", "score": int(len(db_results) > 0)}


def empty_context_hallucination(inputs: dict, outputs: dict) -> dict:
    """
    Detecta alucinación silenciosa: BD retornó 0 resultados pero el LLM
    respondió con información concreta (sin reconocer que no sabe).
    Score: 1 = OK (sin alucinación), 0 = alucinación detectada.
    """
    db_results = outputs.get("db_results")
    if db_results is None:
        db_results = _extract_db_results(outputs.get("context", ""))

    answer = outputs.get("answer", "")
    results_empty = len(db_results) == 0
    answer_claims_knowledge = (
        bool(answer) and
        not _is_no_answer_response(answer) and
        len(answer.strip()) > 20
    )

    is_hallucination = results_empty and answer_claims_knowledge
    return {
        "key": "empty_context_hallucination",
        "score": 0 if is_hallucination else 1,
        "comment": "Hallucination: LLM responded with empty DB results" if is_hallucination else "OK"
    }


def schema_adherence(inputs: dict, outputs: dict) -> dict:
    """
    ¿El Cypher usa únicamente labels de nodos que existen en el schema del grafo?
    Score continuo [0,1]: fracción de labels usados que son válidos.
    """
    context = outputs.get("context", "")
    cypher = outputs.get("cypher_query", "") or _extract_cypher(context)
    schema_labels = outputs.get("schema_labels", [])

    if not cypher:
        return {"key": "schema_adherence", "score": 1.0, "comment": "No Cypher to check"}
    if not schema_labels:
        return {"key": "schema_adherence", "score": 1.0, "comment": "Schema not available"}

    # Extrae labels de nodos: (alias:Label) o (:Label)
    used_labels = set(re.findall(r'\(\s*\w*\s*:\s*([A-Za-z_][A-Za-z0-9_]*)', cypher))
    if not used_labels:
        return {"key": "schema_adherence", "score": 1.0, "comment": "No labels found in Cypher"}

    known = set(schema_labels)
    correct = sum(1 for l in used_labels if l in known)
    score = correct / len(used_labels)
    invalid = used_labels - known
    comment = f"Invalid labels: {invalid}" if invalid else "All labels valid"
    return {"key": "schema_adherence", "score": round(score, 3), "comment": comment}


# ─────────────────────────────────────────────
# EVALUADORES LLM-AS-JUDGE
# ─────────────────────────────────────────────

def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Exactitud factual de la respuesta vs ground truth."""
    student = outputs.get("answer", str(outputs))
    ref = reference_outputs.get("answer", str(reference_outputs))
    prompt = f"QUESTION: {inputs['question']}\nGROUND TRUTH: {ref}\nSTUDENT ANSWER: {student}"
    grade = _correctness_llm.invoke([
        {"role": "system", "content": _CORRECTNESS_PROMPT},
        {"role": "user", "content": prompt}
    ])
    return {"key": "correctness", "score": int(grade["correct"]), "comment": grade["explanation"]}


def relevance(inputs: dict, outputs: dict) -> dict:
    """¿La respuesta es relevante a la pregunta?"""
    student = outputs.get("answer", str(outputs))
    prompt = f"QUESTION: {inputs['question']}\nSTUDENT ANSWER: {student}"
    grade = _relevance_llm.invoke([
        {"role": "system", "content": _RELEVANCE_PROMPT},
        {"role": "user", "content": prompt}
    ])
    return {"key": "relevance", "score": int(grade["relevant"]), "comment": grade["explanation"]}


def graphrag_groundedness(inputs: dict, outputs: dict) -> dict:
    """
    Groundedness mejorado para GraphRAG: verifica que la respuesta esté
    soportada EXCLUSIVAMENTE por los registros retornados de Neo4j.
    """
    answer = outputs.get("answer", str(outputs))
    db_results = outputs.get("db_results")
    if db_results is None:
        db_results = _extract_db_results(outputs.get("context", ""))

    # Si BD vacía, groundedness determinista
    if not db_results:
        if not _is_no_answer_response(answer) and len(answer.strip()) > 20:
            return {
                "key": "graphrag_groundedness",
                "score": 0,
                "comment": "Answer claims knowledge but database returned no results"
            }
        return {"key": "graphrag_groundedness", "score": 1, "comment": "Correctly acknowledged no results"}

    facts = _format_db_results(db_results)
    prompt = f"DATABASE RESULTS (source of truth):\n{facts}\n\nSTUDENT ANSWER: {answer}"
    grade = _grounded_llm.invoke([
        {"role": "system", "content": _GRAPHRAG_GROUNDED_PROMPT},
        {"role": "user", "content": prompt}
    ])
    return {"key": "graphrag_groundedness", "score": int(grade["grounded"]), "comment": grade["explanation"]}


def cypher_semantic_correctness(inputs: dict, outputs: dict) -> dict:
    """
    ¿El Cypher generado responde semánticamente a la pregunta dada el schema?
    Score continuo [0,1].
    """
    context = outputs.get("context", "")
    cypher = outputs.get("cypher_query", "") or _extract_cypher(context)
    schema_labels = outputs.get("schema_labels", [])

    if not cypher:
        return {"key": "cypher_semantic_correctness", "score": 0.0, "comment": "No Cypher generated"}

    schema_info = f"Available node labels: {schema_labels}" if schema_labels else "Schema not available"
    prompt = f"QUESTION: {inputs['question']}\n{schema_info}\nGENERATED CYPHER:\n{cypher}"
    grade = _cypher_llm.invoke([
        {"role": "system", "content": _CYPHER_SEMANTIC_PROMPT},
        {"role": "user", "content": prompt}
    ])
    return {
        "key": "cypher_semantic_correctness",
        "score": round(float(grade["score"]), 3),
        "comment": grade["explanation"]
    }


# ─────────────────────────────────────────────
# SCORE DE CONFIANZA COMPUESTO
# ─────────────────────────────────────────────

def confidence_score(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """
    Score de confianza compuesto [0,1] para GraphRAG.

    Fórmula:
        confidence = 0.35 * correctness
                   + 0.25 * graphrag_groundedness
                   + 0.20 * cypher_result_nonempty
                   + 0.10 * relevance
                   + 0.10 * schema_adherence

    Penalización: -0.30 si empty_context_hallucination detectada.
    """
    # Deterministas (rápido, sin LLM)
    nonempty  = cypher_result_nonempty(inputs, outputs)["score"]
    schema_sc = schema_adherence(inputs, outputs)["score"]
    halluc_ok = empty_context_hallucination(inputs, outputs)["score"]  # 0=hallucination, 1=ok

    # LLM judges
    c = correctness(inputs, outputs, reference_outputs)["score"]
    r = relevance(inputs, outputs)["score"]
    g = graphrag_groundedness(inputs, outputs)["score"]

    base = (
        0.35 * c +
        0.25 * g +
        0.20 * nonempty +
        0.10 * r +
        0.10 * schema_sc
    )

    penalty = 0.30 if halluc_ok == 0 else 0.0
    score = round(max(0.0, min(1.0, base - penalty)), 3)

    comment = (
        f"correctness={c} | groundedness={g} | cypher_nonempty={nonempty} "
        f"| relevance={r} | schema={schema_sc:.2f} "
        f"| hallucination_penalty={'YES (-0.30)' if penalty > 0 else 'NO'} "
        f"→ confidence={score}"
    )
    return {"key": "confidence_score", "score": score, "comment": comment}


# ─────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────

_CORRECTNESS_PROMPT = """You are grading a student answer against a ground truth.
Rules:
- Grade ONLY on factual accuracy relative to ground truth.
- Extra correct information is fine.
- Conflicting statements = incorrect.
- Be strict about facts: names, numbers, cities, job titles must match."""

_RELEVANCE_PROMPT = """You are grading if a student answer is relevant to the question asked.
The answer must directly address what was asked. Off-topic or evasive answers are not relevant."""

_GRAPHRAG_GROUNDED_PROMPT = """You are verifying if a student answer is supported by database query results.
The DATABASE RESULTS are the ONLY source of truth.
Rules:
- Any entity, name, number, or fact in the answer must appear in the database results.
- If the database says N records and the answer claims a different count, it is not grounded.
- Reasonable paraphrasing of database values is OK.
- Invented information not present in the results = hallucination = not grounded."""

_CYPHER_SEMANTIC_PROMPT = """You are an expert in Neo4j Cypher and graph databases.
Given a question and a Cypher query, score how well the Cypher answers the question.
Scoring guide:
- 1.0: Cypher correctly retrieves exactly what the question asks.
- 0.7: Cypher retrieves relevant data with minor issues (extra fields, slightly off filter).
- 0.4: Cypher partially addresses the question but misses key requirements.
- 0.1: Cypher is mostly wrong but syntactically attempts something.
- 0.0: Cypher is completely wrong or queries wrong entities entirely.
Return a float score between 0.0 and 1.0."""


# ─────────────────────────────────────────────
# FUNCIÓN PRINCIPAL — evaluate_graphrag
# ─────────────────────────────────────────────

def evaluate_graphrag(
    rag_fn: Callable[[dict], dict],
    dataset: List[Dict],
    dataset_name: str = "graphrag-eval-dataset",
    project: str = "graphrag-eval-project",
    evaluators: Optional[List] = None,
    **kwargs
):
    """
    Evaluador especializado para GraphRAG con Neo4j.
    Incluye métricas deterministas + LLM-judge + confidence score compuesto.

    El rag_fn debe retornar:
        {"answer": str, "context": str, "cypher_query": str,
         "db_results": list, "schema_labels": list}
    """
    client = Client()

    # Crear o reutilizar dataset en LangSmith
    try:
        ds = client.create_dataset(dataset_name=dataset_name)
        client.create_examples(dataset_id=ds.id, examples=dataset)
        print(f"✅ Dataset creado: '{dataset_name}' ({len(dataset)} ejemplos)")
    except (LangSmithConflictError, Exception) as e:
        if "already exists" in str(e) or "Conflict" in str(e):
            datasets = list(client.list_datasets(dataset_name=dataset_name))
            ds = datasets[0] if datasets else None
            if ds is None:
                raise RuntimeError(f"Dataset '{dataset_name}' existe pero no se pudo encontrar.")
            print(f"⚠️  Dataset '{dataset_name}' ya existe, se reutiliza.")
        else:
            raise

    if evaluators is None:
        evaluators = [
            cypher_generated,
            cypher_result_nonempty,
            empty_context_hallucination,
            schema_adherence,
            correctness,
            relevance,
            graphrag_groundedness,
            cypher_semantic_correctness,
            confidence_score,
        ]

    print(f"\n🚀 Evaluadores activos ({len(evaluators)}):")
    for ev in evaluators:
        print(f"   {'⚙️ ' if ev.__name__ in ('cypher_generated','cypher_result_nonempty','empty_context_hallucination','schema_adherence') else '🤖 '}{ev.__name__}")
    print()

    results = client.evaluate(
        rag_fn,
        data=dataset_name,
        evaluators=evaluators,
        experiment_prefix=project,
        **kwargs
    )

    print(f"\n✅ Evaluación completada!")
    print(f"🌐 Resultados en LangSmith: https://smith.langchain.com")
    return results


# ─────────────────────────────────────────────
# BACKWARD COMPAT — evaluate_rag (RAG vectorial)
# ─────────────────────────────────────────────

def groundedness(inputs: dict, outputs: dict) -> dict:
    """Groundedness genérico (RAG vectorial). Para GraphRAG usar graphrag_groundedness."""
    student_answer = outputs.get('answer', str(outputs))
    docs = outputs.get("documents", [])
    if docs:
        doc_string = "\n\n".join(
            doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in docs
        )
    else:
        doc_string = outputs.get("context", student_answer)
    prompt = f"FACTS: {doc_string}\nSTUDENT ANSWER: {student_answer}"
    grade = _grounded_llm.invoke([
        {"role": "system", "content": _GRAPHRAG_GROUNDED_PROMPT},
        {"role": "user", "content": prompt}
    ])
    return {"key": "groundedness", "score": int(grade["grounded"]), "comment": grade["explanation"]}


def retrieval_relevance(inputs: dict, outputs: dict) -> dict:
    """Relevancia del contexto recuperado vs pregunta (RAG vectorial)."""
    docs = outputs.get("documents", [])
    if docs:
        doc_string = "\n\n".join(
            doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in docs
        )
    else:
        doc_string = outputs.get("context", outputs.get("answer", ""))
    prompt = f"FACTS: {doc_string}\nQUESTION: {inputs['question']}"
    grade = _relevance_llm.invoke([
        {"role": "system", "content": _RELEVANCE_PROMPT},
        {"role": "user", "content": prompt}
    ])
    return {"key": "retrieval_relevance", "score": int(grade["relevant"]), "comment": grade["explanation"]}


def evaluate_rag(
    rag_fn: Callable[[dict], dict],
    dataset: List[Dict],
    dataset_name: str = "rag-eval-dataset",
    project: str = "rag-eval-project",
    use_correctness: bool = True,
    use_relevance: bool = True,
    use_groundedness: bool = True,
    use_retrieval_relevance: bool = True,
    **kwargs
):
    """Evaluador genérico para RAG vectorial. Para GraphRAG usar evaluate_graphrag."""
    client = Client()
    try:
        ds = client.create_dataset(dataset_name=dataset_name)
        client.create_examples(dataset_id=ds.id, examples=dataset)
        print(f"✅ Dataset creado: '{dataset_name}' con {len(dataset)} ejemplos")
    except (LangSmithConflictError, Exception) as e:
        if "already exists" in str(e) or "Conflict" in str(e):
            datasets = list(client.list_datasets(dataset_name=dataset_name))
            ds = datasets[0] if datasets else None
            if ds is None:
                raise RuntimeError(f"Dataset '{dataset_name}' exists but could not be found.")
            print(f"⚠️ Dataset '{dataset_name}' ya existe, se reutiliza.")
        else:
            raise

    evs = []
    if use_correctness:      evs.append(correctness)
    if use_relevance:        evs.append(relevance)
    if use_groundedness:     evs.append(groundedness)
    if use_retrieval_relevance: evs.append(retrieval_relevance)

    print(f"\n🚀 Ejecutando evaluación con: {[e.__name__ for e in evs]}")
    results = client.evaluate(
        rag_fn, data=dataset_name, evaluators=evs,
        experiment_prefix=project, **kwargs
    )
    print(f"\n🌐 Ver resultados en: https://smith.langchain.com")
    return results
