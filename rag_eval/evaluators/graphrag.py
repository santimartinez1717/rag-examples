"""
Evaluadores avanzados específicos para GraphRAG.
Complementa rag_evaluator.py con métricas de mayor complejidad:

1. cypher_complexity_score     — mide traversal depth, aggregaciones, filtros
2. multihop_required_detector  — detecta si la pregunta requería multi-hop
3. multihop_execution_score    — ¿el Cypher ejecutó el multi-hop necesario?
4. failure_mode_classifier     — clasifica POR QUÉ falló (schema/syntax/semantic/empty)
5. relationship_direction_score— ¿usó las relaciones en la dirección correcta?
6. answer_completeness         — ¿respondió todos los aspectos de la pregunta?
7. calibration_report          — ECE y curva de calibración sobre un conjunto de resultados
"""

import re
import ast
import json
from typing import Callable, List, Dict, Optional, Any
from langchain_openai import ChatOpenAI
from typing_extensions import Annotated, TypedDict


# ─────────────────────────────────────────────
# LLMs
# ─────────────────────────────────────────────

_llm = ChatOpenAI(model="gpt-4.1", temperature=0)


# ─────────────────────────────────────────────
# Helpers compartidos
# ─────────────────────────────────────────────

def _get_cypher(outputs: dict) -> str:
    cypher = outputs.get("cypher_query", "")
    if not cypher:
        ctx = outputs.get("context", "")
        m = re.search(r'Cypher Query:\s*(.*?)\n\nDatabase Results:', ctx, re.DOTALL)
        cypher = m.group(1).strip() if m else ""
    return cypher

def _get_db_results(outputs: dict) -> list:
    db = outputs.get("db_results")
    if db is not None:
        return db if isinstance(db, list) else []
    ctx = outputs.get("context", "")
    m = re.search(r'Database Results:\s*(.*)', ctx, re.DOTALL)
    if not m:
        return []
    try:
        result = ast.literal_eval(m.group(1).strip())
        return result if isinstance(result, list) else []
    except Exception:
        return []


# ─────────────────────────────────────────────
# 1. CYPHER COMPLEXITY SCORE
# ─────────────────────────────────────────────

def cypher_complexity_score(inputs: dict, outputs: dict) -> dict:
    """
    Score de complejidad estructural del Cypher generado [0, 1].
    No mide si es correcto, sino qué tan rico es estructuralmente.

    Componentes:
    - MATCH clauses (traversal hops)
    - WHERE filters
    - Aggregations (COUNT, SUM, AVG, COLLECT, MAX, MIN)
    - Variable-length paths (*N..M)
    - OPTIONAL MATCH
    - WITH pipeline stages
    - ORDER BY / LIMIT
    """
    cypher = _get_cypher(outputs).upper()
    if not cypher:
        return {"key": "cypher_complexity_score", "score": 0.0, "comment": "No Cypher"}

    scores = {
        "match_clauses": min(len(re.findall(r'\bMATCH\b', cypher)) / 3, 1.0) * 0.25,
        "where_filters": min(len(re.findall(r'\bWHERE\b', cypher)) / 2, 1.0) * 0.15,
        "aggregations": min(len(re.findall(r'\b(COUNT|SUM|AVG|COLLECT|MAX|MIN)\b', cypher)) / 2, 1.0) * 0.25,
        "variable_length": 0.20 if re.search(r'\*\d*\.\.\d*|\*\d+', cypher) else 0.0,
        "optional_match": 0.10 if 'OPTIONAL MATCH' in cypher else 0.0,
        "pipeline": min(len(re.findall(r'\bWITH\b', cypher)) / 2, 1.0) * 0.05,
    }
    total = sum(scores.values())
    detail = " | ".join(f"{k}={v:.2f}" for k, v in scores.items() if v > 0)
    return {"key": "cypher_complexity_score", "score": round(total, 3), "comment": detail}


# ─────────────────────────────────────────────
# 2. MULTI-HOP DETECTION + EVALUATION
# ─────────────────────────────────────────────

class MultihopGrade(TypedDict):
    requires_multihop: Annotated[bool, ..., "True if the question requires traversing 2+ relationships"]
    hops_required: Annotated[int, ..., "Number of relationship hops required (1, 2, or 3+)"]
    explanation: Annotated[str, ..., "Why this hop count is needed"]

_multihop_llm = _llm.with_structured_output(MultihopGrade, method="json_schema", strict=True)

_MULTIHOP_PROMPT = """You are an expert in Neo4j graph databases and Cypher queries.
Analyze a question and determine if answering it requires traversing multiple relationships in the graph.

Graph schema context:
- Employee -[:REPORTS_TO]-> Employee
- Order -[:ORDERED_BY]-> Customer
- Order -[:PROCESSED_BY]-> Employee
- Order -[:SHIPPED_BY]-> Shipper
- Order -[:INCLUDES]-> Product
- Product -[:PART_OF]-> Category
- Product -[:SUPPLIED_BY]-> Supplier

Single-hop (1): Direct property lookup or single relationship traversal
  e.g., "Which employees work in London?" → MATCH (e:Employee) WHERE e.city='London'

Multi-hop (2): Requires crossing 2 relationships
  e.g., "Which categories do employees in London sell?" → Employee→Order→Product→Category (3 hops)
  e.g., "Who manages the employees reporting to Fuller?" → Employee→Employee (1 hop, still counts as 2-level hierarchy)

Multi-hop (3+): Requires 3+ relationship traversals
  e.g., "What products did customers from Germany order through employees reporting to Fuller?" → Customer→Order→Product + Employee→Order
"""

def multihop_required_detector(inputs: dict, outputs: dict) -> dict:
    """
    Detecta si la pregunta requería razonamiento multi-hop.
    Útil como feature de segmentación del dataset.
    """
    grade = _multihop_llm.invoke([
        {"role": "system", "content": _MULTIHOP_PROMPT},
        {"role": "user", "content": f"QUESTION: {inputs['question']}"}
    ])
    return {
        "key": "multihop_required",
        "score": int(grade["requires_multihop"]),
        "comment": f"hops_required={grade['hops_required']} | {grade['explanation']}"
    }


def multihop_execution_score(inputs: dict, outputs: dict) -> dict:
    """
    Si la pregunta requería multi-hop, ¿el Cypher lo ejecutó correctamente?
    Score: 1.0 = sí, 0.5 = parcialmente, 0.0 = no (o pregunta no era multi-hop)
    """
    cypher = _get_cypher(outputs)
    if not cypher:
        return {"key": "multihop_execution_score", "score": 0.0, "comment": "No Cypher"}

    # Detectar si la pregunta requiere multi-hop
    mh = _multihop_llm.invoke([
        {"role": "system", "content": _MULTIHOP_PROMPT},
        {"role": "user", "content": f"QUESTION: {inputs['question']}"}
    ])

    if not mh["requires_multihop"]:
        return {"key": "multihop_execution_score", "score": 1.0, "comment": "Single-hop question, N/A"}

    required_hops = mh["hops_required"]
    # Contar relaciones en el Cypher generado
    actual_relationships = len(re.findall(r'-\[.*?\]->', cypher))

    if actual_relationships >= required_hops:
        score, note = 1.0, f"Correct: {actual_relationships} relationships for {required_hops} required"
    elif actual_relationships == required_hops - 1:
        score, note = 0.5, f"Partial: {actual_relationships} relationships, needed {required_hops}"
    else:
        score, note = 0.0, f"Insufficient: {actual_relationships} relationships, needed {required_hops}"

    return {"key": "multihop_execution_score", "score": score, "comment": note}


# ─────────────────────────────────────────────
# 3. FAILURE MODE CLASSIFIER
# ─────────────────────────────────────────────

class FailureModeGrade(TypedDict):
    has_failure: Annotated[bool, ..., "True if there is a clear failure mode"]
    failure_mode: Annotated[str, ..., "One of: none, schema_error, syntax_error, semantic_error, empty_result_hallucination, relationship_direction_error, aggregation_error, property_name_error"]
    severity: Annotated[str, ..., "One of: none, minor, major, critical"]
    explanation: Annotated[str, ..., "Brief explanation of the failure"]

_failure_llm = _llm.with_structured_output(FailureModeGrade, method="json_schema", strict=True)

_FAILURE_MODE_PROMPT = """You are an expert in Neo4j GraphRAG systems.
Analyze a question, the Cypher query generated, the database results, and the final answer to classify any failure mode.

Failure modes:
- none: No failure, everything correct
- schema_error: Cypher uses labels/properties that don't exist in schema
- syntax_error: Cypher has syntax issues (even if it ran)
- semantic_error: Cypher is syntactically valid but doesn't answer the question correctly
- empty_result_hallucination: DB returned empty results but LLM gave a confident answer
- relationship_direction_error: Cypher traverses a relationship in wrong direction
- aggregation_error: Question needed COUNT/SUM but Cypher returned raw rows, or vice versa
- property_name_error: Cypher uses wrong property name (e.g., 'name' instead of 'firstName')

Severity:
- none: No failure
- minor: Small issue, answer still mostly correct
- major: Significant issue, answer partially wrong
- critical: Complete failure, answer wrong or hallucinated
"""

def failure_mode_classifier(inputs: dict, outputs: dict) -> list:
    """
    Clasifica el modo de fallo del GraphRAG.
    Retorna múltiples métricas como lista (feature de LangSmith).
    """
    cypher = _get_cypher(outputs)
    db_results = _get_db_results(outputs)
    answer = outputs.get("answer", "")

    prompt = f"""QUESTION: {inputs['question']}
CYPHER GENERATED: {cypher or 'None'}
DATABASE RESULTS: {str(db_results)[:500]}
FINAL ANSWER: {answer[:300]}"""

    grade = _failure_llm.invoke([
        {"role": "system", "content": _FAILURE_MODE_PROMPT},
        {"role": "user", "content": prompt}
    ])

    severity_score = {"none": 1.0, "minor": 0.75, "major": 0.35, "critical": 0.0}

    return [
        {
            "key": "failure_mode",
            "score": 0 if grade["has_failure"] else 1,
            "comment": grade["failure_mode"]
        },
        {
            "key": "failure_severity",
            "score": severity_score.get(grade["severity"], 0.5),
            "comment": f"{grade['severity']}: {grade['explanation']}"
        }
    ]


# ─────────────────────────────────────────────
# 4. RELATIONSHIP DIRECTION SCORE
# ─────────────────────────────────────────────

# Mapa de relaciones correctas en Northwind
NORTHWIND_RELATIONSHIPS = {
    "REPORTS_TO": ("Employee", "Employee"),     # (e1)-[:REPORTS_TO]->(e2): e1 reports to e2
    "ORDERED_BY": ("Order", "Customer"),        # Order → Customer
    "PROCESSED_BY": ("Order", "Employee"),      # Order → Employee
    "SHIPPED_BY": ("Order", "Shipper"),         # Order → Shipper
    "INCLUDES": ("Order", "Product"),           # Order → Product
    "PART_OF": ("Product", "Category"),         # Product → Category
    "SUPPLIED_BY": ("Product", "Supplier"),     # Product → Supplier
}

def relationship_direction_score(inputs: dict, outputs: dict) -> dict:
    """
    Verifica que las relaciones en el Cypher se traversan en la dirección correcta.
    Score: fracción de relaciones con dirección correcta.
    Funciona con el schema de Northwind — adaptar para otros grafos.
    """
    cypher = _get_cypher(outputs)
    if not cypher:
        return {"key": "relationship_direction_score", "score": 1.0, "comment": "No Cypher"}

    # Extraer relaciones usadas: patrón (a)-[:REL]->(b) o (a)<-[:REL]-(b)
    forward = re.findall(r'\)-\[:(\w+)\]->', cypher)
    backward = re.findall(r'<-\[:(\w+)\]-\(', cypher)

    issues = []
    correct = 0
    total = 0

    for rel in forward:
        if rel in NORTHWIND_RELATIONSHIPS:
            total += 1
            correct += 1  # Dirección hacia adelante → esperada

    for rel in backward:
        if rel in NORTHWIND_RELATIONSHIPS:
            total += 1
            # Relación inversa — en Northwind casi siempre es un error
            issues.append(f"{rel} used backwards")

    if total == 0:
        return {"key": "relationship_direction_score", "score": 1.0, "comment": "No known relationships found"}

    score = correct / total
    comment = " | ".join(issues) if issues else "All relationship directions correct"
    return {"key": "relationship_direction_score", "score": round(score, 3), "comment": comment}


# ─────────────────────────────────────────────
# 5. ANSWER COMPLETENESS
# ─────────────────────────────────────────────

class CompletenessGrade(TypedDict):
    score: Annotated[float, ..., "Completeness score 0.0 to 1.0"]
    missing_aspects: Annotated[str, ..., "What aspects of the question were not addressed"]
    explanation: Annotated[str, ..., "Brief explanation"]

_completeness_llm = _llm.with_structured_output(CompletenessGrade, method="json_schema", strict=True)

_COMPLETENESS_PROMPT = """You are evaluating if a student answer addresses ALL aspects of a question.
Score:
- 1.0: Answer completely addresses all parts of the question
- 0.7: Answer addresses the main question but misses secondary details
- 0.4: Answer partially addresses the question
- 0.0: Answer completely misses the question or only addresses a minor part

Focus on completeness, not correctness. An answer can be complete but wrong."""

def answer_completeness(inputs: dict, outputs: dict) -> dict:
    """¿La respuesta aborda TODOS los aspectos de la pregunta?"""
    answer = outputs.get("answer", "")
    prompt = f"QUESTION: {inputs['question']}\nSTUDENT ANSWER: {answer}"
    grade = _completeness_llm.invoke([
        {"role": "system", "content": _COMPLETENESS_PROMPT},
        {"role": "user", "content": prompt}
    ])
    return {
        "key": "answer_completeness",
        "score": round(float(grade["score"]), 3),
        "comment": grade["explanation"]
    }


# ─────────────────────────────────────────────
# 6. CONFIDENCE SCORE v2 (más completo)
# ─────────────────────────────────────────────

from rag_eval.evaluators.base import (
    correctness, relevance, graphrag_groundedness,
    cypher_result_nonempty, schema_adherence, empty_context_hallucination
)

def confidence_score_v2(inputs: dict, outputs: dict, reference_outputs: dict) -> list:
    """
    Score de confianza v2 — usa SOLO métricas deterministas para evitar duplicar LLM calls.
    Las métricas LLM (correctness, groundedness, etc.) se calculan como evaluadores separados
    y se combinan en post-proceso con compute_confidence_from_scores().

    Structural score: combina señales del Cypher y retrieval.
    """
    nonempty   = cypher_result_nonempty(inputs, outputs)["score"]
    schema_sc  = schema_adherence(inputs, outputs)["score"]
    rel_dir    = relationship_direction_score(inputs, outputs)["score"]
    halluc_ok  = empty_context_hallucination(inputs, outputs)["score"]
    complexity = cypher_complexity_score(inputs, outputs)["score"]

    structural = 0.5 * nonempty + 0.3 * schema_sc + 0.2 * rel_dir
    penalty = 0.40 if halluc_ok == 0 else 0.0
    structural_penalized = round(max(0.0, structural - penalty), 3)

    comment = (
        f"nonempty={nonempty} schema={schema_sc:.2f} rel_dir={rel_dir:.2f} "
        f"complexity={complexity:.2f} halluc_penalty={'YES' if penalty > 0 else 'NO'} "
        f"→ structural={structural_penalized}"
    )

    return [
        {"key": "structural_score",   "score": structural_penalized, "comment": comment},
        {"key": "cypher_complexity",  "score": round(complexity, 3), "comment": "Cypher structural richness [0,1]"},
    ]


def compute_confidence_from_scores(
    correctness_score: float,
    groundedness_score: float,
    structural_score: float,
    completeness_score: float = 1.0,
    relevance_score: float = 1.0,
    hallucination_ok: bool = True,
) -> float:
    """
    Combina métricas ya calculadas en un confidence score final.
    Útil para post-procesado sobre resultados de LangSmith.

    Fórmula:
        quality    = 0.4*correctness + 0.3*groundedness + 0.2*completeness + 0.1*relevance
        confidence = 0.4*structural + 0.6*quality  - 0.40 si hallucination
    """
    quality    = 0.4 * correctness_score + 0.3 * groundedness_score + 0.2 * completeness_score + 0.1 * relevance_score
    base       = 0.4 * structural_score + 0.6 * quality
    penalty    = 0.40 if not hallucination_ok else 0.0
    return round(max(0.0, min(1.0, base - penalty)), 3)


# ─────────────────────────────────────────────
# 7. CALIBRATION REPORT
# ─────────────────────────────────────────────

def compute_calibration_report(
    scores: List[float],
    labels: List[int],
    n_bins: int = 5
) -> dict:
    """
    Calcula Expected Calibration Error (ECE) y statistics de calibración.

    Args:
        scores: Lista de confidence scores [0,1] del sistema
        labels: Lista de labels binarios (1=correcto, 0=incorrecto)
        n_bins: Número de bins para ECE

    Returns:
        dict con ECE, accuracy, avg_confidence, y bins detallados
    """
    if len(scores) != len(labels) or len(scores) == 0:
        return {"error": "Invalid input"}

    n = len(scores)
    bin_size = 1.0 / n_bins
    bins = []

    for i in range(n_bins):
        lo = i * bin_size
        hi = (i + 1) * bin_size
        indices = [j for j, s in enumerate(scores) if lo <= s < hi]
        if i == n_bins - 1:  # último bin incluye 1.0
            indices = [j for j, s in enumerate(scores) if lo <= s <= hi]

        if not indices:
            continue

        bin_conf = sum(scores[j] for j in indices) / len(indices)
        bin_acc  = sum(labels[j] for j in indices) / len(indices)
        bins.append({
            "range": f"[{lo:.1f}, {hi:.1f})",
            "n": len(indices),
            "avg_confidence": round(bin_conf, 3),
            "accuracy": round(bin_acc, 3),
            "gap": round(abs(bin_conf - bin_acc), 3)
        })

    # ECE = weighted average of |confidence - accuracy| per bin
    ece = sum(b["n"] * b["gap"] for b in bins) / n if bins else 0.0
    overall_acc = sum(labels) / n
    overall_conf = sum(scores) / n

    return {
        "n_samples": n,
        "ece": round(ece, 4),
        "overall_accuracy": round(overall_acc, 3),
        "overall_confidence": round(overall_conf, 3),
        "confidence_gap": round(overall_conf - overall_acc, 3),
        "bins": bins,
        "interpretation": (
            "Well calibrated (ECE < 0.05)" if ece < 0.05
            else "Slightly miscalibrated (0.05 ≤ ECE < 0.15)" if ece < 0.15
            else "Poorly calibrated (ECE ≥ 0.15)"
        )
    }


# ─────────────────────────────────────────────
# 8. EVALUATE_GRAPHRAG_ADVANCED — función principal
# ─────────────────────────────────────────────

from langsmith import Client
from langsmith.utils import LangSmithConflictError
import time


def evaluate_graphrag_advanced(
    rag_fn: Callable[[dict], dict],
    dataset: List[Dict],
    dataset_name: Optional[str] = None,
    project: str = "graphrag-eval-advanced",
    evaluator_set: str = "full",  # "full", "fast", "structural", "semantic"
    **kwargs
):
    """
    Evaluador avanzado para GraphRAG — todos los evaluadores especializados.

    evaluator_set:
        "fast"       — solo deterministas (sin LLM, muy rápido)
        "structural" — deterministas + Cypher quality (sin answer quality)
        "semantic"   — LLM judges de answer quality
        "full"       — todo (recomendado)
    """
    client = Client()

    if dataset_name is None:
        dataset_name = f"graphrag-advanced-{int(time.time())}"

    # Crear dataset
    try:
        ds = client.create_dataset(dataset_name=dataset_name)
        client.create_examples(dataset_id=ds.id, examples=dataset)
        print(f"✅ Dataset '{dataset_name}' ({len(dataset)} ejemplos)")
    except (LangSmithConflictError, Exception) as e:
        if "already exists" in str(e) or "Conflict" in str(e):
            datasets = list(client.list_datasets(dataset_name=dataset_name))
            if not datasets:
                raise RuntimeError(f"Dataset '{dataset_name}' existe pero no encontrado.")
            print(f"⚠️  Dataset '{dataset_name}' reutilizado.")
        else:
            raise

    # Seleccionar evaluadores según el preset
    fast_evals = [
        cypher_complexity_score,
        relationship_direction_score,
    ]
    # importar los básicos
    from rag_eval.evaluators.base import (
        cypher_generated, cypher_result_nonempty,
        empty_context_hallucination, schema_adherence
    )
    deterministic = [
        cypher_generated,
        cypher_result_nonempty,
        empty_context_hallucination,
        schema_adherence,
        cypher_complexity_score,
        relationship_direction_score,
    ]
    semantic_evals = [
        correctness,
        relevance,
        graphrag_groundedness,
        answer_completeness,
        multihop_required_detector,
        multihop_execution_score,
        failure_mode_classifier,
        confidence_score_v2,
    ]

    presets = {
        "fast":       deterministic,
        "structural": deterministic + [cypher_complexity_score],
        "semantic":   semantic_evals,
        "full":       deterministic + semantic_evals,
    }
    evaluators = presets.get(evaluator_set, presets["full"])

    # Eliminar duplicados manteniendo orden
    seen = set()
    evaluators = [e for e in evaluators if not (e.__name__ in seen or seen.add(e.__name__))]

    print(f"\n🚀 Evaluadores ({len(evaluators)}) — preset '{evaluator_set}':")
    for ev in evaluators:
        icon = "⚙️ " if ev.__name__ in (
            "cypher_generated", "cypher_result_nonempty",
            "empty_context_hallucination", "schema_adherence",
            "cypher_complexity_score", "relationship_direction_score"
        ) else "🤖 "
        print(f"   {icon}{ev.__name__}")
    print()

    results = client.evaluate(
        rag_fn,
        data=dataset_name,
        evaluators=evaluators,
        experiment_prefix=project,
        **kwargs
    )
    print("\n✅ Evaluación completada!")
    print(f"🌐 https://smith.langchain.com")
    return results


def print_results_summary(experiment_name: str):
    """Imprime resumen de métricas de un experimento de LangSmith."""
    client = Client()
    runs = list(client.list_runs(project_name=experiment_name, run_type="chain"))

    all_scores = {}
    for run in runs:
        for fb in client.list_feedback(run_ids=[run.id]):
            if fb.score is not None:
                all_scores.setdefault(fb.key, []).append(fb.score)

    if not all_scores:
        print("No scores found.")
        return

    print(f"\n{'='*50}")
    print(f"RESUMEN: {experiment_name}")
    print(f"{'='*50}")

    groups = {
        "⚙️  DETERMINISTAS":  ["cypher_generated", "cypher_result_nonempty", "empty_context_hallucination", "schema_adherence"],
        "📐 ESTRUCTURA":      ["cypher_complexity_score", "relationship_direction_score"],
        "🤖 ANSWER QUALITY":  ["correctness", "relevance", "graphrag_groundedness", "answer_completeness"],
        "🧠 RAZONAMIENTO":    ["multihop_required", "multihop_execution_score", "failure_mode", "failure_severity"],
        "🎯 SCORES COMP.":    ["confidence_score_v2", "structural_score", "quality_score"],
    }

    for group_name, keys in groups.items():
        found = {k: all_scores[k] for k in keys if k in all_scores}
        if not found:
            continue
        print(f"\n{group_name}:")
        for k, scores in found.items():
            mean = sum(scores) / len(scores)
            bar = "█" * int(mean * 10) + "░" * (10 - int(mean * 10))
            print(f"   {k:<35} {bar} {mean:.3f}  (n={len(scores)})")

    # Confidence score v2 distribution
    if "confidence_score_v2" in all_scores:
        c = all_scores["confidence_score_v2"]
        print(f"\n🎯 CONFIDENCE V2 DISTRIBUTION (n={len(c)}):")
        bins = [(0, 0.3, "Bajo [0-30%)"), (0.3, 0.6, "Medio [30-60%)"),
                (0.6, 0.85, "Alto [60-85%)"), (0.85, 1.01, "Muy alto [85-100%]")]
        for lo, hi, label in bins:
            cnt = sum(1 for s in c if lo <= s < hi)
            bar = "█" * cnt
            print(f"   {label:<20} {bar} {cnt}")
        print(f"   Promedio: {sum(c)/len(c):.3f} ({sum(c)/len(c)*100:.1f}%)")
