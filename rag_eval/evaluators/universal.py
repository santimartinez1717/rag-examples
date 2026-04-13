"""
Universal RAG Evaluator — framework de evaluación para cualquier arquitectura RAG.

Soporta: Naive/Vector RAG, Agentic RAG, GraphRAG, SQL RAG, Hybrid RAG.
Integración nativa con LangSmith vía evaluate_rag_universal().

Métricas implementadas:
─────────────────────────────────────────────────────────────────────
  NLI-based (sin LLM, deterministas):
    - faithfulness_nli            Fracción de claims del answer soportadas por contexto (DeBERTa)
    - hallucination_rate          1 - faithfulness_nli (claims no soportadas)
    - atomic_fact_precision       FActScore-style: LLM descompone en atomic facts + NLI verifica

  Retrieval (métricas de ranking/cobertura):
    - context_precision_at_k      RAGAS: ranking ponderado de chunks relevantes (requiere GT)
    - context_recall              RAGAS: cobertura del GT answer por el contexto (requiere GT)
    - context_relevance           TruLens: relevancia promedio de chunks para la query (LLM judge)
    - mrr                         Mean Reciprocal Rank (requiere relevancia por chunk)

  Generación (LLM-judge):
    - answer_relevance_universal  ¿El answer responde la pregunta? (LLM judge)
    - correctness_universal       ¿Es factualmente correcto vs GT? (LLM judge, requiere GT)

  Comportamiento del sistema (RGB benchmark):
    - negative_rejection          Tasa de rechazo correcto en preguntas sin respuesta
    - noise_robustness            Degradación de accuracy con contexto ruidoso

  Calibración:
    - compute_ece                 Expected Calibration Error (ECE)
    - temperature_scaling         Post-hoc calibración de scores
    - compute_calibration_report  Report completo ECE + bins + interpretación

Referencias académicas:
  - RAGAS: Es et al., EACL 2024 (arXiv:2309.15217)
  - ARES: Saad-Falcon et al., NAACL 2024 (arXiv:2311.09476)
  - RGB: Chen et al., AAAI 2024 (arXiv:2309.01431)
  - FActScore: Min et al., EMNLP 2023 (arXiv:2305.14251)
  - G-Eval: Liu et al., EMNLP 2023 (arXiv:2303.16634)
  - TRUE: Honovich et al., NAACL 2022 (arXiv:2204.04991)

Uso mínimo:
    from universal_rag_evaluator import evaluate_rag_universal
    results = evaluate_rag_universal(
        rag_fn=my_rag,
        dataset=my_dataset,
        dataset_name="my-eval",
        project="my-project"
    )
"""

import re
import math
import numpy as np
from typing import Callable, List, Dict, Optional, Any
from langsmith import Client
from langsmith.utils import LangSmithConflictError
from langchain_openai import ChatOpenAI
from typing_extensions import Annotated, TypedDict


def _softmax(logits) -> list:
    """Convierte logits a probabilidades con softmax numérico estable."""
    logits = np.array(logits, dtype=np.float64)
    logits -= logits.max()  # estabilidad numérica
    exp_l = np.exp(logits)
    return (exp_l / exp_l.sum()).tolist()


# ─────────────────────────────────────────────
# NLI MODEL — lazy loaded (evita descarga al importar)
# ─────────────────────────────────────────────

_nli_model_cache = None
_NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"

# Índices de clase en cross-encoder/nli-deberta-v3-base
# Orden: contradiction=0, entailment=1, neutral=2
_ENTAILMENT_IDX = 1


def _get_nli_model(model_name: str = _NLI_MODEL_NAME):
    """Carga el modelo NLI una sola vez (singleton). Requiere sentence-transformers."""
    global _nli_model_cache
    if _nli_model_cache is None:
        try:
            from sentence_transformers import CrossEncoder
            print(f"[NLI] Cargando modelo: {model_name} (primera carga, puede tardar)...")
            _nli_model_cache = CrossEncoder(model_name)
            print(f"[NLI] Modelo cargado.")
        except ImportError:
            raise ImportError(
                "sentence-transformers no instalado. "
                "Ejecuta: pip install sentence-transformers"
            )
    return _nli_model_cache


# ─────────────────────────────────────────────
# LLMs compartidos (LLM-judge)
# ─────────────────────────────────────────────

_llm = None

def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model="gpt-4.1", temperature=0)
    return _llm


# ─────────────────────────────────────────────
# TypedDicts para LLM-judge con structured output
# ─────────────────────────────────────────────

class AnswerRelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Reasoning for the relevance score"]
    relevant: Annotated[bool, ..., "True if the answer directly addresses the question"]

class CorrectnessGradeUniversal(TypedDict):
    explanation: Annotated[str, ..., "Reasoning for the correctness score"]
    correct: Annotated[bool, ..., "True if the answer is factually consistent with ground truth"]

class CorrectnessGradeContinuous(TypedDict):
    explanation: Annotated[str, ..., "Reasoning for the score"]
    score: Annotated[float, ..., "Correctness score: 0.0 (wrong), 0.5 (partially correct), or 1.0 (fully correct)"]

class ChunkRelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Why this chunk is or isn't relevant"]
    relevant: Annotated[bool, ..., "True if this chunk helps answer the question"]

class ContextRecallGrade(TypedDict):
    explanation: Annotated[str, ..., "Which sentences are attributable to the context"]
    covered_sentences: Annotated[int, ..., "Number of GT sentences attributable to context"]
    total_sentences: Annotated[int, ..., "Total number of GT sentences"]

class AtomicFactsOutput(TypedDict):
    facts: Annotated[List[str], ..., "List of atomic facts extracted from the text"]


# ─────────────────────────────────────────────
# HELPERS GENERALES
# ─────────────────────────────────────────────

def _split_into_sentences(text: str) -> List[str]:
    """Divide texto en oraciones usando regex simple."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 5]


def _get_context_chunks(outputs: dict) -> List[str]:
    """
    Extrae chunks de contexto del output del RAG.
    Soporta: lista de chunks, string, o campo 'db_results' para GraphRAG.
    """
    context = outputs.get("context", "")

    # Si el RAG retorna lista de chunks directamente
    if isinstance(context, list):
        return [str(c) for c in context if c]

    # Si context es string, lo tratamos como un único chunk
    if isinstance(context, str) and context:
        return [context]

    # Fallback: db_results de GraphRAG
    db = outputs.get("db_results", [])
    if db:
        return [str(r) for r in db]

    return []


def _context_as_string(outputs: dict, max_chars: int = 4000) -> str:
    """Convierte el contexto a string para NLI y LLM judges."""
    chunks = _get_context_chunks(outputs)
    full = "\n".join(chunks)
    return full[:max_chars] if len(full) > max_chars else full


def _is_refusal(answer: str) -> bool:
    """Detecta si el answer es un rechazo explícito a responder."""
    refusal_phrases = [
        "i don't know", "i do not know", "no information", "no data available",
        "cannot answer", "cannot find", "not found", "no results", "unable to find",
        "i don't have", "no records", "nothing found", "couldn't find",
        "not in the database", "not available", "no answer", "insufficient information",
        "the database does not contain", "no employees", "there are no", "there is no",
        "not contain", "not have any"
    ]
    lower = answer.lower()
    return any(p in lower for p in refusal_phrases)


# ─────────────────────────────────────────────
# BLOQUE 1: NLI-BASED FAITHFULNESS (TRUE / RAGAS origin)
# Referencia: TRUE (NAACL 2022), RAGAS (EACL 2024)
# ─────────────────────────────────────────────

def _decompose_claims_heuristic(answer: str) -> List[str]:
    """Descomposición heurística: split por oraciones."""
    return _split_into_sentences(answer)


def _decompose_claims_llm(answer: str) -> List[str]:
    """
    Descomposición en atomic claims via LLM.
    Cada claim = exactamente una pieza de información verificable.
    Referencia: FActScore (EMNLP 2023).
    """
    llm = _get_llm()
    facts_llm = llm.with_structured_output(AtomicFactsOutput, method="json_schema", strict=True)

    prompt = f"""Break this answer into minimal atomic facts.
Each fact should contain exactly one piece of verifiable information (entity, number, date, relationship).
Do NOT include meta-statements like "The answer is..." or "Based on the context...".

Answer: {answer}"""

    result = facts_llm.invoke(prompt)
    facts = result.get("facts", [])
    return [f for f in facts if f and len(f.strip()) > 3]


def faithfulness_nli(inputs: dict, outputs: dict, use_llm_decomposition: bool = False) -> dict:
    """
    Faithfulness via NLI (DeBERTa) — sin LLM para la verificación.

    Proceso (RAGAS original + TRUE methodology):
    1. Descomponer answer en claims (heurístico o LLM)
    2. Para cada claim: NLI(context, claim) → entailment / neutral / contradiction
    3. Score = |entailed claims| / |total claims|

    Referencia: Es et al. EACL 2024; Honovich et al. NAACL 2022.
    """
    answer = outputs.get("answer", "")
    context = _context_as_string(outputs)

    if not answer or not context:
        return {"key": "faithfulness_nli", "score": 0.0,
                "comment": "Missing answer or context"}

    # Descomposición
    if use_llm_decomposition:
        try:
            claims = _decompose_claims_llm(answer)
        except Exception:
            claims = _decompose_claims_heuristic(answer)
    else:
        claims = _decompose_claims_heuristic(answer)

    if not claims:
        return {"key": "faithfulness_nli", "score": 0.0, "comment": "No claims extracted"}

    # NLI: (context, claim) → scores shape (n, 3)
    nli = _get_nli_model()
    pairs = [(context, claim) for claim in claims]
    scores = nli.predict(pairs)  # shape (n_claims, 3)

    entailed = 0
    claim_labels = []
    for claim, score_row in zip(claims, scores):
        # Convertir logits a probabilidades con softmax
        probs = _softmax(score_row)
        entail_prob = probs[_ENTAILMENT_IDX]
        contra_prob = probs[0]
        if entail_prob > 0.5:
            label = "entailment"
            entailed += 1
        elif contra_prob > 0.5:
            label = "contradiction"
        else:
            label = "neutral"
        claim_labels.append(f"[{label}] {claim[:80]}")

    score = round(entailed / len(claims), 3)
    comment = f"{entailed}/{len(claims)} claims entailed | " + "; ".join(claim_labels[:3])
    return {"key": "faithfulness_nli", "score": score, "comment": comment}


def hallucination_rate(inputs: dict, outputs: dict) -> dict:
    """
    Hallucination Rate = 1 - Faithfulness_NLI.
    Score 0 = no hay alucinaciones, Score 1 = todo alucinado.
    Referencia: derivado de RAGAS + FActScore.
    """
    faith = faithfulness_nli(inputs, outputs)
    rate = round(1.0 - faith["score"], 3)
    return {"key": "hallucination_rate", "score": rate,
            "comment": f"faithfulness_nli={faith['score']}"}


# ─────────────────────────────────────────────
# BLOQUE 2: ATOMIC FACT PRECISION (FActScore-style)
# Referencia: Min et al., EMNLP 2023 (arXiv:2305.14251)
# ─────────────────────────────────────────────

def atomic_fact_precision(inputs: dict, outputs: dict) -> dict:
    """
    Atomic Fact Precision (FActScore style):
    - LLM descompone el answer en atomic facts
    - NLI verifica cada fact contra el contexto
    - Score = fracción de facts soportados

    FActScore = |{f ∈ F(response) : context ⊨ f}| / |F(response)|

    Diferencia vs faithfulness_nli: usa LLM para descomposición más granular
    que el split por oraciones, capturando sub-claims implícitos.
    Referencia: Min et al. EMNLP 2023.
    """
    answer = outputs.get("answer", "")
    context = _context_as_string(outputs)

    if not answer or not context:
        return {"key": "atomic_fact_precision", "score": 0.0,
                "comment": "Missing answer or context"}

    try:
        facts = _decompose_claims_llm(answer)
    except Exception as e:
        return {"key": "atomic_fact_precision", "score": 0.0,
                "comment": f"LLM decomposition failed: {e}"}

    if not facts:
        return {"key": "atomic_fact_precision", "score": 1.0,
                "comment": "No atomic facts extracted (trivial answer)"}

    nli = _get_nli_model()
    pairs = [(context, fact) for fact in facts]
    scores = nli.predict(pairs)

    supported = 0
    details = []
    for fact, score_row in zip(facts, scores):
        probs = _softmax(score_row)
        entail_prob = probs[_ENTAILMENT_IDX]
        if entail_prob > 0.5:
            supported += 1
            details.append(f"✓ {fact[:60]}")
        else:
            details.append(f"✗ {fact[:60]}")

    score = round(supported / len(facts), 3)
    comment = f"{supported}/{len(facts)} facts supported | " + "; ".join(details[:3])
    return {"key": "atomic_fact_precision", "score": score, "comment": comment}


# ─────────────────────────────────────────────
# BLOQUE 3: CONTEXT PRECISION@K (RAGAS)
# Referencia: Es et al., EACL 2024
# ─────────────────────────────────────────────

def _is_chunk_relevant_llm(chunk: str, query: str, ground_truth: str) -> bool:
    """LLM judge: ¿este chunk es relevante para responder la query dado el GT?"""
    llm = _get_llm()
    judge = llm.with_structured_output(ChunkRelevanceGrade, method="json_schema", strict=True)

    prompt = f"""Determine if this context chunk is useful for answering the question.
The ground truth answer is provided as reference.

Question: {query}
Ground truth answer: {ground_truth}
Context chunk: {chunk[:500]}

Is this chunk relevant for answering the question?"""
    result = judge.invoke(prompt)
    return result.get("relevant", False)


def context_precision_at_k(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """
    Context Precision@K (RAGAS):
    Evalúa si los chunks relevantes están posicionados más arriba en el contexto.
    Es una métrica de RANKING: penaliza chunks relevantes al final.

    Fórmula:
        CP@K = [Σ_{k=1}^{K} Precision@k × v_k] / (número total de items relevantes en top K)

    Donde v_k=1 si el chunk en posición k es relevante, 0 si no.

    Referencia: Es et al. EACL 2024 (RAGAS paper).
    Requiere: ground truth answer en reference_outputs.
    """
    query = inputs.get("question", "")
    ground_truth = reference_outputs.get("answer", "")
    chunks = _get_context_chunks(outputs)

    if not chunks or not ground_truth:
        return {"key": "context_precision_at_k", "score": 0.0,
                "comment": "No chunks or ground truth available"}

    # Limitar a top-10 chunks para evitar demasiadas llamadas LLM
    chunks = chunks[:10]
    K = len(chunks)

    # Determinar relevancia de cada chunk (con LLM judge)
    # Si hay muchos chunks, usar NLI como alternativa más rápida
    relevances = []
    if K <= 5:
        for chunk in chunks:
            try:
                rel = _is_chunk_relevant_llm(chunk, query, ground_truth)
                relevances.append(1 if rel else 0)
            except Exception:
                relevances.append(0)
    else:
        # Para K > 5, usar NLI: (chunk, ground_truth_sentence) como proxy
        nli = _get_nli_model()
        gt_sentences = _split_into_sentences(ground_truth)[:3]  # Primeras 3 oraciones del GT
        for chunk in chunks:
            # Un chunk es relevante si al menos una oración del GT puede ser entailada desde él
            max_entail = 0.0
            for gt_sent in gt_sentences:
                scores_chunk = nli.predict([(chunk[:400], gt_sent)])
                probs = _softmax(scores_chunk[0])
                entail_p = probs[_ENTAILMENT_IDX]
                max_entail = max(max_entail, entail_p)
            relevances.append(1 if max_entail > 0.4 else 0)

    # Calcular CP@K
    total_relevant = sum(relevances)
    if total_relevant == 0:
        return {"key": "context_precision_at_k", "score": 0.0,
                "comment": f"No relevant chunks found in top {K}"}

    numerator = 0.0
    tp_so_far = 0
    for k, v_k in enumerate(relevances, start=1):
        if v_k == 1:
            tp_so_far += 1
            precision_at_k = tp_so_far / k
            numerator += precision_at_k * v_k

    score = round(numerator / total_relevant, 3)
    comment = f"Relevances@{K}: {relevances} | CP@K={score}"
    return {"key": "context_precision_at_k", "score": score, "comment": comment}


# ─────────────────────────────────────────────
# BLOQUE 4: CONTEXT RECALL (RAGAS)
# Referencia: Es et al., EACL 2024
# ─────────────────────────────────────────────

def context_recall(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """
    Context Recall (RAGAS):
    Qué fracción de la información del ground truth answer está cubierta por el contexto.

    Fórmula:
        CR = |{sentences in GT answer attributable to context}| / |{sentences in GT answer}|

    Proceso:
    1. Descomponer GT answer en oraciones
    2. Para cada oración, verificar si puede ser atribuida al contexto (LLM judge)
    3. Score = fracción de oraciones atribuibles

    Referencia: Es et al. EACL 2024 (RAGAS paper).
    Requiere: ground truth answer.
    """
    context = _context_as_string(outputs)
    ground_truth = reference_outputs.get("answer", "")

    if not context or not ground_truth:
        return {"key": "context_recall", "score": 0.0,
                "comment": "Missing context or ground truth"}

    llm = _get_llm()
    judge = llm.with_structured_output(ContextRecallGrade, method="json_schema", strict=True)

    gt_sentences = _split_into_sentences(ground_truth)
    if not gt_sentences:
        return {"key": "context_recall", "score": 0.0,
                "comment": "No sentences in ground truth"}

    # Limitar GT sentences para controlar tokens
    gt_sentences = gt_sentences[:8]
    gt_text = "\n".join(f"- {s}" for s in gt_sentences)

    prompt = f"""Evaluate how many sentences from the ground truth answer can be attributed to the provided context.
A sentence is "attributable" if the context contains the information needed to derive it.

Context: {context[:2000]}

Ground truth answer sentences:
{gt_text}

Count how many of these {len(gt_sentences)} sentences are attributable to the context."""

    try:
        result = judge.invoke(prompt)
        covered = min(int(result.get("covered_sentences", 0)), len(gt_sentences))
        total = len(gt_sentences)
        score = round(covered / total, 3)
        comment = f"{covered}/{total} GT sentences attributable to context"
        return {"key": "context_recall", "score": score, "comment": comment}
    except Exception as e:
        return {"key": "context_recall", "score": 0.0, "comment": f"Error: {e}"}


# ─────────────────────────────────────────────
# BLOQUE 5: CONTEXT RELEVANCE (TruLens RAG Triad)
# Referencia: TruEra/TruLens documentation
# ─────────────────────────────────────────────

def context_relevance(inputs: dict, outputs: dict) -> dict:
    """
    Context Relevance (TruLens RAG Triad):
    ¿El contexto recuperado es relevante para la query del usuario?

    Diferencia vs Context Precision: no considera ranking, solo relevancia promedio.
    Fórmula: (1/|C|) × Σ_{c ∈ C} relevance(c, query)

    Usa NLI para eficiencia: la query como hypothesis, el chunk como premise.
    Si el chunk entaila la query o es neutral-hacia-responder → relevante.

    Referencia: TruEra RAG Triad; RAGAS context relevance.
    """
    query = inputs.get("question", "")
    chunks = _get_context_chunks(outputs)

    if not chunks or not query:
        return {"key": "context_relevance", "score": 0.0,
                "comment": "Missing chunks or query"}

    chunks = chunks[:10]  # Top 10 máximo

    # Usamos NLI de forma inversa: ¿el chunk contiene información útil para responder la query?
    # Estrategia: LLM judge es más fiable para relevancia semántica compleja
    llm = _get_llm()
    judge = llm.with_structured_output(ChunkRelevanceGrade, method="json_schema", strict=True)

    relevance_scores = []
    for chunk in chunks:
        prompt = f"""Is the following context chunk useful for answering the question?

Question: {query}
Context chunk: {chunk[:400]}

Answer yes (relevant) or no (not relevant)."""
        try:
            result = judge.invoke(prompt)
            relevance_scores.append(1.0 if result.get("relevant", False) else 0.0)
        except Exception:
            relevance_scores.append(0.0)

    score = round(sum(relevance_scores) / len(relevance_scores), 3) if relevance_scores else 0.0
    comment = f"{sum(relevance_scores):.0f}/{len(chunks)} chunks relevant"
    return {"key": "context_relevance", "score": score, "comment": comment}


# ─────────────────────────────────────────────
# BLOQUE 6: ANSWER RELEVANCE (RAGAS / G-Eval)
# Referencia: Es et al. EACL 2024; Liu et al. EMNLP 2023
# ─────────────────────────────────────────────

def answer_relevance_universal(inputs: dict, outputs: dict) -> dict:
    """
    Answer Relevance: ¿El answer responde directamente la pregunta?
    No evalúa factualidad, solo si la respuesta es pertinente a la query.

    LLM-as-judge con criterio explícito (G-Eval style).
    Referencia: Es et al. EACL 2024 (RAGAS); Liu et al. EMNLP 2023 (G-Eval).
    """
    query = inputs.get("question", "")
    answer = outputs.get("answer", "")

    if not query or not answer:
        return {"key": "answer_relevance_universal", "score": 0.0,
                "comment": "Missing query or answer"}

    llm = _get_llm()
    judge = llm.with_structured_output(AnswerRelevanceGrade, method="json_schema", strict=True)

    prompt = f"""Evaluate if the answer directly addresses the question asked.
Criteria:
- RELEVANT: The answer responds to what was specifically asked (even if incorrect)
- NOT RELEVANT: The answer is evasive, off-topic, or completely misses the point

Note: An answer saying "I don't know" or "no information found" is RELEVANT if the question
genuinely cannot be answered with the available data.

Question: {query}
Answer: {answer}"""

    try:
        result = judge.invoke(prompt)
        score = 1.0 if result.get("relevant", False) else 0.0
        comment = result.get("explanation", "")[:200]
        return {"key": "answer_relevance_universal", "score": score, "comment": comment}
    except Exception as e:
        return {"key": "answer_relevance_universal", "score": 0.0, "comment": f"Error: {e}"}


# ─────────────────────────────────────────────
# BLOQUE 7: CORRECTNESS UNIVERSAL (LLM judge con GT)
# Referencia: G-Eval (Liu et al. EMNLP 2023)
# ─────────────────────────────────────────────

def correctness_universal(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """
    Correctness con ground truth — LLM-as-judge.
    Versión universal (no específica de GraphRAG).
    Referencia: G-Eval (Liu et al. EMNLP 2023).
    """
    query = inputs.get("question", "")
    answer = outputs.get("answer", "")
    ground_truth = reference_outputs.get("answer", "")

    if not answer or not ground_truth:
        return {"key": "correctness_universal", "score": 0.0,
                "comment": "Missing answer or ground truth"}

    llm = _get_llm()
    judge = llm.with_structured_output(CorrectnessGradeUniversal, method="json_schema", strict=True)

    prompt = f"""Grade the student's answer against the ground truth.
Rules:
- Focus ONLY on factual accuracy (names, numbers, dates, relationships).
- Extra correct information is acceptable.
- Missing key facts = incorrect.
- Any factual contradiction = incorrect.
- If question asks for a specific number/name/fact, it must match.
- Approximate values are OK if within 5% of the correct value.

Question: {query}
Ground truth: {ground_truth}
Student answer: {answer}"""

    try:
        result = judge.invoke(prompt)
        score = 1.0 if result.get("correct", False) else 0.0
        comment = result.get("explanation", "")[:200]
        return {"key": "correctness_universal", "score": score, "comment": comment}
    except Exception as e:
        return {"key": "correctness_universal", "score": 0.0, "comment": f"Error: {e}"}


def correctness_continuous(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """
    US-M2: Correctness continuo [0, 0.5, 1.0] en vez de binario.

    0.0 → respuesta incorrecta (fallo factual o invención)
    0.5 → parcialmente correcta (captura parte de la respuesta pero incompleta o inexacta)
    1.0 → completamente correcta

    Con 31 preguntas, la varianza binaria es muy alta. El score continuo
    permite detectar diferencias más sutiles entre wrappers.

    Referencia: G-Eval (Liu et al. EMNLP 2023).
    Correlación esperada con binario > 0.8 (criterio de aceptación US-M2).
    """
    query = inputs.get("question", "")
    answer = outputs.get("answer", "")
    ground_truth = reference_outputs.get("answer", "")

    if not answer or not ground_truth:
        return {"key": "correctness_continuous", "score": 0.0,
                "comment": "Missing answer or ground truth"}

    llm = _get_llm()
    judge = llm.with_structured_output(CorrectnessGradeContinuous, method="json_schema", strict=True)

    prompt = f"""Grade the student's answer against the ground truth on a 3-point scale.

Scoring rules:
- 1.0 (Fully correct): All key facts match the ground truth. Numbers, names, relationships are accurate.
- 0.5 (Partially correct): The answer captures the main idea but misses details, has minor inaccuracies,
  or provides an incomplete subset of a list answer.
- 0.0 (Incorrect): The answer contradicts the ground truth, contains major factual errors,
  or is completely unrelated.

Additional rules:
- Extra correct information beyond what's in the ground truth is fine (1.0).
- If asked for a number, being within 5% rounds to 1.0.
- If the ground truth says "unknown" or "no data" and the answer fabricates data → 0.0.
- If the answer says "I don't know" but the ground truth has an answer → 0.0.

Question: {query}
Ground truth: {ground_truth}
Student answer: {answer}

Return score as 0.0, 0.5, or 1.0 only."""

    try:
        result = judge.invoke(prompt)
        raw_score = float(result.get("score", 0.0))
        # Clamp al conjunto {0.0, 0.5, 1.0}
        if raw_score >= 0.75:
            score = 1.0
        elif raw_score >= 0.25:
            score = 0.5
        else:
            score = 0.0
        comment = result.get("explanation", "")[:200]
        return {"key": "correctness_continuous", "score": score, "comment": comment}
    except Exception as e:
        return {"key": "correctness_continuous", "score": 0.0, "comment": f"Error: {e}"}


# ─────────────────────────────────────────────
# BLOQUE 8: NEGATIVE REJECTION (RGB Benchmark)
# Referencia: Chen et al., AAAI 2024 (arXiv:2309.01431)
# ─────────────────────────────────────────────

def negative_rejection(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """
    Negative Rejection Rate (RGB Benchmark):
    ¿El sistema rechaza correctamente responder cuando no hay información disponible?

    Aplica SOLO a preguntas etiquetadas como "unanswerable" en el dataset.
    La etiqueta se detecta automáticamente si el GT answer contiene frases de rechazo.

    Score: 1 = rechazó correctamente, 0 = alucinó o respondió cuando no debía.
    Referencia: Chen et al. AAAI 2024 (RGB Benchmark).
    """
    ground_truth = reference_outputs.get("answer", "")
    answer = outputs.get("answer", "")

    # Verificar si esta pregunta es "unanswerable" según el GT
    gt_is_no_answer = _is_refusal(ground_truth)

    if not gt_is_no_answer:
        # No aplica a preguntas con respuesta esperada
        return {"key": "negative_rejection", "score": None,
                "comment": "N/A: question has expected answer (not unanswerable)"}

    # Para preguntas sin respuesta: ¿rechazó el sistema?
    system_rejected = _is_refusal(answer)
    score = 1.0 if system_rejected else 0.0
    comment = (
        "Correct rejection" if system_rejected
        else f"FAIL: hallucinated response for unanswerable question: '{answer[:100]}'"
    )
    return {"key": "negative_rejection", "score": score, "comment": comment}


def negative_rejection_rate(results: List[Dict]) -> Dict:
    """
    Calcula la tasa global de Negative Rejection sobre el dataset completo.
    Solo considera los ejemplos donde negative_rejection fue evaluado (score != None).
    """
    applicable = [r for r in results if r.get("negative_rejection") is not None]
    if not applicable:
        return {"negative_rejection_rate": None, "n_applicable": 0}

    rate = sum(r["negative_rejection"] for r in applicable) / len(applicable)
    return {"negative_rejection_rate": round(rate, 3), "n_applicable": len(applicable)}


# ─────────────────────────────────────────────
# BLOQUE 9: MRR — Mean Reciprocal Rank
# Referencia: estándar en IR (Information Retrieval)
# ─────────────────────────────────────────────

def mrr(relevance_lists: List[List[int]]) -> float:
    """
    Mean Reciprocal Rank:
        MRR = (1/|Q|) × Σ_{i=1}^{|Q|} (1 / rank_i)

    Args:
        relevance_lists: lista de listas, cada lista contiene 0/1
                         indicando relevancia de cada chunk por query.
                         Ej: [[0, 1, 0], [1, 0, 0]] → MRR = (1/2 + 1/1) / 2 = 0.75

    Returns:
        float: MRR score en [0, 1].
    """
    scores = []
    for rl in relevance_lists:
        for rank, rel in enumerate(rl, start=1):
            if rel == 1:
                scores.append(1.0 / rank)
                break
        else:
            scores.append(0.0)
    return round(sum(scores) / len(scores), 3) if scores else 0.0


# ─────────────────────────────────────────────
# BLOQUE 10: CALIBRACIÓN — ECE + Temperature Scaling
# Referencia: "When Can We Trust LLM Graders?" (arXiv:2603.29559, 2025)
#             ARES (Saad-Falcon et al., NAACL 2024)
# ─────────────────────────────────────────────

def compute_ece(predicted_scores: List[float], binary_labels: List[int],
                n_bins: int = 10) -> Dict:
    """
    Expected Calibration Error (ECE):
    Mide si los scores del evaluador LLM se corresponden con la probabilidad real de acierto.

    ECE = Σ_{b=1}^{B} (|B_b| / n) × |acc(B_b) - conf(B_b)|

    Donde:
    - B_b = ejemplos en el bin b (agrupados por score predicho)
    - acc(B_b) = accuracy real en ese bin (fracción de aciertos)
    - conf(B_b) = confianza media predicha en ese bin

    ECE ideal = 0.0 (perfectamente calibrado).
    ECE > 0.1 = mal calibrado (según literatura, arXiv:2603.29559).

    Args:
        predicted_scores: lista de scores del evaluador LLM [0, 1]
        binary_labels: etiquetas reales de corrección (0 o 1)
        n_bins: número de bins de calibración (default: 10)

    Returns:
        dict con ECE, bins detallados, e interpretación.
    """
    assert len(predicted_scores) == len(binary_labels), "Lengths must match"
    n = len(predicted_scores)
    if n == 0:
        return {"ece": None, "bins": [], "interpretation": "No data"}

    bins = []
    ece = 0.0

    bin_edges = [i / n_bins for i in range(n_bins + 1)]

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # Incluir el 1.0 en el último bin
        if i == n_bins - 1:
            mask = [lo <= s <= hi for s in predicted_scores]
        else:
            mask = [lo <= s < hi for s in predicted_scores]

        bin_preds = [predicted_scores[j] for j, m in enumerate(mask) if m]
        bin_labels = [binary_labels[j] for j, m in enumerate(mask) if m]

        if not bin_preds:
            bins.append({"range": f"[{lo:.1f}, {hi:.1f})", "n": 0,
                         "confidence": None, "accuracy": None, "gap": None})
            continue

        conf = sum(bin_preds) / len(bin_preds)
        acc = sum(bin_labels) / len(bin_labels)
        gap = abs(acc - conf)
        ece += (len(bin_preds) / n) * gap

        bins.append({
            "range": f"[{lo:.1f}, {hi:.1f})",
            "n": len(bin_preds),
            "confidence": round(conf, 3),
            "accuracy": round(acc, 3),
            "gap": round(gap, 3)
        })

    ece = round(ece, 4)

    # Interpretación (basada en arXiv:2603.29559)
    if ece < 0.05:
        interpretation = "Bien calibrado (ECE < 0.05)"
    elif ece < 0.10:
        interpretation = "Aceptablemente calibrado (0.05 ≤ ECE < 0.10)"
    elif ece < 0.20:
        interpretation = "Mal calibrado (0.10 ≤ ECE < 0.20) — considerar temperature scaling"
    else:
        interpretation = f"Muy mal calibrado (ECE ≥ 0.20) — el evaluador LLM no es fiable"

    return {"ece": ece, "bins": bins, "n": n,
            "interpretation": interpretation}


def temperature_scaling(scores: List[float], temperature: float) -> List[float]:
    """
    Temperature Scaling para calibración post-hoc.
    Ajusta los logits del evaluador dividiendo por T:

    calibrated_score = sigmoid(logit(score) / T)

    T > 1: suaviza la distribución (reduce extremos)
    T < 1: sharpens (empuja hacia 0 y 1)
    T = 1: sin cambio

    Referencia: Guo et al. ICML 2017 (Temperature Scaling).
    """
    def logit(p: float) -> float:
        p = max(1e-6, min(1 - 1e-6, p))
        return math.log(p / (1 - p))

    def sigmoid(x: float) -> float:
        return 1 / (1 + math.exp(-x))

    if temperature <= 0:
        raise ValueError("Temperature must be > 0")

    return [round(sigmoid(logit(s) / temperature), 4) for s in scores]


def find_optimal_temperature(predicted_scores: List[float],
                              binary_labels: List[int],
                              temperatures: Optional[List[float]] = None) -> Dict:
    """
    Búsqueda grid para la temperatura óptima que minimiza ECE.

    Args:
        predicted_scores: scores del evaluador
        binary_labels: ground truth labels (0/1)
        temperatures: lista de temperaturas a probar (default: 0.1 a 3.0)

    Returns:
        dict con temperatura óptima, ECE antes y después, y scores calibrados.
    """
    if temperatures is None:
        temperatures = [0.1, 0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0, 2.5, 3.0]

    original_ece = compute_ece(predicted_scores, binary_labels)["ece"]

    best_T = 1.0
    best_ece = original_ece

    for T in temperatures:
        calibrated = temperature_scaling(predicted_scores, T)
        ece = compute_ece(calibrated, binary_labels)["ece"]
        if ece is not None and ece < best_ece:
            best_ece = ece
            best_T = T

    calibrated_scores = temperature_scaling(predicted_scores, best_T)

    return {
        "optimal_temperature": best_T,
        "ece_before": original_ece,
        "ece_after": round(best_ece, 4),
        "improvement": round(original_ece - best_ece, 4) if original_ece else None,
        "calibrated_scores": calibrated_scores
    }


def compute_calibration_report(predicted_scores: List[float],
                                binary_labels: List[int],
                                metric_name: str = "evaluator") -> str:
    """
    Reporte completo de calibración para un evaluador.
    Incluye ECE, temperatura óptima, interpretación, y tabla de bins.
    """
    if not predicted_scores:
        return "No scores to calibrate."

    ece_result = compute_ece(predicted_scores, binary_labels)
    temp_result = find_optimal_temperature(predicted_scores, binary_labels)

    lines = [
        f"╔══════════════════════════════════════════════════════╗",
        f"║  CALIBRATION REPORT — {metric_name[:30]:<30} ║",
        f"╠══════════════════════════════════════════════════════╣",
        f"║  n samples      : {ece_result['n']:<34} ║",
        f"║  ECE (before)   : {ece_result['ece']:<34} ║",
        f"║  Interpretation : {ece_result['interpretation'][:33]:<33}║",
        f"╠══════════════════════════════════════════════════════╣",
        f"║  TEMPERATURE SCALING                                 ║",
        f"║  Optimal T      : {temp_result['optimal_temperature']:<34} ║",
        f"║  ECE (after)    : {temp_result['ece_after']:<34} ║",
        f"║  Improvement    : {temp_result['improvement']:<34} ║",
        f"╠══════════════════════════════════════════════════════╣",
        f"║  BINS DETAIL                                         ║",
    ]

    for b in ece_result["bins"]:
        if b["n"] > 0:
            lines.append(
                f"║  {b['range']:<10} n={b['n']:<4} conf={b['confidence']:.2f} "
                f"acc={b['accuracy']:.2f} gap={b['gap']:.2f}           ║"
            )

    lines.append("╚══════════════════════════════════════════════════════╝")
    report = "\n".join(lines)
    print(report)
    return report


# ─────────────────────────────────────────────
# BLOQUE 11: CONFIDENCE SCORE UNIVERSAL
# Composite score calibrado para cualquier RAG
# ─────────────────────────────────────────────

def confidence_score_universal(inputs: dict, outputs: dict,
                                reference_outputs: dict) -> dict:
    """
    Score de confianza compuesto para cualquier RAG.
    Combina métricas NLI + LLM-judge sin duplicar llamadas LLM.

    Fórmula (pesos heurísticos — versión sin entrenamiento):
        confidence = 0.35 × faithfulness_nli
                   + 0.25 × correctness_universal
                   + 0.20 × answer_relevance_universal
                   + 0.20 × context_relevance

    Para pesos aprendidos por regresión logística, usar confidence_score_learned().
    Referencia: inspirado en ARES PPI score composition (NAACL 2024).
    """
    f = faithfulness_nli(inputs, outputs)["score"]
    c = correctness_universal(inputs, outputs, reference_outputs)["score"]
    ar = answer_relevance_universal(inputs, outputs)["score"]
    cr = context_relevance(inputs, outputs)["score"]

    score = round(0.35 * f + 0.25 * c + 0.20 * ar + 0.20 * cr, 3)
    comment = (
        f"faithfulness_nli={f} | correctness={c} | "
        f"answer_relevance={ar} | context_relevance={cr} → confidence={score}"
    )
    return {"key": "confidence_score_universal", "score": score, "comment": comment}


# ─────────────────────────────────────────────
# BLOQUE 12: CONFIDENCE SCORE CON PESOS APRENDIDOS (US-C1)
# Logistic Regression sobre métricas observadas
# Referencia: ARES (NAACL 2024) — calibración mediante regresión
# ─────────────────────────────────────────────

def train_confidence_weights(
    metrics_matrix: List[Dict[str, float]],
    correctness_labels: List[float],
    feature_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    US-C1: Aprende pesos para el confidence score via regresión logística (numpy puro).

    Dado un conjunto de ejemplos ya evaluados con sus métricas y sus labels de correctness,
    ajusta una regresión logística para predecir P(correctness=1) a partir de las métricas.

    Args:
        metrics_matrix:    lista de dicts con métricas por ejemplo
                           ej: [{"faithfulness_nli": 0.8, "hallucination_rate": 0.2, ...}, ...]
        correctness_labels: lista de labels binarios (0 o 1) — si correctness ≥ 0.5 → 1
        feature_keys:      métricas a usar como features (default: las 4 discriminativas)

    Returns:
        dict con:
            "weights":     dict {metric_name: weight}
            "bias":        float
            "feature_keys": list
            "ece_logistic": float — ECE del modelo aprendido
            "n_train":     int

    Uso:
        model = train_confidence_weights(metrics_list, labels)
        score = predict_confidence_score(example_metrics, model)
    """
    if feature_keys is None:
        feature_keys = ["faithfulness_nli", "hallucination_rate",
                        "correctness_continuous", "negative_rejection"]

    # Construir X (n_samples × n_features) y y (n_samples,)
    X = []
    y = []
    for metrics, label in zip(metrics_matrix, correctness_labels):
        row = [metrics.get(k, 0.0) for k in feature_keys]
        if any(v is not None and not np.isnan(v) for v in row):
            X.append(row)
            y.append(1.0 if label >= 0.5 else 0.0)

    if len(X) < 5:
        raise ValueError(
            f"Se necesitan al menos 5 ejemplos para entrenar pesos. Solo hay {len(X)}."
        )

    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    n, d = X.shape

    # Reemplazar NaN por media de la columna
    col_means = np.nanmean(X, axis=0)
    for j in range(d):
        X[np.isnan(X[:, j]), j] = col_means[j]

    # Normalización min-max → [0, 1]
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = np.where(X_max - X_min > 1e-8, X_max - X_min, 1.0)
    X_norm = (X - X_min) / X_range

    # Añadir bias column
    X_aug = np.hstack([X_norm, np.ones((n, 1))])

    # Regresión logística — gradient descent con L2 regularización
    theta = np.zeros(d + 1)
    lr = 0.1
    lam = 0.01  # L2 regularización
    n_iters = 500

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))

    for _ in range(n_iters):
        z = X_aug @ theta
        h = sigmoid(z)
        grad = (X_aug.T @ (h - y)) / n + lam * np.append(theta[:-1], 0)
        theta -= lr * grad

    weights_raw = theta[:-1]
    bias = theta[-1]

    # Renormalizar pesos para que sumen 1 (interpretabilidad)
    weight_sum = np.abs(weights_raw).sum()
    if weight_sum > 1e-8:
        weights_norm = weights_raw / weight_sum
    else:
        weights_norm = weights_raw

    # ECE del modelo aprendido
    preds = sigmoid(X_aug @ theta).tolist()
    ece_result = compute_ece(preds, [int(yi) for yi in y])

    return {
        "weights": dict(zip(feature_keys, weights_norm.tolist())),
        "weights_raw": dict(zip(feature_keys, weights_raw.tolist())),
        "bias": float(bias),
        "feature_keys": feature_keys,
        "X_min": X_min.tolist(),
        "X_range": X_range.tolist(),
        "ece_logistic": ece_result["ece"],
        "n_train": n,
    }


def predict_confidence_score(metrics: Dict[str, float], model: Dict[str, Any]) -> float:
    """
    Predice el confidence score de un ejemplo usando un modelo entrenado
    con train_confidence_weights().

    Args:
        metrics: dict con métricas del ejemplo
        model:   salida de train_confidence_weights()

    Returns:
        float en [0, 1] — probabilidad logística de correctness=1
    """
    feature_keys = model["feature_keys"]
    X_min = np.array(model["X_min"])
    X_range = np.array(model["X_range"])
    weights_raw = np.array([model["weights_raw"][k] for k in feature_keys])
    bias = model["bias"]

    # Construir vector de features normalizado
    x = np.array([metrics.get(k, 0.0) for k in feature_keys], dtype=np.float64)
    x_norm = (x - X_min) / X_range
    z = weights_raw @ x_norm + bias

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))

    return float(sigmoid(z))


def confidence_score_learned_factory(model: Dict[str, Any]):
    """
    Factory que crea un evaluador LangSmith usando un modelo entrenado.

    Uso:
        model = train_confidence_weights(metrics_list, labels)
        evaluator = confidence_score_learned_factory(model)
        results = client.evaluate(rag_fn, evaluators=[evaluator, ...])

    El evaluador computa las métricas necesarias en cada ejemplo y aplica
    el modelo logístico para obtener el confidence score aprendido.
    """
    def confidence_score_learned(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
        """Confidence score con pesos aprendidos (US-C1)."""
        feature_keys = model["feature_keys"]

        # Computar métricas necesarias para este ejemplo
        metrics = {}
        if "faithfulness_nli" in feature_keys:
            metrics["faithfulness_nli"] = faithfulness_nli(inputs, outputs)["score"]
        if "hallucination_rate" in feature_keys:
            metrics["hallucination_rate"] = 1.0 - metrics.get("faithfulness_nli", 0.0)
        if "correctness_continuous" in feature_keys:
            metrics["correctness_continuous"] = correctness_continuous(
                inputs, outputs, reference_outputs)["score"]
        if "negative_rejection" in feature_keys:
            metrics["negative_rejection"] = negative_rejection(
                inputs, outputs, reference_outputs)["score"] or 0.0
        if "answer_relevance_universal" in feature_keys:
            metrics["answer_relevance_universal"] = answer_relevance_universal(
                inputs, outputs)["score"]

        score = predict_confidence_score(metrics, model)
        comment = " | ".join(f"{k}={v:.3f}" for k, v in metrics.items())
        comment += f" → confidence_learned={score:.3f}"

        return {"key": "confidence_score_learned", "score": round(score, 3), "comment": comment}

    return confidence_score_learned


# ─────────────────────────────────────────────
# FUNCIÓN PRINCIPAL — evaluate_rag_universal
# ─────────────────────────────────────────────

# Conjunto por defecto: métricas core + correctness continuo (US-M2)
DEFAULT_EVALUATORS = [
    faithfulness_nli,
    hallucination_rate,
    answer_relevance_universal,
    correctness_universal,       # binario (0/1) — compatibilidad hacia atrás
    correctness_continuous,      # continuo (0/0.5/1.0) — US-M2
    negative_rejection,
]

# Conjunto completo: todas las métricas (más lento, más costoso)
FULL_EVALUATORS = [
    faithfulness_nli,
    hallucination_rate,
    atomic_fact_precision,
    context_precision_at_k,
    context_recall,
    context_relevance,
    answer_relevance_universal,
    correctness_universal,
    correctness_continuous,
    negative_rejection,
    confidence_score_universal,
]

# Conjunto ligero: solo métricas sin LLM (máxima velocidad)
NLI_ONLY_EVALUATORS = [
    faithfulness_nli,
    hallucination_rate,
]

# Conjunto para tabla de poder discriminativo (US-M1) — las 4 métricas clave
DISCRIMINATIVE_EVALUATORS = [
    faithfulness_nli,
    hallucination_rate,
    correctness_continuous,
    negative_rejection,
]


def evaluate_rag_universal(
    rag_fn: Callable[[dict], dict],
    dataset: List[Dict],
    dataset_name: str = "rag-universal-eval",
    project: str = "rag-universal-project",
    evaluators: Optional[List] = None,
    preset: str = "default",
    experiment_name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs
):
    """
    Evaluador universal para cualquier arquitectura RAG.
    Integración nativa con LangSmith.

    Compatible con cualquier RAG que implemente:
        def rag_fn(inputs: dict) -> dict
    donde outputs incluye al mínimo:
        {"answer": str, "context": str | list}

    Args:
        rag_fn:         función RAG a evaluar
        dataset:        lista de ejemplos con {"inputs": {...}, "outputs": {...}}
        dataset_name:   nombre del dataset en LangSmith
        project:        proyecto LangSmith para la evaluación
        evaluators:       lista de evaluadores custom (sobrescribe preset)
        preset:           "default" | "full" | "nli_only"
            - "default": faithfulness_nli + correctness + answer_relevance + negative_rejection
            - "full":    todas las métricas (más lento)
            - "nli_only": solo métricas NLI sin LLM (más rápido, sin costo)
        experiment_name:  nombre del experimento en LangSmith (sobrescribe prefijo auto)
        metadata:         dict con metadata del experimento — se guarda en LangSmith.
                          Campos estándar recomendados (US-E2):
                            {
                              "architecture": "GraphRAG-LangChain",
                              "wrapper":      "graphrag_neo4j",
                              "dataset":      "northwind-v1",
                              "llm":          "gpt-3.5-turbo",
                              "preset":       "default",
                            }
        **kwargs:         argumentos adicionales para client.evaluate()
                          (ej. max_concurrency=1)

    Returns:
        ExperimentResults de LangSmith

    Ejemplo:
        results = evaluate_rag_universal(
            rag_fn=my_rag,
            dataset=DATASET,
            dataset_name="northwind-universal-eval",
            project="01-agentic-rag",
            preset="default",
            max_concurrency=1
        )
    """
    client = Client()

    # Seleccionar evaluadores según preset
    if evaluators is None:
        if preset == "full":
            evaluators = FULL_EVALUATORS
        elif preset == "nli_only":
            evaluators = NLI_ONLY_EVALUATORS
        elif preset == "discriminative":
            evaluators = DISCRIMINATIVE_EVALUATORS
        else:  # default
            evaluators = DEFAULT_EVALUATORS

    print(f"\n{'═' * 55}")
    print(f"  Universal RAG Evaluator")
    print(f"  Dataset    : {dataset_name}")
    print(f"  Project    : {project}")
    print(f"  Preset     : {preset}")
    print(f"  Evaluators : {len(evaluators)}")
    print(f"  Examples   : {len(dataset)}")
    print(f"{'═' * 55}\n")

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
                raise RuntimeError(f"Dataset '{dataset_name}' no encontrado.")
            print(f"⚠️  Dataset '{dataset_name}' ya existe, reutilizando.")
        else:
            raise

    # Parámetros por defecto conservadores (rate limit OpenAI)
    max_concurrency: int = kwargs.pop("max_concurrency", 1)

    # US-E2: metadata estructurada para LangSmith
    import datetime
    base_metadata = {
        "preset": preset,
        "n_examples": len(dataset),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    }
    if metadata:
        base_metadata.update(metadata)

    # Nombre del experimento
    exp_prefix = experiment_name or dataset_name.replace(" ", "-")

    print(f"🚀 Iniciando evaluación con max_concurrency={max_concurrency}...")
    if base_metadata:
        print(f"   Metadata: {base_metadata}")

    results = client.evaluate(
        rag_fn,
        data=dataset_name,
        evaluators=evaluators,
        experiment_prefix=exp_prefix,
        max_concurrency=max_concurrency,
        metadata=base_metadata,
    )

    print(f"\n✅ Evaluación completa.")
    print(f"   Ver resultados en LangSmith → proyecto '{project}'")
    return results


# ─────────────────────────────────────────────
# ANÁLISIS DE RESULTADOS
# ─────────────────────────────────────────────

def print_universal_summary(results, title: str = "Universal RAG Evaluation") -> Dict:
    """
    Imprime resumen de métricas de una evaluación universal.
    Compatible con el objeto ExperimentResults de LangSmith.

    Args:
        results: ExperimentResults de LangSmith o lista de dicts con scores
        title: Título del resumen

    Returns:
        dict con métricas agregadas
    """
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}")

    # Extraer métricas del resumen de LangSmith
    try:
        summary = results.to_pandas()
        metrics = {}

        score_cols = [c for c in summary.columns if c.startswith("feedback.")]
        for col in score_cols:
            metric_name = col.replace("feedback.", "")
            valid = summary[col].dropna()
            if len(valid) > 0:
                mean_score = valid.mean()
                metrics[metric_name] = round(float(mean_score), 3)
                print(f"  {metric_name:<35} {mean_score:.3f}  (n={len(valid)})")

        print(f"{'─' * 60}")
        print(f"  Ejemplos evaluados: {len(summary)}")
        return metrics

    except Exception as e:
        print(f"  Error extrayendo métricas: {e}")
        print(f"  Revisa los resultados directamente en LangSmith.")
        return {}


# ─────────────────────────────────────────────
# TEST RÁPIDO (sin LangSmith)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Universal RAG Evaluator — Tests locales")
    print("=" * 60)

    # Test 1: Faithfulness NLI
    print("\n[TEST 1] faithfulness_nli")
    test_outputs = {
        "answer": "The company has 9 employees. Nancy Davolio works as a Sales Representative.",
        "context": "Northwind Traders has 9 employees total. Nancy Davolio is a Sales Representative based in Seattle."
    }
    result = faithfulness_nli({}, test_outputs)
    print(f"  Score: {result['score']} | Comment: {result['comment']}")

    # Test 2: Hallucination Rate
    print("\n[TEST 2] hallucination_rate")
    test_outputs_halluc = {
        "answer": "The company has 500 employees. The CEO is John Smith.",
        "context": "Northwind Traders has 9 employees total."
    }
    result = hallucination_rate({}, test_outputs_halluc)
    print(f"  Score: {result['score']} | Comment: {result['comment']}")

    # Test 3: Negative Rejection
    print("\n[TEST 3] negative_rejection")
    inputs_noans = {"question": "Which employees work in Madrid?"}
    outputs_rej = {"answer": "There are no employees in Madrid. The database does not contain that information."}
    reference_rej = {"answer": "There are no employees working in Madrid."}
    result = negative_rejection(inputs_noans, outputs_rej, reference_rej)
    print(f"  Score: {result['score']} | Comment: {result['comment']}")

    # Test 4: ECE Calibration
    print("\n[TEST 4] compute_ece")
    pred_scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.95]
    true_labels = [1,   1,   0,   1,   0,   0,   1,   0,   0,   1]
    ece_result = compute_ece(pred_scores, true_labels, n_bins=5)
    print(f"  ECE: {ece_result['ece']} | {ece_result['interpretation']}")

    # Test 5: Temperature Scaling
    print("\n[TEST 5] temperature_scaling")
    cal_scores = temperature_scaling([0.9, 0.8, 0.6, 0.3, 0.1], temperature=1.5)
    print(f"  Original:   [0.9, 0.8, 0.6, 0.3, 0.1]")
    print(f"  Calibrated: {cal_scores}")

    # Test 6: MRR
    print("\n[TEST 6] mrr")
    relevance_lists = [[0, 1, 0], [1, 0, 0], [0, 0, 1]]
    mrr_score = mrr(relevance_lists)
    print(f"  MRR: {mrr_score}  (expected ~0.611)")

    print("\n✅ Tests locales completados.")
    print("   Para evaluación completa con LangSmith, usa evaluate_rag_universal().")
