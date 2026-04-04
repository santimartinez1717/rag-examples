# Evaluadores Avanzados para GraphRAG (Neo4j + LangChain + LangSmith)

> Investigación completa — 2026-03-15
> Objetivo: superar los 4 evaluadores binarios actuales en `rag_evaluator.py`

---

## Resumen ejecutivo

Los evaluadores actuales (correctness, relevance, groundedness, retrieval_relevance) son todos **binarios (0/1) y LLM-as-judge genérico**. Para un TFG sobre confianza en RAG, hay 4 dimensiones de mejora:

1. **Continuidad**: pasar de binario a scores continuos [0,1]
2. **Groundedness con NLI**: reemplazar LLM-judge por DeBERTa (determinista, gratuito, rápido)
3. **Métricas RAGAS**: faithfulness, answer relevancy, context precision/recall con fórmulas matemáticas
4. **Métricas GraphRAG-específicas**: validación Cypher, faithfulness de grafo, cobertura de entidades

---

## Parte 1 — RAGAS: Métricas con base matemática

### 1.1 Faithfulness (anti-alucinación)

**Qué mide**: ¿Cuántas afirmaciones de la respuesta están respaldadas por el contexto recuperado?

**Algoritmo**:
1. LLM descompone la respuesta en claims atómicos (sin pronombres, self-contained)
2. Para cada claim: ¿el contexto lo entails? (sí/no via LLM)
3. Score = claims_soportados / total_claims

**Fórmula**:
```
Faithfulness = |{s ∈ statements : context ⊨ s}| / |statements|
```

**Integración LangSmith**:
```python
from ragas.metrics import faithfulness
from ragas import evaluate as ragas_evaluate
from datasets import Dataset

def ragas_faithfulness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    ds = Dataset.from_dict({
        "question": [inputs["question"]],
        "answer": [outputs["answer"]],
        "contexts": [outputs.get("contexts", [outputs.get("context", "")])],
        "ground_truth": [reference_outputs.get("answer", "")],
    })
    result = ragas_evaluate(ds, metrics=[faithfulness])
    return {"key": "ragas_faithfulness", "score": result["faithfulness"]}
```

---

### 1.2 Answer Relevancy (relevancia por embeddings)

**Qué mide**: ¿Cuán pertinente es la respuesta a la pregunta? Penaliza respuestas incompletas o fuera de tema.

**Algoritmo**:
1. LLM genera `n=3` preguntas hipotéticas a partir de la respuesta sola
2. Cosine similarity entre embedding de la pregunta original y cada pregunta generada
3. Score = media de similarities

**Fórmula**:
```
AnswerRelevancy = (1/n) Σᵢ cos(E(gᵢ), E(q))
```
donde `gᵢ` = pregunta generada i, `q` = pregunta original, `E(·)` = embedding

> ⚠️ Esta métrica NO mide corrección factual, solo alineación temática.

---

### 1.3 Context Precision

**Qué mide**: Señal-a-ruido del contexto recuperado. ¿Están los chunks relevantes rankeados más arriba?

**Fórmula**:
```
ContextPrecision@K = Σₖ [(Precisionₖ × vₖ)] / |relevant chunks in top K|
```
**Requiere**: `question`, `retrieved_contexts`, `ground_truth`

---

### 1.4 Context Recall

**Qué mide**: ¿Cuánto del ground truth puede atribuirse al contexto recuperado?

**Fórmula**:
```
ContextRecall = |{claims in ground_truth supported by context}| / |claims in ground_truth|
```

---

### 1.5 Context Entity Recall

**Qué mide**: Qué fracción de entidades nombradas del ground truth aparece en el contexto recuperado.

**Fórmula**:
```
ContextEntityRecall = |CE ∩ GE| / |GE|
```
CE = entidades en contexto, GE = entidades en ground truth

**Especialmente útil para GraphRAG**: mide si Neo4j devolvió los nodos correctos.

---

### 1.6 Noise Sensitivity

**Qué mide**: ¿Con qué frecuencia el sistema produce claims incorrectos cuando hay docs irrelevantes en el contexto?

**Fórmula** (lower is better):
```
NoiseSensitivity = |incorrect claims in answer| / |total claims in answer|
```

---

### Instalación y uso RAGAS con LangSmith

```bash
pip install ragas datasets sentence-transformers
```

```python
# Método recomendado: RAGAS como custom evaluator de LangSmith
from langsmith import Client
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas import evaluate as ragas_evaluate
from datasets import Dataset

def make_ragas_evaluator(metric):
    def evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
        contexts = outputs.get("contexts", [])
        if not contexts:
            # Para GraphRAG: parsear el campo context
            raw = outputs.get("context", "")
            contexts = [raw] if raw else [""]

        ds = Dataset.from_dict({
            "question": [inputs["question"]],
            "answer": [outputs["answer"]],
            "contexts": [contexts],
            "ground_truth": [reference_outputs.get("answer", "")],
        })
        result = ragas_evaluate(ds, metrics=[metric])
        return {"key": f"ragas_{metric.name}", "score": float(result[metric.name])}
    evaluator.__name__ = f"ragas_{metric.name}"
    return evaluator
```

---

## Parte 2 — NLI Groundedness con DeBERTa

### Por qué DeBERTa en lugar de GPT-4 para groundedness

| | GPT-4 judge | DeBERTa NLI |
|---|---|---|
| Costo | ~$0.01/eval | Gratis |
| Velocidad | ~2s/eval | ~50ms/eval |
| Determinismo | No (temperatura) | Sí |
| Precisión | Alta | Media-alta |
| Requiere internet | Sí | No (local) |

**Modelo**: `cross-encoder/nli-deberta-v3-base` (HuggingFace)
- Input: (premise=contexto, hypothesis=claim)
- Output: [contradiction, entailment, neutral] probabilities

### Implementación completa

```python
from sentence_transformers import CrossEncoder, SentenceTransformer, util
import re

# Cargar una sola vez
nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-base")
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # para selección de chunks

def extract_atomic_claims(answer: str) -> list[str]:
    """Split en frases. Para producción, usar LLM para descomponer claims complejos."""
    sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
    return [s for s in sentences if len(s) > 10]

def nli_groundedness_score(answer: str, context_chunks: list[str], top_k: int = 3) -> dict:
    """
    Groundedness continuo [0,1] usando NLI entailment.
    Para cada claim, busca los chunks más relevantes y verifica entailment.
    """
    claims = extract_atomic_claims(answer)
    if not claims:
        return {"score": 1.0, "claims": []}

    chunk_embeddings = embedder.encode(context_chunks, convert_to_tensor=True)

    claim_results = []
    for claim in claims:
        # Encontrar chunks más relevantes para este claim
        claim_emb = embedder.encode(claim, convert_to_tensor=True)
        cos_scores = util.cos_sim(claim_emb, chunk_embeddings)[0]
        top_idx = cos_scores.topk(min(top_k, len(context_chunks))).indices.tolist()
        relevant_context = " ".join([context_chunks[i] for i in top_idx])

        # NLI: ¿el contexto entails el claim?
        score = nli_model.predict([(relevant_context, claim)], apply_softmax=True)[0]
        # score = [contradiction_p, entailment_p, neutral_p]
        claim_results.append({
            "claim": claim,
            "entailment": float(score[1]),
            "supported": bool(score[1] > 0.5),
        })

    n_supported = sum(c["supported"] for c in claim_results)
    return {
        "score": n_supported / len(claims),
        "n_claims": len(claims),
        "n_supported": n_supported,
        "claims": claim_results,
    }

# Wrapper para LangSmith
def deberta_groundedness_evaluator(inputs: dict, outputs: dict) -> dict:
    answer = outputs.get("answer", "")
    # Para GraphRAG: extraer contexto del campo context formateado
    context = outputs.get("context", "")
    chunks = [context] if context else [answer]

    result = nli_groundedness_score(answer, chunks)
    return {
        "key": "nli_groundedness",
        "score": result["score"],
        "comment": f"{result['n_supported']}/{result['n_claims']} claims entailed",
    }
```

**Umbrales orientativos**: ≥ 0.9 = bien fundamentado; 0.7–0.9 = parcialmente; < 0.7 = riesgo de alucinación.

---

## Parte 3 — Scoring continuo con rúbricas

### Por qué escala de 4 puntos

Investigación de Databricks (2024): las escalas de 4-5 puntos son óptimas. Escalas más finas (1-10) generan inconsistencia en el juez.

```python
RUBRIC_PROMPT = """Eres un evaluador experto de sistemas RAG.

Puntúa la siguiente respuesta en escala 0.0-1.0 según esta rúbrica:

1.0 — Completamente correcta, completa, sin claims sin soporte
0.75 — Mayormente correcta, omisiones menores o un detalle sin soporte
0.5 — Parcialmente correcta; partes correctas y partes incorrectas o poco claras
0.25 — Mayormente incorrecta o incompleta, relevancia superficial
0.0 — Completamente incorrecta, irrelevante, o contradice el contexto

Pregunta: {question}
Contexto: {context}
Respuesta: {answer}
Respuesta de referencia: {reference}

Responde SOLO con JSON: {{"score": <float>, "reasoning": "<una frase>"}}"""

import json
from langchain_openai import ChatOpenAI

_rubric_llm = ChatOpenAI(model="gpt-4.1", temperature=0)

def rubric_correctness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    prompt = RUBRIC_PROMPT.format(
        question=inputs["question"],
        context=outputs.get("context", ""),
        answer=outputs.get("answer", ""),
        reference=reference_outputs.get("answer", "N/A"),
    )
    response = _rubric_llm.invoke(prompt).content
    try:
        parsed = json.loads(response)
        return {"key": "rubric_correctness", "score": parsed["score"], "comment": parsed["reasoning"]}
    except:
        return {"key": "rubric_correctness", "score": 0.0, "comment": "parse error"}
```

---

## Parte 4 — Evaluadores GraphRAG-específicos (Cypher)

### Helpers de parsing (compatibles con wrapper actual)

El `outputs["context"]` del wrapper actual tiene el formato:
```
"Cypher Query: MATCH (n:Employee) RETURN n\n\nDatabase Results: [{'n': {...}}]"
```

```python
import ast, re

def extract_cypher(context: str) -> str:
    if "Cypher Query:" in context:
        return context.split("Cypher Query:")[1].split("\n\nDatabase Results:")[0].strip()
    return ""

def extract_db_results(context: str) -> list:
    if "Database Results:" in context:
        raw = context.split("Database Results:")[1].strip()
        try:
            return ast.literal_eval(raw)
        except:
            return []
    return []
```

---

### 4.1 Cypher generado (binario — ¿se generó algo?)

```python
def cypher_generated(inputs: dict, outputs: dict) -> dict:
    cypher = extract_cypher(outputs.get("context", ""))
    has_query = bool(cypher) and any(
        kw in cypher.upper() for kw in ["MATCH", "RETURN", "CALL"]
    )
    return {"key": "cypher_generated", "score": 1 if has_query else 0}
```

---

### 4.2 Cypher con resultados no vacíos

```python
def cypher_result_nonempty(inputs: dict, outputs: dict) -> dict:
    db_results = extract_db_results(outputs.get("context", ""))
    return {"key": "cypher_result_nonempty", "score": 1 if len(db_results) > 0 else 0}

def cypher_result_count(inputs: dict, outputs: dict) -> dict:
    db_results = extract_db_results(outputs.get("context", ""))
    return {"key": "cypher_result_count", "score": len(db_results)}
```

---

### 4.3 Validación sintáctica Cypher (sin BD)

```bash
pip install antlr4-cypher antlr4-python3-runtime
```

```python
from antlr4 import InputStream, CommonTokenStream
from antlr4_cypher import CypherLexer, CypherParser

def cypher_syntactic_valid(inputs: dict, outputs: dict) -> dict:
    cypher = extract_cypher(outputs.get("context", ""))
    if not cypher:
        return {"key": "cypher_syntactic_valid", "score": 0}

    errors = []
    class ErrorCollector:
        def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
            errors.append(f"L{line}:{column} {msg}")

    stream = InputStream(cypher)
    lexer = CypherLexer(stream)
    lexer.removeErrorListeners(); lexer.addErrorListener(ErrorCollector())
    tokens = CommonTokenStream(lexer)
    parser = CypherParser(tokens)
    parser.removeErrorListeners(); parser.addErrorListener(ErrorCollector())
    parser.oC_Cypher()
    return {
        "key": "cypher_syntactic_valid",
        "score": 1 if not errors else 0,
        "comment": "; ".join(errors) if errors else "valid"
    }
```

---

### 4.4 Complejidad del Cypher

```python
def cypher_complexity_metrics(inputs: dict, outputs: dict) -> list[dict]:
    cypher = extract_cypher(outputs.get("context", "")).upper()
    if not cypher:
        return [{"key": "cypher_match_count", "score": 0}]

    return [
        {"key": "cypher_match_count", "score": len(re.findall(r'\bMATCH\b', cypher))},
        {"key": "cypher_relationship_patterns", "score": len(re.findall(r'-\[|\]-', outputs.get("context", "")))},
        {"key": "cypher_has_where", "score": 1 if re.search(r'\bWHERE\b', cypher) else 0},
        {"key": "cypher_has_aggregation", "score": 1 if re.search(r'\b(COUNT|SUM|AVG|MAX|MIN|COLLECT)\b', cypher) else 0},
        {"key": "cypher_variable_length_paths", "score": len(re.findall(r'\*\d*\.\.', cypher))},
    ]
```

---

### 4.5 Graph Faithfulness (¿las relaciones del answer existen en el grafo?)

```python
import json
from neo4j import GraphDatabase

TRIPLE_EXTRACTION_PROMPT = """Extrae todos los triples factuales del texto como (sujeto, predicado, objeto).
Texto: {text}
Devuelve JSON: [{{"subject": "...", "predicate": "...", "object": "..."}}]
Si no hay triples, devuelve []."""

def extract_triples(text: str, llm) -> list[dict]:
    response = llm.invoke(TRIPLE_EXTRACTION_PROMPT.format(text=text)).content
    try:
        return json.loads(response)
    except:
        return []

def verify_triple_in_neo4j(triple: dict, driver) -> bool:
    query = """
    MATCH (s)-[r]->(o)
    WHERE toLower(s.name) CONTAINS toLower($subject)
      AND toLower(o.name) CONTAINS toLower($object)
    RETURN count(*) as cnt
    """
    with driver.session() as session:
        result = session.run(query, subject=triple["subject"], object=triple["object"])
        return result.single()["cnt"] > 0

def graph_faithfulness_evaluator(inputs: dict, outputs: dict) -> dict:
    from langchain_openai import ChatOpenAI
    from neo4j import GraphDatabase

    llm = ChatOpenAI(model="gpt-4.1", temperature=0)
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

    triples = extract_triples(outputs.get("answer", ""), llm)
    if not triples:
        return {"key": "graph_faithfulness", "score": 1.0, "comment": "no triples to verify"}

    verified = [verify_triple_in_neo4j(t, driver) for t in triples]
    score = sum(verified) / len(verified)
    driver.close()
    return {
        "key": "graph_faithfulness",
        "score": score,
        "comment": f"{sum(verified)}/{len(verified)} triples verified in graph"
    }
```

---

### 4.6 Entity Coverage (¿el answer menciona los nodos clave?)

```python
def entity_coverage_evaluator(inputs: dict, outputs: dict) -> dict:
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
    question = inputs["question"]
    answer = outputs.get("answer", "").lower()

    # Buscar entidades relevantes en el grafo por keywords de la pregunta
    keywords = [w for w in question.lower().split() if len(w) > 3]
    query = """
    MATCH (n) WHERE any(kw IN $keywords WHERE toLower(n.name) CONTAINS kw)
    RETURN n.name as name LIMIT 10
    """
    with driver.session() as session:
        results = session.run(query, keywords=keywords)
        relevant_entities = {r["name"].lower() for r in results if r["name"]}

    driver.close()
    if not relevant_entities:
        return {"key": "entity_coverage", "score": 1.0}

    mentioned = sum(1 for e in relevant_entities if e in answer)
    return {
        "key": "entity_coverage",
        "score": mentioned / len(relevant_entities),
        "comment": f"{mentioned}/{len(relevant_entities)} entities mentioned"
    }
```

---

## Parte 5 — LangSmith avanzado

### Acceso al objeto Run (latencia, metadata, tokens)

```python
from langsmith.schemas import Run

def latency_evaluator(run: Run, inputs: dict, outputs: dict) -> list[dict]:
    """Métricas de rendimiento desde el objeto Run de LangSmith."""
    return [
        {"key": "execution_latency_sec", "score": run.latency},
        {"key": "total_tokens", "score": run.total_tokens or 0},
        {"key": "total_cost_usd", "score": float(run.total_cost or 0)},
        {"key": "had_error", "score": 1 if run.error else 0},
    ]
```

### Enriquecer trazas con metadata GraphRAG

En `graphrag_wrapper_standalone.py`, añadir antes del return:

```python
import langsmith as ls

rt = ls.get_current_run_tree()
if rt:
    rt.metadata["cypher_query"] = cypher_query
    rt.metadata["db_result_count"] = len(db_results) if isinstance(db_results, list) else 0
    rt.metadata["cypher_generated"] = bool(cypher_query)
```

Esto hace que en LangSmith UI aparezcan estos campos como metadata en cada traza.

### A/B Testing: GraphRAG vs otro RAG

```python
# Paso 1: correr ambas arquitecturas contra el mismo dataset
results_graph = client.evaluate(
    neo4j_graphrag_wrapper_standalone,
    data="shared-eval-dataset",
    evaluators=[correctness, relevance, groundedness],
    experiment_prefix="graphrag-neo4j-v1",
)

results_vector = client.evaluate(
    vector_rag_wrapper,
    data="shared-eval-dataset",
    evaluators=[correctness, relevance, groundedness],
    experiment_prefix="vectorrag-v1",
)

# Paso 2: evaluación comparativa pairwise
def pairwise_evaluator(inputs: dict, outputs: list[dict], runs: list, **kwargs) -> dict:
    # outputs[0] = GraphRAG, outputs[1] = VectorRAG
    graphrag_ans = outputs[0].get("answer", "")
    vector_ans = outputs[1].get("answer", "")
    # LLM judge elige cuál es mejor...
    return {
        "key": "preferred_architecture",
        "scores": {
            runs[0].id: 1,  # actualizar según juicio
            runs[1].id: 0,
        }
    }

client.evaluate(
    target=(results_graph.experiment_name, results_vector.experiment_name),
    evaluators=[pairwise_evaluator],
)
```

### num_repetitions: medir varianza del RAG

```python
# Correr cada ejemplo 3 veces para medir consistencia
results = client.evaluate(
    rag_fn,
    data="eval-dataset",
    evaluators=[correctness, nli_groundedness],
    num_repetitions=3,
    experiment_prefix="graphrag-variance-test",
)
# En LangSmith UI verás la distribución de scores por ejemplo
```

---

## Parte 6 — Referencias académicas clave

| Paper | Contribución | Link |
|---|---|---|
| **GraphRAG-Bench** (arXiv:2506.02404, ICLR 2026) | Benchmark 1018 preguntas, 5 tipos, 16 temas. Evalúa HippoRAG 2, LightRAG, Microsoft GraphRAG | https://arxiv.org/abs/2506.02404 |
| **KG-Based RAG Evaluation** (arXiv:2510.02549) | Extiende RAGAS al paradigma KG. Multi-hop semantic matching + community overlap | https://arxiv.org/abs/2510.02549 |
| **GraphRAG Survey** (arXiv:2408.08921, ACM TOIS) | Las métricas RAG estándar son insuficientes para grafos — no capturan integridad estructural | https://arxiv.org/abs/2408.08921 |
| **When to use Graphs in RAG** (arXiv:2506.05690) | GraphRAG supera RAG estándar en multi-hop; propone routing por tipo de pregunta | https://arxiv.org/abs/2506.05690 |
| **Case-Aware LLM-as-a-Judge** (arXiv:2602.20379) | Score multidimensional: w1·Faithfulness + w2·Relevance + ... con severity bands | https://arxiv.org/html/2602.20379 |
| **RAGAS** (ACL EACL 2024) | Framework de referencia, fórmulas de faithfulness/relevancy/precision/recall | https://arxiv.org/abs/2309.15217 |

### Benchmark dataset recomendado

**neo4j/text2cypher-2024v1** (HuggingFace): 44k pares NL→Cypher con schema.
Campos: `question`, `cypher`, `schema` — directamente usable para evaluar tu GraphCypherQAChain.

---

## Plan de implementación recomendado

### Fase A — Inmediata (sin nuevas dependencias)
- [ ] Reemplazar evaluadores binarios por versiones con rúbrica continua (rubric_correctness)
- [ ] Añadir `cypher_generated`, `cypher_result_nonempty`, `cypher_result_count` — ya tienes el context parseado

### Fase B — Corto plazo (1-2 días)
- [ ] `pip install sentence-transformers` → implementar `deberta_groundedness_evaluator`
- [ ] `pip install ragas datasets` → wrapper RAGAS faithfulness + answer_relevancy

### Fase C — Medio plazo (Neo4j conectado)
- [ ] `graph_faithfulness_evaluator` — verificar triples del answer en Neo4j
- [ ] `entity_coverage_evaluator` — nodos relevantes mencionados en answer
- [ ] `pip install antlr4-cypher` → `cypher_syntactic_valid`

### Fase D — TFG score de confianza
```
confianza = 0.30 * rubric_correctness
          + 0.20 * nli_groundedness
          + 0.15 * ragas_faithfulness
          + 0.15 * ragas_answer_relevancy
          + 0.10 * cypher_result_nonempty    # específico GraphRAG
          + 0.10 * entity_coverage           # específico GraphRAG
```

---

*Fuentes: Neo4j Labs text2cypher benchmark, RAGAS docs, LangSmith SDK source, arXiv 2506.02404/2510.02549/2408.08921/2506.05690, HuggingFace cross-encoder/nli-deberta-v3-base, Databricks RAG eval best practices*
