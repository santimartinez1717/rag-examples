# Investigación Exhaustiva: Evaluación de Sistemas GraphRAG con LangChain y LangSmith

> Fecha: 2026-03-28
> Objetivo: Framework de evaluación general para cualquier GraphRAG de LangChain
> Contexto: TFG — "Hacia el 95% de Confianza: Monitorización y Calibración de Agentes de IA basados en RAG"

---

## Índice

1. [Métricas específicas para GraphRAG vs RAG vectorial](#1-métricas-específicas-para-graphrag-vs-rag-vectorial)
2. [LangSmith para evaluación de GraphRAG](#2-langsmith-para-evaluación-de-graphrag)
3. [Frameworks complementarios](#3-frameworks-complementarios-más-allá-de-langsmith)
4. [Consideraciones específicas de GraphRAG](#4-consideraciones-específicas-de-graphrag)
5. [Arquitectura de un evaluador general](#5-arquitectura-de-un-evaluador-general)
6. [Benchmark datasets para GraphRAG](#6-benchmark-datasets-para-graphrag)
7. [Score de confianza compuesto](#7-score-de-confianza-compuesto)
8. [Referencias académicas](#8-referencias-académicas)

---

## 1. Métricas específicas para GraphRAG vs RAG vectorial

### 1.1 Tabla comparativa: qué mide cada paradigma

| Dimensión | RAG Vectorial | GraphRAG (Neo4j + Cypher) |
|-----------|--------------|--------------------------|
| Retrieval | Cosine similarity sobre embeddings | Ejecución de query Cypher sobre grafo |
| Contexto | Lista de chunks de texto | Subgrafo: nodos + relaciones + propiedades |
| Faithfulness | ¿El answer está en los chunks? | ¿El answer está soportado por los triples del grafo? |
| Retrieval quality | Context precision/recall sobre chunks | Cypher validity + result non-empty + entity coverage |
| Hallucination | Información no presente en chunks | Relaciones/entidades inventadas que no existen en el grafo |
| Multi-hop | Difícil — depende de reranking | Natural — el grafo permite traversals en profundidad |
| Debugging | Ver qué chunks se recuperaron | Ver qué Cypher se generó, qué resultados retornó |

### 1.2 Métricas únicas de GraphRAG (no existen en RAG vectorial)

#### Grupo A — Calidad del Cypher generado

**1. Cypher Generated (binaria)**
¿El LLM generó algún query Cypher? Si falla aquí, todo lo demás falla.
```python
def cypher_generated(outputs: dict) -> bool:
    cypher = _extract_cypher(outputs.get("context", ""))
    return bool(cypher) and any(kw in cypher.upper() for kw in ["MATCH", "RETURN", "CALL"])
```

**2. Cypher Syntactic Validity (binaria)**
¿Es sintácticamente válido sin ejecutarlo? Detecta errores de generación.
- Herramienta: `antlr4-cypher` (parser Cypher open-source)
- Valor: 1 = válido, 0 = error de sintaxis (con mensaje de error)

**3. Cypher Execution Success (binaria)**
¿Se ejecutó sin error en Neo4j? Filtra queries sintácticamente correctos pero semánticamente erróneos (labels inexistentes, propiedades mal nombradas).

**4. Cypher Result Non-Empty (binaria)**
¿La query retornó al menos 1 resultado? Detecta queries válidas que no encuentran nada.
- Impacto: si es 0, el LLM respondió basado en un contexto vacío → alta probabilidad de alucinación.

**5. Cypher Complexity Score (continua)**
Score compuesto de:
- Número de cláusulas MATCH (traversal depth)
- Presencia de WHERE (filtrado específico)
- Presencia de aggregaciones (COUNT, SUM, COLLECT...)
- Variable-length paths (`*1..3`) — multi-hop explícito
- Número de patrones de relaciones (`-[]-`)

**6. Cypher Semantic Correctness (LLM-as-judge)**
¿El Cypher generado responde correctamente a la pregunta dado el schema?
```
Input: question + schema + generated_cypher
Judge: ¿este Cypher, dado el schema, produciría los datos necesarios para responder la pregunta?
Output: score [0,1] + reasoning
```

**7. Schema Adherence (determinista)**
¿El Cypher usa únicamente labels y propiedades que existen en el schema?
```python
def schema_adherence(cypher: str, schema: dict) -> float:
    """
    Extrae labels del Cypher, verifica si existen en el schema.
    schema = {"nodes": ["Employee", "Department"], "relationships": [...]}
    """
    known_labels = set(schema.get("node_labels", []))
    used_labels = set(re.findall(r':\s*([A-Za-z_][A-Za-z0-9_]*)', cypher))
    if not used_labels:
        return 1.0
    correct = sum(1 for l in used_labels if l in known_labels)
    return correct / len(used_labels)
```

#### Grupo B — Calidad del retrieval de grafo

**8. Entity Coverage (continua)**
¿El contexto recuperado contiene las entidades relevantes para la pregunta?
```
Fórmula: |entidades_en_contexto ∩ entidades_relevantes_pregunta| / |entidades_relevantes_pregunta|
```

**9. Graph Faithfulness (continua)**
¿Las relaciones afirmadas en el answer existen como triples en el grafo?
```
Algoritmo:
1. LLM extrae triples (sujeto, predicado, objeto) del answer
2. Para cada triple: MATCH (s)-[r]->(o) WHERE s.name CONTAINS $subj AND o.name CONTAINS $obj RETURN count(*)
3. Score = triples_verificados / triples_totales
```

**10. Relationship Precision (continua)**
¿Los tipos de relaciones en los resultados de Neo4j son relevantes para la pregunta?
```
Input: tipos de relaciones en db_results + pregunta original
Judge (LLM): ¿son estas relaciones relevantes?
```

**11. Multi-hop Depth Score (continua)**
¿Qué profundidad de traversal requirió el query?
- 1-hop: `MATCH (a)-[r]->(b)`
- 2-hop: `MATCH (a)-[r1]->(b)-[r2]->(c)`
- Variable: `MATCH (a)-[*1..3]->(b)`
- Score normalizado: `depth / max_expected_depth`

#### Grupo C — Métricas de generación adaptadas a grafo

**12. Graph-Grounded Faithfulness**
Equivalente de faithfulness RAGAS pero con el contexto de grafo como fuente de verdad:
```
Input: answer + db_results (como lista de dicts de Neo4j)
Algoritmo RAGAS: descomponer answer en claims atómicos, verificar si cada claim es entailed por los db_results
```

**13. Empty Context Hallucination Rate**
¿Con qué frecuencia el LLM genera una respuesta cuando el contexto está vacío?
- Detecta el patrón más peligroso: Cypher se ejecutó, retornó 0 resultados, pero el LLM "respondió" de todas formas.
- Implementación: `score = 1 if db_results_empty AND answer_nonempty else 0`

### 1.3 Métricas universales adaptadas a GraphRAG

Las métricas de RAG estándar siguen siendo válidas pero requieren adaptación:

| Métrica estándar | Adaptación para GraphRAG |
|----------------|--------------------------|
| Faithfulness | `context` = output de Neo4j (lista de dicts), no chunks de texto |
| Context Precision | Dificil — no hay ranking de chunks. Se puede medir si CADA fila de db_results fue usada en el answer |
| Context Recall | ¿Cuánto del ground truth está soportado por db_results? |
| Answer Relevancy | Sin cambios — mide embedding similarity entre pregunta y answer |
| Correctness | Sin cambios — compara contra ground truth |

---

## 2. LangSmith para evaluación de GraphRAG

### 2.1 evaluate() API — Referencia completa

```python
from langsmith import Client

client = Client()

results = client.evaluate(
    target=rag_fn,                # Callable[[dict], dict]
    data="dataset-name",          # str (nombre) | UUID | Iterator[Example]
    evaluators=[                  # List[Callable]
        correctness,
        relevance,
        groundedness,
        cypher_validity,
    ],
    experiment_prefix="graphrag-neo4j-v1",  # str — prefijo experimento
    description="Evaluación con métricas GraphRAG específicas",  # str
    max_concurrency=4,            # int — paralelismo
    # num_repetitions=3,          # int — repetir cada ejemplo N veces (para varianza)
)
```

**Versión asíncrona** para datasets grandes:
```python
results = await client.aevaluate(
    target=rag_fn,
    data="dataset-name",
    evaluators=evaluators,
    experiment_prefix="graphrag-async",
    max_concurrency=8,
)
```

### 2.2 Firmas de evaluadores — Tres patrones

**Patrón 1: reference-based** (necesita ground truth)
```python
def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """inputs: del dataset. outputs: del rag_fn. reference_outputs: ground truth."""
    # inputs["question"], outputs["answer"], reference_outputs["answer"]
    return {"key": "correctness", "score": 0.0 or 1.0, "comment": "..."}
```

**Patrón 2: reference-free** (solo inputs + outputs del RAG)
```python
def cypher_result_nonempty(inputs: dict, outputs: dict) -> dict:
    """No necesita ground truth — evalúa comportamiento interno."""
    db_results = _extract_db_results(outputs.get("context", ""))
    return {"key": "cypher_result_nonempty", "score": 1 if db_results else 0}
```

**Patrón 3: acceso al objeto Run** (para métricas de rendimiento)
```python
from langsmith.schemas import Run, Example

def latency_evaluator(run: Run, example: Example) -> list[dict]:
    """Accede a métricas de rendimiento del Run de LangSmith."""
    return [
        {"key": "latency_sec", "score": run.latency},
        {"key": "total_tokens", "score": run.total_tokens or 0},
        {"key": "total_cost_usd", "score": float(run.total_cost or 0)},
        {"key": "had_error", "score": 1 if run.error else 0},
    ]
```

**Patrón 4: evaluador que retorna múltiples métricas**
```python
def cypher_complexity(inputs: dict, outputs: dict) -> list[dict]:
    """Un evaluador puede retornar múltiples métricas a la vez."""
    cypher = _extract_cypher(outputs.get("context", "")).upper()
    return [
        {"key": "cypher_match_count", "score": len(re.findall(r'\bMATCH\b', cypher))},
        {"key": "cypher_has_where", "score": 1 if 'WHERE' in cypher else 0},
        {"key": "cypher_has_aggregation", "score": 1 if any(ag in cypher for ag in ['COUNT','SUM','AVG']) else 0},
    ]
```

### 2.3 Evaluadores LLM-as-judge para calidad de Cypher

El patrón más poderoso para evaluar la calidad semántica del Cypher generado:

```python
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict, Annotated
import json

llm = ChatOpenAI(model="gpt-4.1", temperature=0)

CYPHER_QUALITY_PROMPT = """Eres un experto en Neo4j y Cypher Query Language.

Evalúa la calidad del query Cypher generado para responder la pregunta dada, usando el schema del grafo.

Schema del grafo:
{schema}

Pregunta del usuario:
{question}

Cypher generado:
{cypher}

Resultados de la base de datos:
{db_results}

Evalúa en escala 0.0-1.0:
1.0 — El Cypher es correcto, eficiente y retorna exactamente los datos necesarios
0.75 — Correcto pero subóptimo (redundante, innecesariamente complejo)
0.5 — Parcialmente correcto (retorna datos relevantes pero incompletos o con ruido)
0.25 — Mayormente incorrecto (mala interpretación del schema o la pregunta)
0.0 — Completamente incorrecto, vacío, o sintácticamente inválido

Responde SOLO con JSON: {{"score": <float>, "reasoning": "<una frase>"}}"""

def cypher_quality_evaluator(inputs: dict, outputs: dict) -> dict:
    context = outputs.get("context", "")
    cypher = _extract_cypher(context)
    db_results = _extract_db_results(context)

    if not cypher:
        return {"key": "cypher_quality", "score": 0.0, "comment": "no cypher generated"}

    # Obtener schema del grafo (se puede cachear)
    schema = inputs.get("schema", "Not provided")

    prompt = CYPHER_QUALITY_PROMPT.format(
        schema=schema,
        question=inputs["question"],
        cypher=cypher,
        db_results=str(db_results)[:500],  # limitar tokens
    )
    response = llm.invoke(prompt).content
    try:
        parsed = json.loads(response)
        return {"key": "cypher_quality", "score": parsed["score"], "comment": parsed["reasoning"]}
    except Exception:
        return {"key": "cypher_quality", "score": 0.0, "comment": "parse error"}
```

### 2.4 Estructura del dataset LangSmith para GraphRAG

**Schema mínimo:**
```python
examples = [
    {
        "inputs": {
            "question": "How many employees report to the CEO?",
            # Campos opcionales pero útiles:
            "schema": "Node labels: Employee, Department...",  # para evaluadores Cypher
        },
        "outputs": {
            "answer": "15 employees report to the CEO",
            # Campos opcionales:
            "expected_cypher": "MATCH (e:Employee)-[:REPORTS_TO]->(c:CEO) RETURN count(e)",
            "expected_entity_count": 15,  # para validación numérica
        },
        "metadata": {
            "category": "multi-hop",         # single-hop | multi-hop | aggregation | path
            "difficulty": "medium",
            "requires_relationship": "REPORTS_TO",
        }
    }
]

# Crear dataset
client = Client()
dataset = client.create_dataset("graphrag-neo4j-eval")
client.create_examples(dataset_id=dataset.id, examples=examples)
```

**Campos que debe retornar el rag_fn para máxima evaluabilidad:**
```python
def rag_fn(inputs: dict) -> dict:
    # ...ejecución del GraphRAG...
    return {
        "answer": str,           # REQUERIDO — respuesta en lenguaje natural
        "context": str,          # RECOMENDADO — "Cypher Query: ...\n\nDatabase Results: [...]"
        # Campos opcionales pero útiles para evaluadores:
        "cypher_query": str,     # Cypher generado (parseado de context)
        "db_results": list,      # Resultados raw de Neo4j
        "db_result_count": int,  # len(db_results)
        "cypher_valid": bool,    # Si se ejecutó sin error
    }
```

### 2.5 Trazas y spans: qué capturar para GraphRAG

**Enriquecimiento de trazas con metadata:**
```python
import langsmith as ls

def neo4j_graphrag_wrapper(inputs: dict) -> dict:
    # ... ejecución del chain ...
    result = chain.invoke({"query": inputs["question"]})

    # Extraer intermediate_steps
    intermediate_steps = result.get("intermediate_steps", [])
    cypher_query = intermediate_steps[0].get("query", "") if intermediate_steps else ""
    db_results = intermediate_steps[1].get("context", []) if len(intermediate_steps) > 1 else []

    # Enriquecer la traza LangSmith con metadata GraphRAG
    rt = ls.get_current_run_tree()
    if rt:
        rt.metadata.update({
            "cypher_query": cypher_query,
            "db_result_count": len(db_results) if isinstance(db_results, list) else 0,
            "cypher_generated": bool(cypher_query),
            "results_empty": len(db_results) == 0 if isinstance(db_results, list) else True,
        })

    context = f"Cypher Query: {cypher_query}\n\nDatabase Results: {db_results}"
    return {"answer": result.get("result", ""), "context": context}
```

**Qué aparece en LangSmith UI con `return_intermediate_steps=True`:**
```
Run tree:
└── GraphCypherQAChain (root)
    ├── LLMChain (cypher_generation)    ← genera el Cypher
    │   └── ChatOpenAI
    ├── Neo4j (execution)               ← ejecuta el Cypher
    └── LLMChain (qa_generation)        ← genera la respuesta final
        └── ChatOpenAI
```

Cada span tiene: `inputs`, `outputs`, `latency`, `tokens`, `error`.

### 2.6 Evaluadores built-in de LangSmith relevantes

LangSmith no tiene evaluadores built-in específicos para GraphRAG. Los disponibles en `langchain/smith` son evaluadores genéricos:

| Evaluador | Tipo | Aplicable a GraphRAG |
|-----------|------|---------------------|
| `StringDistance` | determinista | Para comparar Cypher esperado vs generado |
| `ExactMatch` | determinista | Para respuestas con valor exacto (números, nombres) |
| `EmbeddingDistance` | embeddings | Para answer relevancy aproximada |
| `LLMStringEvaluator` | LLM-as-judge | Para correctness/relevance |
| Custom Python functions | cualquiera | La opción más flexible |

**Recomendación:** Para GraphRAG, usar exclusivamente custom evaluators. Los built-in son demasiado genéricos.

---

## 3. Frameworks complementarios: más allá de LangSmith

### 3.1 RAGAS: compatibilidad con GraphRAG

**Estado del soporte (2026):** RAGAS está diseñado para RAG con documentos. No tiene soporte nativo para contexto de grafo. Sin embargo, todas sus métricas pueden adaptarse convirtiendo el output de Neo4j a texto.

**Métricas RAGAS y su compatibilidad:**

| Métrica RAGAS | Campos requeridos | Compatible con GraphRAG | Adaptación necesaria |
|--------------|-------------------|------------------------|---------------------|
| `faithfulness` | question, answer, contexts (list[str]) | Si | `contexts = [str(db_results)]` |
| `answer_relevancy` | question, answer | Si | Sin cambios |
| `context_precision` | question, contexts, ground_truth | Parcial | `contexts = [str(db_results)]` |
| `context_recall` | question, contexts, ground_truth | Parcial | `contexts = [str(db_results)]` |
| `context_entity_recall` | contexts, ground_truth | Si | Mide si entidades del GT están en db_results |
| `noise_sensitivity` | question, answer, contexts, ground_truth | Si | Útil para detectar alucinaciones con grafo vacío |

**Wrapper RAGAS para LangSmith con contexto de grafo:**
```python
from ragas.metrics import faithfulness, answer_relevancy, context_entity_recall
from ragas import evaluate as ragas_evaluate
from datasets import Dataset

def make_ragas_evaluator(metric):
    def evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
        # Adaptar contexto de grafo al formato RAGAS
        raw_context = outputs.get("context", "")
        db_results = _extract_db_results(raw_context)

        # Convertir db_results a texto legible
        if db_results:
            context_text = "\n".join(str(row) for row in db_results)
        else:
            cypher = _extract_cypher(raw_context)
            context_text = f"Cypher executed: {cypher}\nNo results returned."

        ds = Dataset.from_dict({
            "question": [inputs["question"]],
            "answer": [outputs.get("answer", "")],
            "contexts": [[context_text]],
            "ground_truth": [reference_outputs.get("answer", "")],
        })
        result = ragas_evaluate(ds, metrics=[metric])
        score = float(result[metric.name])
        return {"key": f"ragas_{metric.name}", "score": score}

    evaluator.__name__ = f"ragas_{metric.name}"
    return evaluator

# Crear evaluadores para LangSmith
ragas_faithfulness_eval = make_ragas_evaluator(faithfulness)
ragas_relevancy_eval = make_ragas_evaluator(answer_relevancy)
ragas_entity_recall_eval = make_ragas_evaluator(context_entity_recall)
```

**Limitación clave:** RAGAS asume que `contexts` es una lista de fragmentos de texto. Para GraphRAG, hay que serializar los resultados de Neo4j. La métrica `context_entity_recall` es especialmente útil porque mide si las entidades del ground truth aparecen en los db_results — perfectamente alineada con lo que queremos medir en GraphRAG.

### 3.2 DeepEval: compatibilidad con GraphRAG

**Estado del soporte (2026):** DeepEval tiene 50+ métricas pero **no tiene soporte nativo para GraphRAG o Cypher**. Sin embargo, su `LLMTestCase` es flexible.

**Métricas DeepEval aplicables a GraphRAG:**

```python
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    HallucinationMetric,
    GEval,
)
from deepeval.test_case import LLMTestCase

# Adaptar output GraphRAG al formato DeepEval
def graphrag_to_deepeval_testcase(inputs: dict, outputs: dict, reference: dict) -> LLMTestCase:
    context_text = _serialize_neo4j_context(outputs.get("context", ""))
    return LLMTestCase(
        input=inputs["question"],
        actual_output=outputs.get("answer", ""),
        expected_output=reference.get("answer", ""),
        retrieval_context=[context_text],  # lista de strings
    )
```

**Ventaja clave de DeepEval sobre RAGAS para GraphRAG: G-Eval**
G-Eval permite definir métricas completamente custom con criterios en lenguaje natural:
```python
cypher_quality_metric = GEval(
    name="CypherSemanticCorrectness",
    criteria="""
    Evalúa si el query Cypher responde correctamente a la pregunta dado el schema del grafo.
    Penaliza: queries que ignoran relaciones relevantes, labels incorrectos, falta de filtros necesarios.
    Valora: traversal correcto, uso apropiado de aggregaciones, respeto del schema.
    """,
    evaluation_steps=[
        "Analiza si el Cypher usa los node labels correctos del schema",
        "Verifica si el traversal de relaciones es correcto para la pregunta",
        "Comprueba si los filtros WHERE son apropiados",
        "Evalúa si las aggregaciones (COUNT, etc.) son las correctas",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
)
```

**Integración DeepEval con LangSmith:**
```python
def make_deepeval_evaluator(metric):
    def evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
        test_case = graphrag_to_deepeval_testcase(inputs, outputs, reference_outputs)
        metric.measure(test_case)
        return {
            "key": metric.name,
            "score": metric.score,
            "comment": metric.reason,
        }
    evaluator.__name__ = metric.name
    return evaluator
```

### 3.3 TruLens: compatibilidad con GraphRAG

**Estado del soporte (2026):** TruLens usa el "RAG Triad" (Context Relevance, Groundedness, Answer Relevance). Es menos maduro que RAGAS/DeepEval pero tiene integración nativa con LangChain.

**Limitación crítica para GraphRAG:** TruLens asume que el contexto viene de un `VectorStoreRetriever`. No tiene mecanismo built-in para capturar el output de Neo4j como contexto.

**Uso con GraphRAG (workaround):**
```python
from trulens.apps.langchain import TruChain
from trulens.core import TruSession

session = TruSession()

# TruLens necesita que la chain esté decorada
tru_recorder = TruChain(
    chain,
    app_name="GraphRAG-Neo4j",
    feedbacks=[f_groundedness, f_answer_relevance, f_context_relevance],
)

# El problema: TruLens buscará un retriever en la chain para extraer el contexto
# Para GraphRAG, hay que definir manualmente cómo se extrae el contexto
```

**Veredicto:** TruLens no merece la complejidad de integración para GraphRAG. RAGAS + DeepEval + evaluadores custom LangSmith son superiores.

### 3.4 Evaluación offline vs online

| | Offline (desarrollo) | Online (producción) |
|--|---------------------|---------------------|
| Dataset | Curado, con ground truth | Tráfico real, sin GT |
| Evaluadores | Todos (incluyen reference-based) | Solo reference-free |
| LangSmith API | `client.evaluate()` | Automations + Rules |
| Frecuencia | On-demand (por experimento) | Continua (por traza) |
| Coste | Controlado | Variable (sampling) |
| Uso | Comparar arquitecturas, regresión | Monitorización, alertas |

**Configurar evaluación online en LangSmith:**
1. UI → Projects → "01-agentic-rag" → Automations → Add Rule
2. Configurar: trigger = "New trace", filter = `error IS NULL`, sampling = 0.3 (30%)
3. Action: "Online Evaluation" → seleccionar evaluadores (solo reference-free)
4. Los resultados aparecen como feedback en cada traza

---

## 4. Consideraciones específicas de GraphRAG

### 4.1 Evaluación de la calidad del grafo de conocimiento subyacente

El grafo en sí puede ser la fuente del problema. Métricas de calidad del KG:

**Completeness (cobertura):** ¿Qué fracción de entidades/relaciones esperadas están en el grafo?
```python
def graph_completeness(driver, expected_node_types: list, expected_rel_types: list) -> dict:
    with driver.session() as session:
        # Contar nodos por label
        node_counts = {}
        for label in expected_node_types:
            result = session.run(f"MATCH (n:{label}) RETURN count(n) as cnt")
            node_counts[label] = result.single()["cnt"]

        # Contar relaciones por tipo
        rel_counts = {}
        for rel_type in expected_rel_types:
            result = session.run(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) as cnt")
            rel_counts[rel_type] = result.single()["cnt"]

    return {
        "node_counts": node_counts,
        "rel_counts": rel_counts,
        "empty_node_types": [k for k, v in node_counts.items() if v == 0],
        "empty_rel_types": [k for k, v in rel_counts.items() if v == 0],
    }
```

**Connectivity (conectividad):** ¿Qué fracción de nodos están aislados (sin relaciones)?
```cypher
MATCH (n)
WITH count(n) as total
MATCH (n)
WHERE NOT (n)--()
RETURN toFloat(count(n)) / total as isolated_node_ratio
```

GraphRAG-Bench usa `organization = non-isolated nodes ratio` como métrica de calidad del grafo construido.

**Redundancy:** ¿Hay nodos o relaciones duplicadas que pueden causar resultados inflados?

### 4.2 Incomplete graph / missing edges: detectar cuando el RAG falla por el grafo

Este es el fallo más insidioso: el LLM genera Cypher correcto, se ejecuta sin error, pero retorna 0 resultados porque el grafo no tiene la información. El sistema entonces "inventa" la respuesta.

**Pipeline de diagnóstico:**
```python
def diagnose_empty_results(inputs: dict, outputs: dict, driver) -> dict:
    """
    Cuando los db_results están vacíos, diagnostica por qué.
    """
    context = outputs.get("context", "")
    cypher = _extract_cypher(context)
    db_results = _extract_db_results(context)

    if db_results:  # no es el caso que nos interesa
        return {"key": "graph_coverage_issue", "score": 0}

    # Si hay Cypher pero sin resultados: analizar si es problema del grafo
    if cypher:
        # Extraer qué labels/relaciones usa el Cypher
        labels_used = re.findall(r':\s*([A-Za-z_][A-Za-z0-9_]*)', cypher)

        with driver.session() as session:
            missing_data = []
            for label in labels_used:
                result = session.run(f"MATCH (n:{label}) RETURN count(n) as cnt").single()
                if result and result["cnt"] == 0:
                    missing_data.append(f"No {label} nodes in graph")

        return {
            "key": "graph_coverage_issue",
            "score": 1 if missing_data else 0,  # 1 = hay problema en el grafo
            "comment": "; ".join(missing_data) if missing_data else "query returned no results (data issue, not schema)"
        }

    return {"key": "graph_coverage_issue", "score": 0}
```

**Métrica derivada:** `empty_context_hallucination_rate`
```
Si db_results == [] AND answer != "I don't know" → hallucination detectada
```

### 4.3 Multi-hop reasoning evaluation

**Definición:** Una pregunta requiere multi-hop si su respuesta necesita traversar al menos 2 relaciones en el grafo.

Ejemplo:
- 1-hop: "¿Qué empleados hay en el departamento X?" → `MATCH (e:Employee)-[:WORKS_IN]->(d:Department {name: "X"})`
- 2-hop: "¿Con quién trabaja el jefe de X?" → `MATCH (m:Manager)-[:MANAGES]->(d:Department {name: "X"})<-[:WORKS_IN]-(e:Employee)`
- 3-hop: "¿Qué proyectos tienen los empleados del jefe de X?"

**Evaluación de multi-hop reasoning:**

```python
def multihop_reasoning_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """
    Evalúa si el sistema resolvió correctamente preguntas multi-hop.
    Requiere metadata en el dataset indicando la profundidad esperada.
    """
    expected_hops = inputs.get("expected_hops", 1)

    cypher = _extract_cypher(outputs.get("context", ""))
    db_results = _extract_db_results(outputs.get("context", ""))

    # Inferir profundidad del Cypher generado
    actual_hops = _count_hops_in_cypher(cypher)

    # Para 2+ hop: penalizar si el Cypher generado es 1-hop (perdió complejidad)
    hop_adequacy = min(actual_hops / max(expected_hops, 1), 1.0)

    # Verificar correctness de la respuesta final
    answer = outputs.get("answer", "")
    ref_answer = reference_outputs.get("answer", "")
    # (usar evaluador de correctness aquí)

    return {
        "key": "multihop_adequacy",
        "score": hop_adequacy,
        "comment": f"Expected {expected_hops}-hop, generated {actual_hops}-hop Cypher"
    }

def _count_hops_in_cypher(cypher: str) -> int:
    """Cuenta el número de saltos (hops) en el Cypher."""
    if not cypher:
        return 0
    # Variable-length paths cuentan como multi-hop
    vl_match = re.search(r'\*(\d+)\.\.(\d+)', cypher)
    if vl_match:
        return int(vl_match.group(2))
    # Contar patrones de relaciones
    rel_patterns = len(re.findall(r'-\[[\w:*\.]+\]-', cypher))
    return rel_patterns if rel_patterns > 0 else 1
```

### 4.4 Hallucination en contexto de grafo

Tres tipos de alucinación específicos de GraphRAG:

**Tipo 1: Relación inventada** — El LLM afirma que A está relacionado con B de una forma que no existe en el grafo.
- Detección: `graph_faithfulness_evaluator` (verificar triples en Neo4j)

**Tipo 2: Entidad inventada** — El LLM menciona un nodo que no existe en el grafo.
- Detección: comparar entidades mencionadas en el answer contra entidades en db_results

**Tipo 3: Contexto vacío ignorado** — El LLM genera respuesta cuando db_results = [].
- Detección: `empty_context_hallucination_rate`
- Este es el más peligroso y el más específico de GraphRAG.

```python
def hallucination_type_classifier(inputs: dict, outputs: dict) -> list[dict]:
    """Clasifica el tipo de alucinación presente."""
    context = outputs.get("context", "")
    db_results = _extract_db_results(context)
    answer = outputs.get("answer", "").lower()

    results = []

    # Tipo 3: contexto vacío pero hay respuesta
    if not db_results and answer and "don't know" not in answer and "no information" not in answer:
        results.append({"key": "hallucination_empty_context", "score": 1})
    else:
        results.append({"key": "hallucination_empty_context", "score": 0})

    # Tipo 2: entidades en answer no aparecen en db_results
    if db_results:
        db_text = str(db_results).lower()
        # Detectar si el answer menciona nombres no presentes en db_results
        # (heurístico: buscar nombres propios en mayúscula)
        answer_names = set(re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', outputs.get("answer", "")))
        db_text_lower = str(db_results).lower()
        phantom_entities = [n for n in answer_names if n.lower() not in db_text_lower]
        results.append({
            "key": "hallucination_phantom_entities",
            "score": min(len(phantom_entities) / max(len(answer_names), 1), 1.0),
            "comment": f"Possible phantom entities: {phantom_entities[:3]}"
        })

    return results
```

---

## 5. Arquitectura de un evaluador general

### 5.1 Diseño del wrapper `rag_fn(inputs: dict) -> dict`

El wrapper es el contrato entre cualquier GraphRAG y el framework de evaluación. Diseño:

```python
from typing import TypedDict, Optional
from abc import ABC, abstractmethod

class GraphRAGOutput(TypedDict):
    """Contrato de salida para cualquier GraphRAG evaluable."""
    answer: str                    # REQUERIDO: respuesta en lenguaje natural
    context: str                   # RECOMENDADO: contexto serializado

    # Campos opcionales para métricas avanzadas
    cypher_query: Optional[str]    # Query Cypher generado
    db_results: Optional[list]     # Resultados raw de Neo4j
    db_result_count: Optional[int] # len(db_results)
    cypher_valid: Optional[bool]   # Si se ejecutó sin error
    cypher_error: Optional[str]    # Mensaje de error si cypher_valid=False
    graph_schema: Optional[str]    # Schema del grafo (para evaluadores Cypher)
    retrieval_latency_ms: Optional[float]  # Latencia de Neo4j
    generation_latency_ms: Optional[float] # Latencia del LLM

class GraphRAGWrapper(ABC):
    """Clase base para wrappers de GraphRAG."""

    @abstractmethod
    def _run_chain(self, question: str) -> dict:
        """Ejecutar la chain específica de GraphRAG."""
        pass

    def __call__(self, inputs: dict) -> GraphRAGOutput:
        """Interfaz universal llamable por LangSmith evaluate()."""
        question = inputs["question"]
        raw = self._run_chain(question)
        return self._normalize_output(raw)

    def _normalize_output(self, raw: dict) -> GraphRAGOutput:
        """Normalizar el output raw de cualquier GraphRAG al formato estándar."""
        # Extraer answer de posibles keys
        answer = (
            raw.get("result") or raw.get("answer") or
            raw.get("output") or raw.get("response") or ""
        )

        # Extraer contexto de intermediate_steps (GraphCypherQAChain)
        context = ""
        cypher_query = ""
        db_results = []

        intermediate_steps = raw.get("intermediate_steps", [])
        if intermediate_steps:
            cypher_query = intermediate_steps[0].get("query", "") if len(intermediate_steps) > 0 else ""
            db_results = intermediate_steps[1].get("context", []) if len(intermediate_steps) > 1 else []
            context = f"Cypher Query: {cypher_query}\n\nDatabase Results: {db_results}"
        else:
            context = raw.get("context", "")

        return GraphRAGOutput(
            answer=str(answer),
            context=context,
            cypher_query=cypher_query,
            db_results=db_results,
            db_result_count=len(db_results) if isinstance(db_results, list) else 0,
        )
```

**Implementación concreta para `GraphCypherQAChain`:**
```python
class Neo4jGraphCypherWrapper(GraphRAGWrapper):
    def __init__(self, chain):
        self._chain = chain

    def _run_chain(self, question: str) -> dict:
        return self._chain.invoke({"query": question})

# Uso:
chain = create_neo4j_graphrag()
rag_fn = Neo4jGraphCypherWrapper(chain)

# Directamente llamable por LangSmith:
results = client.evaluate(rag_fn, data="dataset-name", evaluators=[...])
```

### 5.2 Helpers de parsing (reutilizables en todos los evaluadores)

```python
import ast, re
from typing import Optional

def _extract_cypher(context: str) -> str:
    """Extrae el query Cypher del campo context serializado."""
    if "Cypher Query:" in context:
        return context.split("Cypher Query:")[1].split("\n\nDatabase Results:")[0].strip()
    # Intentar extraer Cypher de texto libre
    match = re.search(r'(?:MATCH|CALL|CREATE|MERGE)\s+.*', context, re.IGNORECASE | re.DOTALL)
    return match.group(0)[:500] if match else ""

def _extract_db_results(context: str) -> list:
    """Extrae los resultados de Neo4j del campo context."""
    if "Database Results:" in context:
        raw = context.split("Database Results:")[1].strip()
        try:
            return ast.literal_eval(raw) if raw != "[]" else []
        except Exception:
            return [{"raw": raw[:200]}]
    return []

def _serialize_neo4j_context(context: str) -> str:
    """Convierte db_results a texto legible para RAGAS/DeepEval."""
    db_results = _extract_db_results(context)
    if not db_results:
        cypher = _extract_cypher(context)
        return f"No results returned for query: {cypher}"
    lines = []
    for i, row in enumerate(db_results[:20]):  # limitar a 20 filas
        if isinstance(row, dict):
            lines.append(", ".join(f"{k}={v}" for k, v in row.items()))
        else:
            lines.append(str(row))
    return "\n".join(lines)
```

### 5.3 Pipeline completo de evaluación

```python
"""
Pipeline completo: dataset → run → evaluate → score → report
Para cualquier GraphRAG de LangChain
"""
from langsmith import Client
from langsmith.utils import LangSmithConflictError
from typing import Callable, List, Dict, Optional
import json

class GraphRAGEvaluator:
    """Framework de evaluación general para cualquier GraphRAG de LangChain."""

    def __init__(self, langsmith_project: str = "graphrag-evaluation"):
        self.client = Client()
        self.project = langsmith_project

    def create_or_get_dataset(self, name: str, examples: List[Dict]) -> str:
        """Crear o reutilizar dataset en LangSmith."""
        try:
            ds = self.client.create_dataset(dataset_name=name)
            self.client.create_examples(dataset_id=ds.id, examples=examples)
            print(f"Dataset '{name}' creado con {len(examples)} ejemplos")
        except (LangSmithConflictError, Exception) as e:
            if "already exists" in str(e) or "Conflict" in str(e):
                print(f"Dataset '{name}' ya existe, reutilizando")
            else:
                raise
        return name

    def evaluate(
        self,
        rag_fn: Callable[[dict], dict],
        dataset_name: str,
        experiment_prefix: str,
        # Evaluadores universales
        include_correctness: bool = True,
        include_relevance: bool = True,
        include_groundedness: bool = True,
        # Evaluadores GraphRAG-específicos
        include_cypher_metrics: bool = True,
        include_graph_faithfulness: bool = False,  # requiere Neo4j conectado
        # Evaluadores avanzados
        include_ragas: bool = False,   # requiere pip install ragas
        include_deberta: bool = False, # requiere pip install sentence-transformers
        # Parámetros
        max_concurrency: int = 4,
        num_repetitions: int = 1,
    ):
        """
        Ejecutar evaluación completa sobre el dataset.
        """
        evaluators = self._build_evaluator_list(
            include_correctness=include_correctness,
            include_relevance=include_relevance,
            include_groundedness=include_groundedness,
            include_cypher_metrics=include_cypher_metrics,
            include_graph_faithfulness=include_graph_faithfulness,
            include_ragas=include_ragas,
            include_deberta=include_deberta,
        )

        print(f"Ejecutando evaluación: {[e.__name__ for e in evaluators]}")

        results = self.client.evaluate(
            rag_fn,
            data=dataset_name,
            evaluators=evaluators,
            experiment_prefix=experiment_prefix,
            max_concurrency=max_concurrency,
            num_repetitions=num_repetitions,
        )

        return results

    def compute_confidence_score(self, results) -> dict:
        """Calcular el score de confianza compuesto."""
        # Extraer scores medios por métrica
        scores = {}
        for result in results:
            for feedback in result.get("feedback", []):
                key = feedback["key"]
                score = feedback.get("score", 0)
                if key not in scores:
                    scores[key] = []
                scores[key].append(score)

        mean_scores = {k: sum(v)/len(v) for k, v in scores.items() if v}

        # Fórmula de confianza compuesta (Fase D)
        confidence = (
            0.30 * mean_scores.get("correctness", 0) +
            0.20 * mean_scores.get("nli_groundedness", mean_scores.get("groundedness", 0)) +
            0.15 * mean_scores.get("ragas_faithfulness", 0) +
            0.15 * mean_scores.get("relevance", 0) +
            0.10 * mean_scores.get("cypher_result_nonempty", 0) +
            0.10 * mean_scores.get("entity_coverage", 1.0)  # default 1 si no se calculó
        )

        return {
            "confidence_score": round(confidence, 4),
            "severity_band": _classify_severity(confidence),
            "metric_scores": mean_scores,
        }

    def _build_evaluator_list(self, **flags) -> list:
        evaluators = []
        if flags.get("include_correctness"):
            evaluators.append(correctness)
        if flags.get("include_relevance"):
            evaluators.append(relevance)
        if flags.get("include_groundedness"):
            evaluators.append(groundedness)
        if flags.get("include_cypher_metrics"):
            evaluators.extend([
                cypher_generated,
                cypher_result_nonempty,
                cypher_complexity,
            ])
        if flags.get("include_graph_faithfulness"):
            evaluators.append(graph_faithfulness_evaluator)
        if flags.get("include_ragas"):
            from ragas.metrics import faithfulness, answer_relevancy
            evaluators.extend([
                make_ragas_evaluator(faithfulness),
                make_ragas_evaluator(answer_relevancy),
            ])
        if flags.get("include_deberta"):
            evaluators.append(deberta_groundedness_evaluator)
        return evaluators


def _classify_severity(score: float) -> str:
    """Severity bands del framework Case-Aware LLM-as-a-Judge (arXiv:2602.20379)."""
    if score >= 0.86:
        return "Minor — Alta confianza"
    elif score >= 0.61:
        return "Moderate — Confianza parcial, revisar casos edge"
    elif score >= 0.31:
        return "Major — Confianza baja, no desplegar en producción"
    else:
        return "Critical — Sistema no confiable"
```

### 5.4 Diagrama de arquitectura completo

```
┌─────────────────────────────────────────────────────────────────┐
│  SCORE DE CONFIANZA CALIBRADO (0-100%)                          │ ← Capa 5
│  Σ(wᵢ · métricaᵢ) → severity band (Critical/Major/Moderate/Minor) │
├─────────────────────────────────────────────────────────────────┤
│  EVALUADORES (3 categorías)                                     │ ← Capa 4
│  Universales: correctness, relevance, groundedness              │
│  GraphRAG-específicos: cypher_validity, graph_faithfulness      │
│  Rendimiento: latency, tokens, cost                             │
├─────────────────────────────────────────────────────────────────┤
│  INFRAESTRUCTURA DE EVALUACIÓN                                  │ ← Capa 3
│  LangSmith evaluate() → trazas + experimentos + comparaciones  │
├─────────────────────────────────────────────────────────────────┤
│  WRAPPER UNIVERSAL: rag_fn(inputs: dict) → GraphRAGOutput       │ ← Capa 2
│  Normaliza: answer, context, cypher_query, db_results           │
├─────────────────────────────────────────────────────────────────┤
│  GRAPHRAG IMPLEMENTATIONS                                       │ ← Capa 1
│  GraphCypherQAChain | LangGraph agent | Custom Neo4j chain      │
└─────────────────────────────────────────────────────────────────┘
               ↕ Neo4j bolt://localhost:7687 ↕
┌─────────────────────────────────────────────────────────────────┐
│  KNOWLEDGE GRAPH                                                │ ← Capa 0
│  Nodes + Relationships + Properties                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Benchmark datasets para GraphRAG

### 6.1 neo4j/text2cypher-2024v1 (HuggingFace) — Recomendado

**El dataset de referencia para Text-to-Cypher evaluation.**

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `question` | str | Pregunta en lenguaje natural |
| `schema` | str | Schema del grafo Neo4j (966 schemas únicos) |
| `cypher` | str | Query Cypher esperado (ground truth) |
| `data_source` | str | 20 fuentes de datos diferentes |
| `instance_id` | str | Identificador único |

**Tamaño:** 44,387 ejemplos (train: 39,554 / test: 4,833)

**Uso en el framework:**
```python
from datasets import load_dataset

text2cypher = load_dataset("neo4j/text2cypher-2024v1", split="test")

# Convertir al formato LangSmith
examples = [
    {
        "inputs": {
            "question": row["question"],
            "schema": row["schema"],
        },
        "outputs": {
            "expected_cypher": row["cypher"],
            # No hay answer en lenguaje natural — hay que generarlo
        },
        "metadata": {"data_source": row["data_source"]}
    }
    for row in text2cypher.select(range(100))  # sample para evaluación
]
```

**Métricas aplicables:**
- Exact Match del Cypher: `outputs["cypher_query"] == reference["expected_cypher"]`
- Structural Match: mismos patrones de traversal
- Execution Match: mismo resultado de ejecución (requires BD)

### 6.2 GraphRAG-Bench (arXiv:2506.02404)

**El benchmark académico de referencia para GraphRAG (2025-2026).**

- 1,018 preguntas college-level de 16 disciplinas
- 5 tipos: Fill-in-blank, Multi-choice, Multi-select, True-or-false, Open-ended
- 9 sistemas GraphRAG evaluados: RAPTOR, LightRAG, HippoRAG, GraphRAG (Microsoft), G-Retriever, GFM-RAG, DALK, KGP, ToG
- Mejor sistema: RAPTOR (73.58% promedio)

**Lecciones para el framework:**
1. Las preguntas MC degradan con GraphRAG — el ruido en el retrieval daña la selección
2. Las preguntas OE (open-ended) mejoran consistentemente con GraphRAG
3. GraphRAG falla en matemáticas/ética — no apto para razonamiento simbólico

**Métricas usadas por GraphRAG-Bench:**
- `Accuracy`: LLM judge que compara answer con ground truth (no exact match)
- `R score`: correspondencia semántica entre rationale generado y gold rationale
- `AR metric`: ¿el answer correcto viene de razonamiento válido (no suerte)?

### 6.3 GR-Bench

Benchmark de grafo con 5 dominios: académico, e-commerce, literatura, salud, legal.
Incluye preguntas multi-hop explícitas.

### 6.4 WebQSP / GrailQA / QALD — KG Question Answering

Benchmarks clásicos de Knowledge Base Question Answering (KBQA):

| Dataset | Tamaño | Base de conocimiento | Multi-hop |
|---------|--------|---------------------|-----------|
| WebQSP | 4,737 | Freebase | Limitado |
| GrailQA | 64,331 | Freebase | Sí (hasta 4-hop) |
| QALD-9 | ~450 | DBpedia | Sí |
| LC-QuAD 2.0 | 30,000 | Wikidata/DBpedia | Sí |

**Limitación:** Están diseñados para SPARQL (no Cypher) y Freebase/Wikidata (no Neo4j custom). Requieren adaptación.

### 6.5 Dataset sintético con LangSmith

Para el TFG, la opción más práctica es generar un dataset propio:

```python
# Generar preguntas sintéticas sobre el grafo existente
QUESTION_GENERATION_PROMPT = """
Dado este schema de Neo4j:
{schema}

Genera 10 preguntas diversas en lenguaje natural que requieran:
- 3 preguntas de 1-hop (filtros simples)
- 3 preguntas de 2-hop (traversal de 2 relaciones)
- 2 preguntas con aggregación (COUNT, SUM)
- 2 preguntas con múltiples condiciones WHERE

Para cada pregunta, proporciona también la respuesta esperada.
Formato: [{{"question": "...", "answer": "...", "type": "1-hop|2-hop|aggregation|multi-condition"}}]
"""

from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI

graph = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="...")
schema = graph.get_schema

llm = ChatOpenAI(model="gpt-4.1", temperature=0.3)
response = llm.invoke(QUESTION_GENERATION_PROMPT.format(schema=schema))
# Parsear el JSON y crear el dataset
```

---

## 7. Score de confianza compuesto

### 7.1 Fórmula recomendada para GraphRAG

Basada en los pesos del framework Case-Aware LLM-as-a-Judge (arXiv:2602.20379) adaptados a GraphRAG:

```
confidence = w₁·correctness
           + w₂·nli_groundedness
           + w₃·ragas_faithfulness
           + w₄·relevance
           + w₅·cypher_result_nonempty   ← específico GraphRAG
           + w₆·entity_coverage          ← específico GraphRAG
           + penalty·empty_context_hallucination_rate
```

**Pesos recomendados (suma = 1.0):**

| Métrica | Peso | Justificación |
|---------|------|---------------|
| `correctness` (LLM-judge) | 0.25 | Anti-alucinación factual |
| `nli_groundedness` (DeBERTa) | 0.20 | Determinista, gratuito, grounding fidelity |
| `ragas_faithfulness` | 0.15 | Validación cruzada con literatura |
| `relevance` | 0.10 | Answer utility |
| `cypher_result_nonempty` | 0.15 | Crítico para GraphRAG — sin datos = sin respuesta válida |
| `entity_coverage` | 0.10 | Retrieval correctness en grafo |
| `penalty` para `empty_context_hallucination` | -0.05 | Penaliza el peor caso |

**Severity bands** (del framework Case-Aware, adaptadas):
```
≥ 0.86 → Minor    — "Alta confianza: apto para producción"
0.61–0.86 → Moderate — "Confianza parcial: revisar casos edge"
0.31–0.61 → Major   — "Confianza baja: requiere mejoras antes de producción"
< 0.31  → Critical  — "Sistema no confiable: no desplegar"
```

### 7.2 Calibración estadística

Para que el score signifique algo estadísticamente:

```python
from sklearn.calibration import calibration_curve
import numpy as np

def calibrate_confidence_scores(predicted_scores: list, actual_correct: list) -> dict:
    """
    Calibra el score de confianza contra correcciones reales.
    Usa Platt Scaling (regresión logística).

    predicted_scores: lista de confidence scores [0,1]
    actual_correct: lista de 0/1 (¿era correcta la respuesta?)
    """
    from sklearn.linear_model import LogisticRegression

    X = np.array(predicted_scores).reshape(-1, 1)
    y = np.array(actual_correct)

    platt = LogisticRegression()
    platt.fit(X, y)

    # Calibrated predictions
    calibrated = platt.predict_proba(X)[:, 1]

    # Expected Calibration Error (ECE)
    fraction_pos, mean_pred = calibration_curve(y, predicted_scores, n_bins=10)
    ece = np.mean(np.abs(fraction_pos - mean_pred))

    return {
        "ece": float(ece),  # < 0.1 = bien calibrado
        "platt_model": platt,
        "interpretation": "El score predice correctamente" if ece < 0.1 else "Score sobreestima/subestima confianza"
    }
```

---

## 8. Referencias académicas

### Papers clave

| Paper | Venue | Contribución | Relevancia para el TFG |
|-------|-------|-------------|----------------------|
| **GraphRAG-Bench** (arXiv:2506.02404) | ICLR 2026 | Benchmark con 5 tipos de preguntas, 9 sistemas. Métricas: Accuracy (LLM-judge), R-score, AR | Referencia de evaluación holística + datasets |
| **KG-Based RAG Evaluation** (arXiv:2510.02549) | 2025 | Extiende RAGAS al paradigma KG con multi-hop semantic matching y community overlap | Métricas RAGAS adaptadas a grafo |
| **GraphRAG Survey** (arXiv:2408.08921) | ACM TOIS 2024 | Las métricas RAG estándar son insuficientes para grafos — no capturan integridad estructural. GR-Bench, GraphQA | Marco teórico de métricas GraphRAG |
| **Case-Aware LLM-as-a-Judge** (arXiv:2602.20379) | 2026 | Score multidimensional con 8 métricas + severity bands (Critical/Major/Moderate/Minor) | Fórmula del score de confianza compuesto |
| **Trust-Score** (arXiv:2409.11242) | ICLR 2025 | Trust-Score holístico para LLMs en RAG: calidad citas + ability to refuse | Justificación académica del score de confianza |
| **RAGAS** (arXiv:2309.15217) | ACL EACL 2024 | Faithfulness, Answer Relevancy, Context Precision/Recall con fórmulas matemáticas | Base de las métricas de referencia |
| **When to use Graphs in RAG** (arXiv:2506.05690) | 2025 | GraphRAG supera RAG estándar en multi-hop; propone routing por tipo de pregunta | Justifica qué tipo de preguntas evaluar |

### Datasets de referencia

| Dataset | Tamaño | Uso |
|---------|--------|-----|
| `neo4j/text2cypher-2024v1` (HuggingFace) | 44,387 pares NL→Cypher | Evaluar calidad del Cypher generado |
| GraphRAG-Bench | 1,018 preguntas multi-domain | Evaluación holística del sistema |
| GrailQA | 64,331 preguntas | Multi-hop KBQA (adaptación necesaria a Cypher) |
| LC-QuAD 2.0 | 30,000 preguntas | Wikidata/DBpedia (adaptación necesaria) |

### Herramientas y librerías

| Herramienta | Uso | Instalación |
|-------------|-----|-------------|
| `langsmith` | Infraestructura de evaluación | `pip install langsmith` |
| `ragas` | Métricas RAGAS (faithfulness, etc.) | `pip install ragas datasets` |
| `deepeval` | G-Eval customizable | `pip install deepeval` |
| `sentence-transformers` | DeBERTa NLI groundedness | `pip install sentence-transformers` |
| `antlr4-cypher` | Validación sintáctica Cypher | `pip install antlr4-cypher antlr4-python3-runtime` |
| `neo4j` | Driver para graph_faithfulness | `pip install neo4j` |
| `scikit-learn` | Calibración estadística (Platt) | `pip install scikit-learn` |

---

## Apéndice: Implementación de referencia completa

### Todos los helpers de parsing

```python
import ast, re, json
from typing import Optional, List

def _extract_cypher(context: str) -> str:
    if "Cypher Query:" in context:
        return context.split("Cypher Query:")[1].split("\n\nDatabase Results:")[0].strip()
    return ""

def _extract_db_results(context: str) -> list:
    if "Database Results:" in context:
        raw = context.split("Database Results:")[1].strip()
        try:
            return ast.literal_eval(raw) if raw and raw != "[]" else []
        except Exception:
            return []
    return []

def _serialize_context_for_nlp(context: str) -> str:
    """Serializa db_results de Neo4j a texto para NLI/RAGAS."""
    db_results = _extract_db_results(context)
    if not db_results:
        return f"No results. Query: {_extract_cypher(context)}"
    rows = []
    for row in db_results[:20]:
        if isinstance(row, dict):
            rows.append(", ".join(f"{k}: {v}" for k, v in row.items() if v is not None))
        else:
            rows.append(str(row))
    return "\n".join(rows)
```

### Evaluador Cypher-quality completo (production-ready)

```python
CYPHER_EVAL_PROMPT = """You are a Neo4j expert evaluating Cypher query quality.

Graph Schema:
{schema}

User Question: {question}

Generated Cypher:
{cypher}

Database Results ({result_count} rows):
{db_results}

Score the Cypher on a 0.0-1.0 scale:
1.0 — Correct, efficient, returns exactly the needed data
0.75 — Correct but suboptimal (redundant patterns, missing efficiency)
0.5 — Partially correct (relevant data but incomplete or with noise)
0.25 — Mostly wrong (misinterprets schema or question)
0.0 — Wrong, empty, or invalid

Respond ONLY with valid JSON: {{"score": <float 0-1>, "reasoning": "<one sentence>"}}"""

_eval_llm = ChatOpenAI(model="gpt-4.1", temperature=0)

def cypher_quality_evaluator(inputs: dict, outputs: dict) -> dict:
    context = outputs.get("context", "")
    cypher = _extract_cypher(context)
    db_results = _extract_db_results(context)

    if not cypher:
        return {"key": "cypher_quality", "score": 0.0, "comment": "no cypher generated"}

    prompt = CYPHER_EVAL_PROMPT.format(
        schema=inputs.get("schema", "Unknown schema"),
        question=inputs.get("question", ""),
        cypher=cypher,
        result_count=len(db_results),
        db_results=str(db_results)[:300],
    )

    try:
        response = _eval_llm.invoke(prompt).content.strip()
        # Limpiar posible markdown
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        parsed = json.loads(response)
        return {
            "key": "cypher_quality",
            "score": float(parsed["score"]),
            "comment": parsed.get("reasoning", ""),
        }
    except Exception as e:
        return {"key": "cypher_quality", "score": 0.0, "comment": f"eval error: {e}"}
```

### Uso mínimo del framework

```python
from langsmith import Client
from graphrag_wrapper_standalone import neo4j_graphrag_wrapper_standalone

# 1. Dataset mínimo
examples = [
    {
        "inputs": {"question": "How many employees are in the company?"},
        "outputs": {"answer": "The company has 25 employees"},
    },
    {
        "inputs": {"question": "Who is the CEO?"},
        "outputs": {"answer": "John Smith is the CEO"},
    },
]

# 2. Crear dataset
client = Client()
ds = client.create_dataset("graphrag-minimal-eval")
client.create_examples(dataset_id=ds.id, examples=examples)

# 3. Evaluar
results = client.evaluate(
    neo4j_graphrag_wrapper_standalone,
    data="graphrag-minimal-eval",
    evaluators=[
        correctness,           # universal
        relevance,             # universal
        groundedness,          # universal
        cypher_generated,      # GraphRAG-specific
        cypher_result_nonempty,# GraphRAG-specific
        cypher_complexity,     # GraphRAG-specific
    ],
    experiment_prefix="graphrag-neo4j-baseline",
    max_concurrency=2,
)

# 4. Score de confianza
evaluator = GraphRAGEvaluator()
score = evaluator.compute_confidence_score(results)
print(f"Confidence: {score['confidence_score']:.2%} — {score['severity_band']}")
```

---

*Fuentes: LangSmith SDK docs, RAGAS docs, DeepEval docs, arXiv:2506.02404, arXiv:2510.02549, arXiv:2408.08921, arXiv:2602.20379, arXiv:2409.11242, arXiv:2309.15217, neo4j/text2cypher-2024v1 (HuggingFace), GraphRAG-Bench GitHub, LangSmith GitHub SDK source, Case-Aware LLM-as-a-Judge severity bands*
