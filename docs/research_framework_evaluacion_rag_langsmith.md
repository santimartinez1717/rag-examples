# Research: Framework Universal de Evaluación RAG sobre LangSmith
**Fecha:** 2026-02-22 | **Depth:** Exhaustive | **Confianza:** Alta

---

## Executive Summary

El estado del arte en 2025-2026 muestra que **no existe un framework único que evalúe todas las arquitecturas RAG de forma comparable y produzca un score de confianza universal**. Los frameworks existentes (Ragas, DeepEval, TruLens) son parciales, orientados a arquitecturas específicas o sin calibración cruzada. La capa LangSmith proporciona la infraestructura (trazas, datasets, evaluadores) pero no el score unificado. Esto valida la propuesta del TFG.

---

## 1. Capacidades de LangSmith: Lo que ya tienes disponible

### 1.1 Evaluación Offline vs Online
LangSmith soporta dos modalidades complementarias que **deben usarse juntas**:

| Modalidad | Cuándo | Para qué |
|-----------|--------|----------|
| **Offline** | Durante desarrollo | Datasets curados, métricas reproducibles, comparación entre arquitecturas |
| **Online** | En producción | Muestreo de tráfico real, detección de degradación, alertas |

**Referencia:** [LangSmith Evaluation Docs](https://docs.langchain.com/langsmith/evaluation)

### 1.2 Evaluadores Custom (Python)
La API es simple y cualquier lógica es válida:
```python
def mi_evaluador(run, example) -> dict:
    # Accedes a: run.outputs, example.inputs, example.outputs (ground truth)
    score = calcular_score(run.outputs["answer"], example.outputs["answer"])
    return {"key": "mi_metrica", "score": score}
```
Soporta: LLM-as-judge, código determinista, modelos NLI, cualquier lógica de negocio.

### 1.3 Automations (Online Monitoring)
LangSmith permite configurar **Rule Automations** que se ejecutan sobre trazas entrantes:
- **Add to Dataset**: para construir datasets de producción automáticamente
- **Add to Annotation Queue**: para revisión humana de casos problemáticos
- **Online Evaluation**: ejecutar evaluadores sobre % del tráfico real

El sampling rate (0-1) controla el coste. Esto es clave para monitorización continua.

### 1.4 Multi-Turn Evals (2025)
Nueva funcionalidad de LangSmith que evalúa:
- Semantic intent y outcomes por turno
- **Agent trajectory**: tool calls y decisiones del agente
- Especialmente relevante para Agentic RAG con LangGraph

**Referencia:** [LangSmith Automations](https://blog.langchain.com/langsmith-production-logging-automations/)

---

## 2. Métricas de Evaluación: El Consenso de la Industria

### 2.1 Métricas Universales (válidas para cualquier RAG)

| Métrica | Qué mide | También llamada |
|---------|----------|-----------------|
| **Faithfulness / Groundedness** | ¿La respuesta está soportada por el contexto recuperado? | La más crítica. Anti-alucinación. |
| **Answer Relevancy** | ¿La respuesta es pertinente a la pregunta? | — |
| **Context Precision** | ¿Los documentos recuperados son relevantes? | Retrieval Relevance |
| **Context Recall** | ¿Se recuperó toda la información necesaria? | — |
| **Correctness** | ¿Es factualmente correcta vs ground truth? | Requiere referencia |

La industria ha convergido en estas 5 métricas como estándar. Tu framework usa 4 de ellas (correctness, relevance, groundedness, retrieval_relevance) — **alineado con el estado del arte**.

### 2.2 Sesgos conocidos del LLM-as-Judge
- Prefiere respuestas más largas
- Sesgo posicional
- Favorece sus propios outputs
- **Solución**: usar datasets con ground truth verificado para calibrar el juez automático

**Referencias:** [EvidentlyAI RAG Guide](https://www.evidentlyai.com/llm-guide/rag-evaluation) | [Patronus AI](https://www.patronus.ai/llm-testing/rag-evaluation-metrics) | [Meilisearch RAG Evaluation](https://www.meilisearch.com/blog/rag-evaluation)

---

## 3. Tipos de Evaluadores: Enfoque Multi-Capa

### 3.1 LLM-as-Judge
- **Pros**: Flexible, entiende matices, no requiere datos de entrenamiento
- **Cons**: Coste API, variabilidad, sesgos sistemáticos
- **Cuándo**: Correctness, Relevance (métricas cualitativas)

### 3.2 Modelos NLI (Natural Language Inference) ⭐ Infrautilizados
Son la alternativa más potente al LLM-as-judge para groundedness/faithfulness:

| Modelo | Tamaño | Uso recomendado |
|--------|--------|-----------------|
| `cross-encoder/nli-deberta-v3-base` | ~184MB | **Mejor balance calidad/velocidad** |
| `khalidalt/DeBERTa-v3-large-mnli` | ~700MB | Máxima precisión |
| `FacebookAI/roberta-large-mnli` | ~355MB | Alternativa consolidada |
| `cross-encoder/nli-roberta-base` | ~125MB | Más ligero |

**Cómo funciona para RAG:**
```
Premise = contexto recuperado
Hypothesis = respuesta generada
Output = {entailment, neutral, contradiction}
→ entailment_score = groundedness_score
```

**Ventajas sobre LLM-as-judge:**
- Sin coste de API
- Determinista (reproducible)
- Milisegundos por evaluación
- Especialmente preciso para groundedness y retrieval_relevance

**Referencias:** [Deepset Groundedness](https://www.deepset.ai/blog/rag-llm-evaluation-groundedness) | [HuggingFace DeBERTa-v3-base-mnli](https://huggingface.co/khalidalt/DeBERTa-v3-large-mnli)

### 3.3 Frameworks Externos (Ragas, DeepEval, TruLens)

| Framework | Fortaleza | Debilidad | Integración LangSmith |
|-----------|-----------|-----------|----------------------|
| **Ragas** | Ligero, fácil, referencia académica | Rígido, poca personalización | ✅ Nativa |
| **DeepEval** | Producción, CI/CD, métricas agénticas | Más complejo | ✅ Disponible |
| **TruLens** | Monitoring básico | Incompleto, menos maduro | Parcial |

**Recomendación**: Usar Ragas para **validación cruzada** (contrastar que tus métricas propias son coherentes con la literatura), no como framework principal.

**Referencias:** [DeepEval vs Ragas](https://deepeval.com/blog/deepeval-vs-ragas) | [Comparativa 2025](https://medium.com/@mahernaija/choosing-the-right-llm-evaluation-framework-in-2025-deepeval-ragas-giskard-langsmith-and-c7133520770c)

---

## 4. El Score de Confianza: Cómo Construirlo

### 4.1 Aproximación por Ponderación Lineal (tu propuesta actual)
```
confianza = w1·correctness + w2·relevance + w3·groundedness + w4·retrieval_relevance
```
**Validada por la literatura**: Databricks, Anyscale y otros frameworks usan exactamente esta aproximación con pesos ajustables según dominio.

Ejemplo de pesos documentados:
- Correctness: 60% (si el dominio prioriza exactitud factual)
- Comprehensiveness: 20%
- Readability: 20%

Para RAG crítico se recomienda ponderar más **faithfulness/groundedness** (es la métrica más directamente relacionada con alucinaciones).

### 4.2 Trust-Score (ICLR 2025) — Referencia Académica Clave
Paper: *"Measuring and Enhancing Trustworthiness of LLMs in RAG through Grounded Attributions and Learning to Refuse"*
- **Arxiv:** [2409.11242](https://arxiv.org/abs/2409.11242)
- **GitHub:** [declare-lab/trust-align](https://github.com/declare-lab/trust-align)
- Introduce **Trust-Score**: métrica holística de confiabilidad de LLMs en RAG
- Evalúa: quality citations, ability to refuse when context is insufficient
- **26/27 modelos** mejoran sustancialmente con Trust-Align vs baselines competitivos
- Benchmarks: ASQA, QAMPARI, ELI5

> Este paper es una referencia directa para justificar académicamente el score de confianza del TFG.

### 4.3 Calibración Estadística (avanzado)
Para ir más allá de la ponderación lineal:
- **Temperature Scaling**: parámetro T que reescala los logits antes del softmax. Simple y efectivo.
- **Platt Scaling**: versión logística de temperatura scaling.
- **Uso en RAG**: calibrar que un score de 0.8 realmente signifique 80% de probabilidad de respuesta correcta.

**Referencia:** [AWS Temperature Scaling](https://docs.aws.amazon.com/prescriptive-guidance/latest/ml-quantifying-uncertainty/temp-scaling.html)

---

## 5. Métricas Específicas por Arquitectura

### 5.1 Agentic RAG (LangGraph)
Métricas que van **más allá** de las universales:
- **Tool Call Count**: número de herramientas invocadas por query (eficiencia)
- **Loop Count / Cycle Count**: iteraciones del grafo (detecta loops infinitos)
- **Query Rewrite Rate**: % de queries que requieren reescritura (indica calidad del retriever)
- **Agent Trajectory Quality**: ¿la secuencia de decisiones fue óptima? (LangSmith Multi-Turn Evals)
- **First-Pass Success Rate**: % de respuestas correctas sin reescritura

### 5.2 GraphRAG (Neo4j + Cypher)
- **Cypher Validity Rate**: % de queries Cypher que se ejecutan sin error sintáctico
- **Cypher Execution Success**: % que retornan resultados (no vacíos)
- **Entity Coverage**: ¿se recuperaron las entidades relevantes para la pregunta?
- **Relationship Precision**: relevancia de las relaciones del subgrafo recuperado

> Nota: GraphRAG-Bench ([GitHub](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark)) es el benchmark de referencia para GraphRAG en 2025, con tasks de fact retrieval, complex reasoning y contextual summarization.

### 5.3 SQL RAG (Text-to-SQL)
- **SQL Validity Rate**: % de queries SQL sintácticamente válidas
- **SQL Execution Success**: % que se ejecutan sin error en la BD
- **Result Non-Empty Rate**: % con resultados (no vacíos)
- **Numerical Accuracy**: precisión en respuestas con valores numéricos

### 5.4 Hybrid RAG (Vector + BM25)
- **Retrieval Diversity**: ¿los documentos de vector y BM25 se complementan o se solapan?
- **Fusion Quality**: calidad de la combinación (RRF score)
- **Semantic vs Keyword Balance**: proporción de documentos de cada fuente

**Referencias:** [Modern RAG Architectures 2025](https://synthimind.net/blog/rag-optimization-strategies-2025/) | [GraphRAG vs SQL RAG](https://towardsdatascience.com/graph-rag-vs-sql-rag/)

---

## 6. Arquitectura Recomendada del Framework

### Capas del framework (de abajo arriba):

```
┌─────────────────────────────────────────────────────┐
│  SCORE DE CONFIANZA CALIBRADO (0-100%)              │  ← Capa 4: Calibración
│  confianza = Σ wi · métricai (+ métricas específicas)│
├─────────────────────────────────────────────────────┤
│  EVALUADORES AUTOMÁTICOS                            │  ← Capa 3: Evaluadores
│  LLM-as-judge + NLI models + validadores code-based │
├─────────────────────────────────────────────────────┤
│  INTERFAZ UNIVERSAL                                 │  ← Capa 2: Abstracción
│  RAGInterface: inputs → {answer, documents, metadata}│
├─────────────────────────────────────────────────────┤
│  LANGSMITH (núcleo de observabilidad)               │  ← Capa 1: Infraestructura
│  Trazas + Datasets + Experimentos + Automations     │
├─────────────────────────────────────────────────────┤
│  AGENTES RAG (Agentic, Graph, SQL, Hybrid)          │  ← Capa 0: Implementaciones
└─────────────────────────────────────────────────────┘
```

### Principios de diseño:
1. **Agnóstico a la arquitectura**: la interfaz universal abstrae el RAG subyacente
2. **Evaluadores intercambiables**: LLM-as-judge puede ser reemplazado por NLI sin cambiar el resto
3. **Scores comparables**: mismo proceso de cálculo para todas las arquitecturas
4. **Offline + Online**: mismos evaluadores funcionan en ambos modos
5. **Extensible**: añadir nueva arquitectura = implementar la interfaz universal

---

## 7. Gaps del Estado del Arte (la aportación del TFG)

| Problema | Herramienta existente | Lo que falta |
|----------|----------------------|--------------|
| Evaluación aislada por arquitectura | Ragas, DeepEval | **Comparación cross-arquitectura** con el mismo proceso |
| Métricas separadas | Cualquier framework | **Score único calibrado** interpretable |
| Evaluación offline solo | LangSmith básico | **Monitorización continua + alertas** |
| LLM-as-judge costoso | Estado del arte | **Evaluadores NLI** como alternativa determinista |
| Sin métricas específicas | Frameworks generales | **Métricas por tipología** (Cypher, SQL, tool calls) |

---

## 8. Recomendaciones para el TFG

### Orden de implementación sugerido:
1. ✅ **Evaluadores LLM-as-judge** (ya hecho, 4 métricas universales)
2. **Evaluadores NLI** con DeBERTa-v3-base — añadir como capa paralela
3. **Interfaz universal** (`RAGInterface` abstract class)
4. **SQL RAG + Hybrid RAG** — con datasets propios
5. **Métricas específicas** (Cypher validity, SQL validity, tool_call_count)
6. **Score calibrado** — ponderación + análisis de sobreconfianza

### Para la justificación académica:
- Citar [Trust-Score (ICLR 2025)](https://arxiv.org/abs/2409.11242) para fundamentar el score de confianza
- Citar [Ragas paper](https://arxiv.org/abs/2309.15217) para las métricas universales
- Usar Ragas/DeepEval como **validación cruzada** de tus evaluadores propios

---

## Fuentes

- [LangSmith Evaluation Docs](https://docs.langchain.com/langsmith/evaluation)
- [LangSmith Automations](https://blog.langchain.com/langsmith-production-logging-automations/)
- [Ragas + LangSmith integration](https://docs.ragas.io/en/stable/howtos/integrations/langsmith/)
- [DeepEval vs Ragas](https://deepeval.com/blog/deepeval-vs-ragas)
- [EvidentlyAI RAG Evaluation Guide](https://www.evidentlyai.com/llm-guide/rag-evaluation)
- [Deepset Groundedness](https://www.deepset.ai/blog/rag-llm-evaluation-groundedness)
- [Trust-Score ICLR 2025 (arxiv)](https://arxiv.org/abs/2409.11242)
- [trust-align GitHub](https://github.com/declare-lab/trust-align)
- [GraphRAG-Bench](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark)
- [Modern RAG Architectures 2025](https://synthimind.net/blog/rag-optimization-strategies-2025/)
- [Graph RAG vs SQL RAG (TDS)](https://towardsdatascience.com/graph-rag-vs-sql-rag/)
- [AWS Temperature Scaling](https://docs.aws.amazon.com/prescriptive-guidance/latest/ml-quantifying-uncertainty/temp-scaling.html)
- [HuggingFace DeBERTa-v3-large-mnli](https://huggingface.co/khalidalt/DeBERTa-v3-large-mnli)
- [LangSmith Multi-Turn Evals](https://www.blog.langchain.com/insights-agent-multiturn-evals-langsmith/)
- [Comparing RAG eval frameworks 2025](https://medium.com/@mahernaija/choosing-the-right-llm-evaluation-framework-in-2025-deepeval-ragas-giskard-langsmith-and-c7133520770c)
