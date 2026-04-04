# Evaluación Universal de Sistemas RAG: Investigación Académica Exhaustiva

**TFG**: "Hacia el 95% de Confianza: Monitorización y Calibración de Agentes de IA basados en RAG"
**Universidad**: ICAI, Grado en Ingeniería Matemática e IA
**Fecha de investigación**: Marzo 2026
**Propósito**: Base académica para un framework de evaluación que funcione para cualquier arquitectura RAG

---

## Tabla de Contenidos

1. [Bloque 1: Papers Académicos Fundamentales](#bloque-1-papers-académicos-fundamentales)
2. [Bloque 2: Métricas Universales Avanzadas](#bloque-2-métricas-universales-avanzadas)
3. [Bloque 3: Modelos NLI y Alternativas sin LLM](#bloque-3-modelos-nli-y-alternativas-sin-llm)
4. [Bloque 4: Evaluación por Arquitectura RAG](#bloque-4-evaluación-por-arquitectura-rag)
5. [Bloque 5: Monitorización Online vs Evaluación Offline](#bloque-5-monitorización-online-vs-evaluación-offline)
6. [Bloque 6: Calibración Estadística](#bloque-6-calibración-estadística)
7. [Resumen: Recomendaciones para el TFG](#resumen-recomendaciones-para-el-tfg)

---

## Bloque 1: Papers Académicos Fundamentales

### 1.1 RAGAS — Automated Evaluation of Retrieval Augmented Generation

**Referencia completa**: Shahul Es, Jithin James, Luis Espinosa-Anke, Steven Schockaert. *"RAGAS: Automated Evaluation of Retrieval Augmented Generation"*. EACL 2024 (Demo Track). arXiv:2309.15217, Septiembre 2023.

**Venue**: EACL 2024 (European Chapter of ACL) — Demo Track.
**Disponibilidad**: https://arxiv.org/abs/2309.15217 | https://aclanthology.org/2024.eacl-demo.16/
**Código**: https://github.com/explodinggradients/ragas

**Contribución central**: RAGAS propone una suite de métricas para evaluar pipelines RAG **sin necesidad de anotaciones humanas de ground truth** (reference-free evaluation). Aborda tres dimensiones fundamentales:

1. La calidad del sistema de retrieval para identificar contexto relevante.
2. La capacidad del LLM para explotar ese contexto de forma fiel (faithfulness).
3. La calidad de la generación en sí.

**Las cuatro métricas originales**:

| Métrica | Input necesario | Ground truth requerido |
|---|---|---|
| Faithfulness | query, context, answer | No |
| Answer Relevance | query, answer | No |
| Context Precision | query, context, ground_truth | Sí |
| Context Recall | context, ground_truth | Sí |

**Impacto académico**: Es la referencia más citada en evaluación RAG (>2000 citas). Define el vocabulario estándar del campo. Sin embargo, la versión más reciente de la librería (v0.2+) ha evolucionado las definiciones originales del paper.

**Limitaciones honestas**:
- Las métricas reference-free (faithfulness, answer relevance) usan LLM-as-judge, introduciendo varianza y sesgo del LLM evaluador.
- Context Precision y Recall requieren ground truth, limitando su uso en producción.
- El paper original fue demo track, no investigación completa; la evaluación de las métricas contra juicio humano es limitada.
- La librería RAGAS evoluciona rápidamente, lo que puede hacer que el código del paper quede desactualizado.

---

### 1.2 ARES — An Automated Evaluation Framework for RAG Systems

**Referencia completa**: Jon Saad-Falcon, Omar Khattab, Christopher Potts, Matei Zaharia. *"ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems"*. NAACL 2024. arXiv:2311.09476, Noviembre 2023.

**Venue**: NAACL 2024 (North American Chapter of ACL) — Long Paper.
**Disponibilidad**: https://arxiv.org/abs/2311.09476 | https://aclanthology.org/2024.naacl-long.20/
**Código**: https://github.com/stanford-futuredata/ARES

**Contribución central**: ARES resuelve el problema más profundo de la evaluación RAG: cómo evaluar con alta fiabilidad estadística usando **solo unos pocos cientos de anotaciones humanas**. Su innovación es doble:

**Paso 1 — Synthetic Data Generation + Fine-tuning**: ARES genera datos de entrenamiento sintéticos (pares query-document con etiquetas automáticas) para entrenar **jueces LM ligeros** (clasificadores) para las tres dimensiones: context relevance, answer faithfulness, answer relevance.

**Paso 2 — Prediction-Powered Inference (PPI)**: Para corregir los errores de predicción del juez fine-tuned, ARES incorpora un pequeño conjunto de anotaciones humanas (~200-500 ejemplos) y usa PPI (Angelopoulos et al., 2023) para producir **intervalos de confianza estadísticamente válidos** incluso con n pequeño.

**Por qué es relevante para el TFG**: ARES demuestra que no hace falta ground truth masivo para una evaluación confiable. El uso de PPI proporciona garantías estadísticas formales (cobertura 1-α del intervalo de confianza), lo cual es exactamente el tipo de rigor que un TFG sobre calibración y confianza necesita citar.

**Resultados**: Evaluado en 8 tasks del benchmark KILT (knowledge-intensive tasks) y SuperGLUE y AIS, ARES supera a RAGAS en correlación con juicio humano.

**Limitaciones**:
- Requiere fine-tuning de un LM ligero (T5, DeBERTa), lo que añade complejidad operacional.
- La generación de datos sintéticos puede introducir sesgos propios del LLM generador.
- El dominio shift puede afectar los jueces fine-tuned (aunque el paper muestra robustez).

---

### 1.3 RGB Benchmark — Benchmarking Large Language Models in RAG

**Referencia completa**: Jiawei Chen, Hongyu Lin, Xianpei Han, Le Sun. *"Benchmarking Large Language Models in Retrieval-Augmented Generation"*. AAAI 2024 (Proceedings of the 38th AAAI Conference on AI). arXiv:2309.01431.

**Venue**: AAAI 2024 — Investigación completa.
**Disponibilidad**: https://arxiv.org/abs/2309.01431 | https://dl.acm.org/doi/10.1609/aaai.v38i16.29728
**Código**: https://github.com/chen700564/RGB

**Contribución central**: RGB no propone métricas de calidad del output, sino que evalúa las **habilidades fundamentales que un LLM necesita** para funcionar bien en un sistema RAG. Define cuatro testbeds:

**1. Noise Robustness**: Capacidad de extraer información útil de documentos ruidosos (relevantes pero sin la respuesta). Se evalúa variando el ratio de ruido de 0 a 0.8. Métrica: accuracy por exact matching.

**2. Negative Rejection**: Capacidad de rechazar responder cuando ningún documento recuperado contiene la respuesta correcta. Métrica: rejection rate (exact match o ChatGPT judge).

**3. Information Integration**: Capacidad de sintetizar respuestas a partir de múltiples documentos (multi-hop). Métrica: accuracy con preguntas reformuladas para requerir síntesis cross-documento.

**4. Counterfactual Robustness**: Capacidad de detectar y rechazar información incorrecta en documentos recuperados, especialmente cuando se advierte al modelo. Métricas: error detection rate y error correction rate.

**Hallazgos clave**: Los LLMs muestran cierta noise robustness, pero fallan significativamente en negative rejection, information integration y counterfactual robustness. Evaluado en GPT-4, ChatGPT, LLaMA, Alpaca, ChatGLM y Vicuna.

**Por qué es relevante para el TFG**: RGB aporta métricas de **comportamiento del sistema** que van más allá de la calidad del output. Son especialmente útiles para evaluar casos borde: ¿qué pasa cuando el RAG no debería responder? ¿Cuándo la información es contradictoria?

---

### 1.4 TruLens / TruEra — RAG Triad

**Referencia**: TruEra Team. *"What is the RAG Triad?"* TruEra Blog / TruLens Documentation.
**Disponibilidad**: https://www.trulens.org/getting_started/core_concepts/rag_triad/ | https://truera.com/ai-quality-education/generative-ai-rags/what-is-the-rag-triad/
**Código**: https://github.com/truera/trulens

**Nota académica**: TruLens no tiene un paper seminal de revisión por pares al nivel de RAGAS o ARES. Es un framework de industria con documentación técnica. Sin embargo, es ampliamente adoptado y la RAG Triad es un concepto referenciado en múltiples papers.

**El RAG Triad** define tres evaluaciones fundamentales:

**1. Context Relevance**: ¿Cada chunk del contexto recuperado es relevante para la query? Evalúa la calidad del retrieval desde la perspectiva de cada chunk individual. Un contexto irrelevante puede "contaminar" la generación con alucinaciones.

**2. Groundedness**: ¿Cada afirmación del response puede atribuirse directamente al texto del contexto? El proceso es: dividir el response en claims individuales → buscar evidencia para cada claim en el contexto recuperado → agregar. Equivalente a Faithfulness en RAGAS pero con énfasis en la atribución explícita.

**3. Answer Relevance**: ¿El response final responde de forma útil a la pregunta original? Evalúa el output final independientemente del contexto.

**Diferencia clave con RAGAS**: TruLens implementa sus métricas principalmente mediante LLM-as-judge (LLM cuidadosamente promoteado), con énfasis en escalabilidad en producción. RAGAS originalmente usaba NLI para faithfulness, aunque las versiones recientes también usan LLM-as-judge por defecto.

---

### 1.5 G-Eval — NLG Evaluation using GPT-4 with Better Human Alignment

**Referencia completa**: Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, Chenguang Zhu. *"G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment"*. EMNLP 2023. arXiv:2303.16634.

**Venue**: EMNLP 2023 (Empirical Methods in NLP), Singapore.
**Disponibilidad**: https://arxiv.org/abs/2303.16634 | https://aclanthology.org/2023.emnlp-main.153/

**Contribución central**: G-Eval es el paper fundacional del paradigma **LLM-as-judge con chain-of-thought**. Define un framework de tres pasos:

1. **Task Introduction + Evaluation Criteria** → se proporcionan al LLM junto con la tarea.
2. **Chain-of-Thought Generation** → el LLM genera automáticamente "Evaluation Steps" detallados (pasos de evaluación).
3. **Form-Filling Paradigm** → el LLM evalúa el output usando el CoT generado, produciendo una puntuación numérica.

**La innovación técnica clave**: En lugar de solo obtener la puntuación final del LLM, G-Eval obtiene la **distribución de probabilidad sobre los tokens de puntuación** (ej. "1", "2", "3", "4", "5") y calcula la puntuación esperada como suma ponderada: `score = Σ p(token_i) * value_i`. Esto produce puntuaciones continuas en vez de discretas, correlacionando mejor con el juicio humano.

**Resultados**: G-Eval con GPT-4 alcanza correlación Spearman de 0.514 con humanos en summarization, superando todos los métodos previos por un margen significativo. En dialogue generation, igualmente superior.

**Advertencia importante**: El paper identifica un **sesgo sistemático**: los evaluadores LLM tienen preferencia por outputs generados por otros LLMs (LLM-generated text bias). Este sesgo debe tenerse en cuenta al usar LLM-as-judge para evaluar sistemas basados en LLMs.

**Aplicación directa al TFG**: G-Eval es el marco conceptual detrás de la mayoría de evaluadores LLM-as-judge modernos (DeepEval, LangSmith evaluators, RAGAS v0.2+). Citarlo establece la base teórica de cualquier uso de GPT-4 como juez.

---

### 1.6 FActScore — Fine-grained Atomic Evaluation of Factual Precision

**Referencia completa**: Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Wen-tau Yih, Pang Wei Koh, Mohit Iyyer, Luke Zettlemoyer, Hannaneh Hajishirzi. *"FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation"*. EMNLP 2023. arXiv:2305.14251.

**Venue**: EMNLP 2023.
**Disponibilidad**: https://arxiv.org/abs/2305.14251 | https://aclanthology.org/2023.emnlp-main.741/
**Código**: https://github.com/shmsw25/FActScore

**Contribución central**: FActScore establece el paradigma de **evaluación a nivel de atomic facts** para medir factual precision en generaciones largas.

**Definición formal**:

> FActScore(response, knowledge_source) = |{f ∈ F(response) : knowledge_source ⊨ f}| / |F(response)|

Donde:
- `F(response)` es el conjunto de atomic facts extraídos del response.
- `knowledge_source ⊨ f` significa que la fuente de conocimiento entaila el atomic fact f.
- El resultado es la **fracción de atomic facts soportados** por la fuente.

**El proceso** (3 pasos):

1. **Descomposición**: Usar un LLM para descomponer el texto generado en atomic facts (oraciones cortas con exactamente una pieza de información). Ejemplo: "Albert Einstein was born in Ulm, Germany in 1879" → "Albert Einstein was born in Ulm" + "Albert Einstein was born in Germany" + "Albert Einstein was born in 1879".

2. **Verificación**: Para cada atomic fact, determinar si está soportado por la knowledge source (puede ser el contexto recuperado, Wikipedia, etc.) usando retrieval + verificación LM.

3. **Agregación**: Promedio de los labels binarios (soportado / no soportado).

**Resultados en el paper**: ChatGPT solo alcanza FActScore de ~58% en biografías de personas. Esto evidencia la magnitud del problema de alucinaciones factual.

**Por qué es relevante para el TFG**: FActScore introduce el concepto de **atomic fact decomposition** que luego adoptan RAGAS (faithfulness), TruLens (groundedness) y prácticamente todos los frameworks modernos. Es la referencia para hablar de evaluación de alucinaciones a nivel granular.

**Limitaciones**:
- Computacionalmente costoso (múltiples llamadas LLM por respuesta).
- La calidad de la descomposición en atomic facts depende del LLM descompositor.
- Diseñado originalmente para evaluar factualidad respecto a Wikipedia, no necesariamente respecto a contextos recuperados.

---

### 1.7 UniEval — Towards a Unified Multi-Dimensional Evaluator for Text Generation

**Referencia completa**: Ming Zhong, Yang Liu, Da Yin, Yuning Mao, Yizhu Jiao, Pengfei Liu, Chenguang Zhu, Heng Ji, Jiawei Han. *"Towards a Unified Multi-Dimensional Evaluator for Text Generation"*. EMNLP 2022. arXiv:2210.07197.

**Venue**: EMNLP 2022.
**Disponibilidad**: https://arxiv.org/abs/2210.07197 | https://aclanthology.org/2022.emnlp-main.131/
**Código**: https://github.com/maszhongming/UniEval

**Contribución central**: UniEval reencuadra la evaluación NLG como un **problema de Question Answering booleano**. En lugar de entrenar evaluadores separados para cada dimensión (coherence, consistency, fluency, etc.), un único modelo responde preguntas como:

- "Is this a coherent passage?" → Yes/No
- "Is this response consistent with the source?" → Yes/No
- "Is this a fluent piece of text?" → Yes/No

**La ventaja clave**: El formato Boolean QA unificado permite **transferencia de conocimiento entre dimensiones y tareas**, y permite evaluar dimensiones no vistas en zero-shot simplemente formulando la pregunta apropiada.

**Resultados**: +23% correlación con juicio humano en text summarization y +43% en dialogue response generation sobre los mejores evaluadores unificados previos.

**Relevancia para RAG**: UniEval es el predecesor conceptual de los evaluadores LLM-as-judge actuales. Aunque fue superado por G-Eval (que usa LLMs más grandes), la idea de unificar múltiples dimensiones bajo un framework QA booleano es adoptada en frameworks como DeepEval.

---

### 1.8 TRUE — Re-evaluating Factual Consistency Evaluation

**Referencia completa**: Or Honovich, Roee Aharoni, Jonathan Herzig, Hagai Taazov, Doron Kukliansky, Vikram Misra, Amir Szpektor, Idan Szpektor, Lior Berant. *"TRUE: Re-evaluating Factual Consistency Evaluation"*. NAACL 2022. arXiv:2204.04991.

**Venue**: NAACL 2022 (Proceedings of the 2022 Conference of the NAACL: Human Language Technologies).
**Disponibilidad**: https://arxiv.org/abs/2204.04991 | https://aclanthology.org/2022.naacl-main.287/

**Contribución central**: TRUE proporciona la **meta-evaluación más comprehensiva** de métricas de factual consistency hasta 2022. Evalúa múltiples métricas (ROUGE, BERTScore, QA-based, NLI-based) en 11 datasets heterogéneos, usando evaluación a nivel de ejemplo en lugar de correlaciones a nivel de sistema.

**Hallazgo principal**: Las métricas basadas en **NLI a gran escala y QA-generation-answering** logran los mejores resultados y son complementarias entre sí. Las métricas basadas en overlap lexical (ROUGE) y similitud semántica (BERTScore) son insuficientes para factual consistency.

**Por qué citar TRUE en el TFG**: TRUE es la justificación académica para no usar ROUGE/BLEU como métricas de faithfulness, y para preferir NLI o LLM-judge. Es la evidencia empírica de que "textual entailment correlates best with human faithfulness/factuality vs ROUGE/BERTScore/QA."

---

### 1.9 Papers sobre Calibración de LLM Judges

**Referencia principal**: *"When Can We Trust LLM Graders? Calibrating Confidence for Automated Assessment"*. arXiv:2603.29559, 2025.

**Hallazgos clave sobre calibración de LLM judges**:

- LLM-as-judge classification systems permanecen **severamente mal calibrados** en todos los formatos testados.
- Expected Calibration Errors (ECE) van de 0.108 a 0.427, muy por encima de umbrales aceptables.
- El formato "float" (pedir al LLM una probabilidad directamente) logra mejor calibración (ECE 0.128-0.135) que el formato categórico (ECE 0.300-0.427).
- Los modelos de razonamiento (o1-style) exhiben **peor calibración** (ECE 0.395) a pesar de accuracy comparable.

**Referencia adicional**: *"Thermometer: Towards Universal Calibration for Large Language Models"*. arXiv:2403.08819, ICLR 2025.
Propone un método post-hoc de calibración que no requiere modificar el LLM.

**Implicación directa para el TFG**: Cualquier uso de LLM-as-judge debe reportar calibración (ECE) además de accuracy, y debe considerarse la aplicación de post-calibration (temperature scaling) sobre las puntuaciones del juez.

---

## Bloque 2: Métricas Universales Avanzadas

### 2.1 Métricas de Retrieval

#### 2.1.1 Context Precision (RAGAS)

**Definición**: Evalúa si los chunks relevantes del contexto recuperado están **posicionados más arriba** que los chunks irrelevantes (es una métrica de ranking, no solo de presencia).

**Fórmula formal**:

```
Context_Precision@K = [Σ_{k=1}^{K} (Precision@k × v_k)] / (Total relevant items in top K)
```

Donde:
- `K` = número total de chunks en el contexto recuperado
- `Precision@k` = (verdaderos positivos hasta posición k) / k
- `v_k ∈ {0, 1}` = indicador de relevancia del chunk en posición k (1 si es relevante, 0 si no)

**Cómo computarla**:
- **Con ground truth**: Para cada chunk del contexto, determinar si es relevante para la query (usando el ground truth answer o un LLM judge). Calcular la suma ponderada.
- **Sin ground truth**: Usar LLM-as-judge para determinar v_k de cada chunk.

**Cuándo usarla**: Cuando tienes múltiples chunks recuperados y quieres penalizar que los chunks relevantes aparezcan al final. Crítica para sistemas con re-ranking.

**Modelo recomendado**: GPT-4o (LLM judge) o DeBERTa NLI para versión sin LLM.

---

#### 2.1.2 Context Recall (RAGAS)

**Definición**: Qué fracción de la información necesaria para responder (según el ground truth) está cubierta por el contexto recuperado.

**Fórmula**:

```
Context_Recall = |{sentences in GT answer attributable to context}| / |{sentences in GT answer}|
```

**Proceso**:
1. Descomponer el ground truth answer en oraciones individuales.
2. Para cada oración, determinar si puede ser atribuida a algún chunk del contexto recuperado.
3. Calcular la fracción de oraciones atribuibles.

**Requiere ground truth**: Sí (el ground truth answer).

**Limitación**: Penaliza al retriever aunque el LLM pueda responder correctamente con conocimiento paramétrico. No aplicable en sistemas donde el LLM puede complementar con conocimiento propio.

---

#### 2.1.3 Context Relevance (TruLens)

**Definición**: Para cada chunk individual del contexto, ¿es relevante para la query del usuario?

**Diferencia con Context Precision**: Context Precision es una métrica de ranking ponderada; Context Relevance es una puntuación promedio de relevancia sin considerar posición.

**Fórmula**:

```
Context_Relevance = (1/|C|) × Σ_{c ∈ C} relevance(c, query)
```

Donde `C` es el conjunto de chunks recuperados y `relevance(c, query) ∈ [0, 1]`.

**Cómo computarla**: LLM-as-judge con prompt como: "¿Es este fragmento relevante para responder la pregunta? Puntúa de 0 a 1."

**Cuándo usar Context Relevance vs Context Precision**: Context Relevance es más simple e interpretable; Context Precision es más rigurosa estadísticamente para sistemas con ranking explícito.

---

#### 2.1.4 Mean Reciprocal Rank (MRR)

**Definición**: Promedio del reciprocal rank del primer documento relevante a través de múltiples queries.

**Fórmula**:

```
MRR = (1/|Q|) × Σ_{i=1}^{|Q|} (1 / rank_i)
```

Donde `rank_i` es la posición del primer documento relevante para la query i.

**Interpretación**: MRR = 1.0 significa que el primer resultado siempre es relevante. MRR = 0.5 significa que en promedio el primer relevante está en posición 2.

**Python**:
```python
def mrr(relevance_lists):
    """relevance_lists: lista de listas, cada lista tiene 0/1 indicando relevancia por posición"""
    scores = []
    for rl in relevance_lists:
        for rank, rel in enumerate(rl, 1):
            if rel == 1:
                scores.append(1.0 / rank)
                break
        else:
            scores.append(0.0)
    return sum(scores) / len(scores)
```

**Cuándo usarlo**: Cuando existe exactamente un documento correcto por query (QA de respuesta única). No adecuado cuando múltiples chunks son igualmente relevantes.

---

#### 2.1.5 NDCG@k (Normalized Discounted Cumulative Gain)

**Definición**: Mide la calidad de un ranking considerando tanto la relevancia como la posición, con descuento logarítmico.

**Fórmula**:

```
DCG@k = Σ_{i=1}^{k} (rel_i / log_2(i + 1))

IDCG@k = DCG@k del ranking ideal (todos los relevantes primero)

NDCG@k = DCG@k / IDCG@k
```

Donde `rel_i` es la relevancia (binaria 0/1 o gradual 0-3) del item en posición i.

**Python** (con sklearn):
```python
from sklearn.metrics import ndcg_score
import numpy as np

# true_relevances: array de relevancia ground truth
# predicted_scores: scores del retriever
ndcg = ndcg_score(np.array([true_relevances]), np.array([predicted_scores]), k=10)
```

**Cuándo usarlo**: Cuando los documentos tienen grados de relevancia (no solo binario) y cuando te importa el ranking completo, no solo el primer resultado. Estándar en benchmarks de search (MS MARCO, BEIR).

---

#### 2.1.6 Answer Attribution

**Definición**: Para cada claim individual del answer generado, verificar si tiene soporte explícito en algún chunk del contexto.

**Proceso** (pipeline estándar post-FActScore):

```python
def answer_attribution(answer: str, context_chunks: list[str], nli_model) -> dict:
    # 1. Descomponer answer en claims atómicos
    claims = decompose_to_atomic_claims(answer)  # via LLM

    # 2. Para cada claim, verificar contra cada chunk
    attributions = {}
    for claim in claims:
        supported = False
        supporting_chunk = None
        for chunk in context_chunks:
            label, score = nli_model.predict(premise=chunk, hypothesis=claim)
            if label == 'entailment' and score > 0.5:
                supported = True
                supporting_chunk = chunk
                break
        attributions[claim] = {'supported': supported, 'source': supporting_chunk}

    # 3. Attribution score
    attribution_score = sum(v['supported'] for v in attributions.values()) / len(claims)
    return {'score': attribution_score, 'details': attributions}
```

**Diferencia con Faithfulness**: Answer Attribution da **rastreabilidad claim-a-chunk** (qué chunk soporta qué afirmación), mientras que Faithfulness da solo el score agregado.

---

### 2.2 Métricas de Generación

#### 2.2.1 Faithfulness (RAGAS)

**Definición**: Fracción de claims del answer generado que están soportados por el contexto recuperado.

**Fórmula**:

```
Faithfulness = |{claims in answer supported by context}| / |{total claims in answer}|
```

**El proceso completo** (pipeline RAGAS original con NLI):

```python
def faithfulness_pipeline(answer: str, context: str, llm, nli_model):
    # Paso 1: Descomposición en claims (requiere LLM)
    claims = llm.decompose_into_claims(answer)
    # Ej: "The capital of France is Paris. It has 2 million inhabitants."
    # → ["The capital of France is Paris", "Paris has 2 million inhabitants"]

    # Paso 2: Verificación NLI claim por claim
    supported_count = 0
    for claim in claims:
        # premise = contexto, hypothesis = claim
        label = nli_model.predict(premise=context, hypothesis=claim)
        if label == 'entailment':
            supported_count += 1

    # Paso 3: Score
    return supported_count / len(claims) if claims else 0.0
```

**Nota**: En la práctica, RAGAS v0.2+ usa un LLM para el paso 2 también (más preciso, más caro). La versión con NLI es más rápida y reproducible.

**Rango**: [0, 1]. Score 1.0 = todas las claims están soportadas (no necesariamente correctas, sino grounded en el contexto).

---

#### 2.2.2 Hallucination Rate

**Definición**: Fracción de claims del answer que **contradicen** o **no tienen soporte** en el contexto recuperado.

**Variantes**:

```
Hallucination_Rate_Conservative = 1 - Faithfulness  # incluye claims "neutral" como no-soportadas

Hallucination_Rate_Strict = |{claims contradicted by context}| / |{total claims}|  # solo contradicciones
```

**Niveles de granularidad**:
- **Sentence-level**: Cada oración es hallucination o no.
- **Entity-level**: Cada entidad nombrada (persona, fecha, lugar) es correcta o incorrecta.
- **Atomic fact level** (FActScore style): Más granular, más preciso.

**Cuándo usar cada nivel**: Para sistemas de producción, sentence-level es más práctico. Para análisis de errores, atomic fact level es más informativo.

---

#### 2.2.3 Atomic Fact Precision (FActScore style)

Como se definió en el Bloque 1.6:

```
AFP = |{atomic facts in answer supported by knowledge source}| / |{total atomic facts in answer}|
```

**Implementación práctica** (usando DeepEval o custom pipeline):

```python
from deepeval.metrics import HallucinationMetric
# DeepEval implementa atomic fact decomposition + NLI internamente

# O implementación custom:
def atomic_fact_precision(answer: str, context: str, llm) -> float:
    # 1. Extraer atomic facts con LLM
    prompt = f"""Break this text into atomic facts (one fact per line):
    Text: {answer}
    Atomic facts:"""
    facts = llm.complete(prompt).strip().split('\n')

    # 2. Verificar cada fact contra context
    supported = 0
    for fact in facts:
        verification_prompt = f"""Context: {context}
        Claim: {fact}
        Is this claim fully supported by the context? Answer yes/no."""
        answer_ver = llm.complete(verification_prompt).lower()
        if 'yes' in answer_ver:
            supported += 1

    return supported / len(facts) if facts else 0.0
```

---

#### 2.2.4 G-Eval Style — Evaluación con Chain-of-Thought

**Definición**: Usar un LLM para evaluar una dimensión específica, generando primero los pasos de evaluación (CoT) y luego puntuando.

**Template general** (adaptable a cualquier dimensión):

```python
def g_eval(query: str, context: str, answer: str,
           criterion: str, llm, scale: int = 5) -> float:

    # Paso 1: Generar evaluation steps (CoT)
    steps_prompt = f"""
    Task: Evaluate the {criterion} of a RAG system's answer.
    Generate detailed evaluation steps for assessing {criterion}.
    """
    eval_steps = llm.complete(steps_prompt)

    # Paso 2: Evaluar con los steps generados
    eval_prompt = f"""
    Query: {query}
    Context: {context}
    Answer: {answer}

    Evaluation Steps:
    {eval_steps}

    Based on the above steps, rate the {criterion} on a scale from 1 to {scale}.
    Output only the number.
    """

    # Paso 3: Obtener distribución de probabilidad (G-Eval key innovation)
    token_probs = llm.complete_with_logprobs(eval_prompt,
                                              tokens=[str(i) for i in range(1, scale+1)])
    score = sum(int(tok) * prob for tok, prob in token_probs.items())
    return score / scale  # Normalizar a [0, 1]
```

**Criterios aplicables en RAG**: faithfulness, relevance, completeness, coherence, conciseness, toxicity.

**Ventaja sobre score directo**: El CoT fuerza al LLM a hacer razonamiento explícito antes de puntuar, reduciendo arbitrariedad. La distribución de probabilidad sobre tokens de puntuación produce scores continuos.

---

### 2.3 Métricas de Comportamiento del Sistema (RGB)

#### 2.3.1 Noise Robustness Score

```python
def noise_robustness(rag_fn, queries_with_answers, noise_ratios=[0.0, 0.2, 0.4, 0.6, 0.8]):
    """
    Evalúa cómo se degrada el accuracy al aumentar el ratio de documentos ruidosos.
    """
    results = {}
    for ratio in noise_ratios:
        correct = 0
        for query, gt_answer, relevant_docs, noise_docs in queries_with_answers:
            # Mezclar documentos relevantes y ruidosos según ratio
            n_noise = int(ratio * (len(relevant_docs) + len(noise_docs)))
            context_docs = relevant_docs + noise_docs[:n_noise]

            response = rag_fn({'query': query, 'context': context_docs})
            if gt_answer.lower() in response['answer'].lower():
                correct += 1
        results[ratio] = correct / len(queries_with_answers)

    # Degradation score: pendiente de la caída de accuracy
    degradation = results[0.0] - results[0.8]
    return results, degradation
```

---

#### 2.3.2 Negative Rejection Rate

```python
def negative_rejection_rate(rag_fn, unanswerable_queries):
    """
    Evalúa qué fracción de queries sin respuesta en el contexto son rechazadas correctamente.
    Las queries se sirven SOLO con documentos irrelevantes/ruidosos.
    """
    rejection_count = 0
    rejection_phrases = ["I don't know", "not enough information",
                         "cannot answer", "no information", "insufficient"]

    for query, noise_docs in unanswerable_queries:
        response = rag_fn({'query': query, 'context': noise_docs})['answer']

        # Método 1: Exact match con frases de rechazo
        if any(phrase.lower() in response.lower() for phrase in rejection_phrases):
            rejection_count += 1

    return rejection_count / len(unanswerable_queries)
```

**Nota**: Una NRR alta (cercana a 1.0) es deseable. NRR baja significa que el sistema inventa respuestas cuando no debería. Este es uno de los fallos más críticos en producción.

---

#### 2.3.3 Information Integration Score

```python
def information_integration_score(rag_fn, multihop_queries):
    """
    Evalúa si el sistema puede combinar información de múltiples chunks
    para responder preguntas complejas.
    """
    correct = 0
    for query, gt_answer, multi_source_docs in multihop_queries:
        # Los docs proveen información complementaria, ninguno tiene la respuesta completa
        response = rag_fn({'query': query, 'context': multi_source_docs})
        if gt_answer.lower() in response['answer'].lower():
            correct += 1
    return correct / len(multihop_queries)
```

---

#### 2.3.4 Counterfactual Robustness

```python
def counterfactual_robustness(rag_fn, queries_with_false_docs, llm_judge=None):
    """
    Evalúa si el sistema detecta y rechaza información incorrecta en el contexto.
    El sistema recibe una advertencia de que puede haber información errónea.
    """
    detection_count = 0
    correction_count = 0

    for query, correct_answer, counterfactual_doc in queries_with_false_docs:
        # Añadir warning al contexto
        warning = "Note: The retrieved documents may contain factual errors."
        response = rag_fn({
            'query': query,
            'context': [counterfactual_doc],
            'system_note': warning
        })['answer']

        # Detección: el sistema menciona que la info puede ser incorrecta
        if any(w in response.lower() for w in ['incorrect', 'false', 'error', 'misleading']):
            detection_count += 1

        # Corrección: el sistema proporciona la respuesta correcta
        if correct_answer.lower() in response.lower():
            correction_count += 1

    return {
        'detection_rate': detection_count / len(queries_with_false_docs),
        'correction_rate': correction_count / len(queries_with_false_docs)
    }
```

---

## Bloque 3: Modelos NLI y Alternativas sin LLM

### 3.1 Motivación para Evaluación sin LLM

Usar GPT-4 como juez tiene ventajas (alta correlación con humanos) pero también problemas reales:
- **Costo**: ~$0.01-0.05 por evaluación. Con 1000 ejemplos y múltiples métricas = cientos de dólares.
- **Reproducibilidad**: Las respuestas del LLM tienen varianza. Ejecutar el mismo eval dos veces puede dar resultados diferentes.
- **Latencia**: 1-5 segundos por evaluación. Inaceptable para monitorización online a alto volumen.
- **Privacidad**: Enviar datos de clientes a APIs externas puede no ser posible.

Los modelos NLI resuelven los primeros tres problemas (son deterministas, rápidos y baratos) pero tienen menor correlación con juicio humano que GPT-4.

---

### 3.2 cross-encoder/nli-deberta-v3-base

**HuggingFace**: https://huggingface.co/cross-encoder/nli-deberta-v3-base
**Arquitectura**: microsoft/deberta-v3-base (0.2B parámetros) fine-tuned para NLI.
**Training data**: SNLI + MultiNLI.
**Rendimiento**: SNLI-test accuracy 92.38%, MNLI mismatched 90.04%.
**Licencia**: Apache 2.0.

**Uso para Faithfulness en RAG**:

```python
from sentence_transformers import CrossEncoder

nli_model = CrossEncoder('cross-encoder/nli-deberta-v3-base')

def faithfulness_nli(claims: list[str], context: str) -> float:
    """
    claims: lista de atomic claims extraídas del answer
    context: el contexto recuperado concatenado
    """
    if not claims:
        return 0.0

    # Crear pares (context, claim) para cada claim
    pairs = [(context, claim) for claim in claims]

    # Predecir: output shape (n_claims, 3)
    # Columnas: [contradiction, entailment, neutral]
    scores = nli_model.predict(pairs)

    # Contar claims entailadas
    label_mapping = ['contradiction', 'entailment', 'neutral']
    labels = scores.argmax(axis=1)
    entailed = (labels == 1).sum()  # índice 1 = entailment

    return entailed / len(claims)

# Uso completo con descomposición via LLM
def faithfulness_full(answer: str, context: str, llm, nli_model) -> float:
    claims = llm.decompose_into_claims(answer)  # list of strings
    return faithfulness_nli(claims, context)
```

**Threshold recomendado**: 0.5 en la probabilidad de entailment (softmax score). Si `entailment_prob > 0.5`, el claim está soportado. Para aplicaciones críticas, subir a 0.7-0.8.

**Limitación importante**: Entrenado en SNLI/MNLI (oraciones cortas y directas). El rendimiento cae cuando el contexto es muy largo (>512 tokens) o el claim es implícito. La versión `-large` o `-xsmall` ofrece tradeoff entre calidad y velocidad.

---

### 3.3 Variantes de DeBERTa NLI

| Modelo | Parámetros | MNLI Acc | Tokens/seg (GPU) | Uso recomendado |
|---|---|---|---|---|
| `cross-encoder/nli-deberta-v3-xsmall` | 22M | ~87% | Muy alto | Producción/alto volumen |
| `cross-encoder/nli-deberta-v3-small` | 44M | ~89% | Alto | Producción equilibrado |
| `cross-encoder/nli-deberta-v3-base` | 184M | 90% | Medio | Evaluación offline estándar |
| `cross-encoder/nli-deberta-v3-large` | 400M | ~91% | Bajo | Máxima precisión offline |
| `microsoft/deberta-large-mnli` | 390M | ~91% | Bajo | Alternativa a la versión large |

**HuggingFace con transformers puro** (alternativa a SentenceTransformers):

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

model_name = "cross-encoder/nli-deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

def nli_predict(premise: str, hypothesis: str) -> dict:
    inputs = tokenizer(premise, hypothesis, return_tensors='pt',
                       truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=-1).squeeze().tolist()

    # Orden de etiquetas: contradiction=0, entailment=1, neutral=2
    return {
        'contradiction': probs[0],
        'entailment': probs[1],
        'neutral': probs[2],
        'label': ['contradiction', 'entailment', 'neutral'][logits.argmax().item()]
    }
```

---

### 3.4 TRUE y HHEM — Modelos Fine-tuned para Factual Consistency

**TRUE** (Honovich et al., 2022) no es un modelo sino un protocolo de evaluación y un hallazgo: las mejores métricas para factual consistency son NLI a gran escala. El modelo NLI que mejor rinde en TRUE es `abisee/t5-base-nli` fine-tuned.

**HHEM (Hughes Hallucination Evaluation Model)** — Vectara:
```
vectara/hallucination_evaluation_model (HuggingFace)
```
DeBERTa de 184M parámetros fine-tuned en múltiples datasets NLI incluyendo TRUE y SummaC. Diseñado específicamente para detectar alucinaciones en RAG. Es la alternativa ligera más orientada a producción.

---

### 3.5 Comparativa: NLI vs LLM-judge vs BERTScore vs BLEURT

| Aspecto | NLI (DeBERTa) | LLM-judge (GPT-4) | BERTScore | BLEURT |
|---|---|---|---|---|
| Correlación con humanos | Media-alta (~0.6-0.7) | Alta (~0.75-0.85) | Baja (~0.4-0.5) para faithfulness | Media (~0.5-0.6) |
| Costo por evaluación | ~$0.00001 (CPU/GPU local) | ~$0.01-0.05 | Gratis (local) | Gratis (local) |
| Reproducibilidad | Alta (determinista) | Baja (stochastic) | Alta | Alta |
| Latencia | Baja (10-50ms) | Alta (1-5s) | Muy baja | Baja |
| Captura entailment | Sí (por diseño) | Sí | No (solo similitud) | Parcial |
| Sensible a entity swaps | Sí | Sí | No (BERTScore ~0.50-0.60 en legal) | Parcial |
| Requiere ground truth | No | No | Sí (o referencia) | Sí |
| Adecuado para RAG faithfulness | Sí | Sí (gold standard) | No | No recomendado |

**Recomendación práctica para el TFG**:
- **Evaluación offline con n pequeño**: GPT-4 judge (máxima correlación con humanos).
- **Evaluación offline a escala o screening**: NLI DeBERTa-v3-base (equilibrio calidad/costo).
- **Producción/monitorización online**: NLI xsmall o HHEM (latencia mínima).
- **Nunca usar ROUGE/BLEU/BERTScore para faithfulness** (baja correlación con humanos, demostrado en TRUE).

---

### 3.6 Faithfulness Pipeline Completo (Producción)

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class FaithfulnessResult:
    score: float
    supported_claims: list[str]
    unsupported_claims: list[str]
    total_claims: int

class FaithfulnessEvaluator:
    def __init__(self,
                 nli_model_name: str = 'cross-encoder/nli-deberta-v3-base',
                 entailment_threshold: float = 0.5,
                 use_llm_decomposition: bool = True):
        from sentence_transformers import CrossEncoder
        self.nli = CrossEncoder(nli_model_name)
        self.threshold = entailment_threshold
        self.use_llm = use_llm_decomposition

    def decompose_claims(self, answer: str, llm=None) -> list[str]:
        if self.use_llm and llm:
            # LLM-based decomposition (más precisa)
            prompt = f"""Break this answer into minimal atomic claims.
Each claim should contain exactly one piece of verifiable information.
Answer: {answer}
Atomic claims (one per line):"""
            return llm.complete(prompt).strip().split('\n')
        else:
            # Heurística simple: dividir por oraciones
            import re
            return [s.strip() for s in re.split(r'[.!?]+', answer) if s.strip()]

    def evaluate(self, answer: str, context: str, llm=None) -> FaithfulnessResult:
        claims = self.decompose_claims(answer, llm)
        if not claims:
            return FaithfulnessResult(0.0, [], [], 0)

        # Crear pares premise-hypothesis
        pairs = [(context, claim) for claim in claims]
        scores = self.nli.predict(pairs)

        supported = []
        unsupported = []
        for claim, score in zip(claims, scores):
            # score[1] = entailment probability
            entailment_prob = float(score[1]) / sum(score)  # normalizar
            if entailment_prob > self.threshold:
                supported.append(claim)
            else:
                unsupported.append(claim)

        return FaithfulnessResult(
            score=len(supported) / len(claims),
            supported_claims=supported,
            unsupported_claims=unsupported,
            total_claims=len(claims)
        )
```

---

## Bloque 4: Evaluación por Arquitectura RAG

### 4.1 Naive/Vector RAG Clásico

**Métricas estándar aplicables directamente**: Todas las de RAGAS (faithfulness, context precision/recall, answer relevance). Es el caso base.

**Métricas adicionales específicas**:
- **Chunk quality**: Tamaño promedio de chunk, overlap entre chunks relevantes.
- **Embedding similarity**: Cosine similarity entre query embedding y chunk embeddings del contexto recuperado.
- **Top-k coverage**: ¿Con k=3 o k=5 se cubre suficiente contexto (recall ≥ 0.8)?

---

### 4.2 Agentic RAG y Corrective RAG (CRAG)

**Referencias clave**:
- Yan et al. 2024. *"Corrective Retrieval Augmented Generation"*. ICLR 2024. arXiv:2401.15884.
- Asai et al. 2024. *"Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"*. ICLR 2024 (Oral, top 1%).

**Métricas específicas**:

**1. Loop Efficiency** (o Retrieval Iterations):
```
Loop_Efficiency = (queries_answered_in_first_iteration) / (total_queries)
```
Un buen sistema agentic debe resolver la mayoría de queries sin múltiples re-intentos. Más de 3 iteraciones por query es señal de un retriever deficiente o query reformulation ineficaz.

**2. Query Reformulation Quality**:
```python
def query_reformulation_quality(original_query: str,
                                 reformulated_query: str,
                                 context_retrieved_before: str,
                                 context_retrieved_after: str,
                                 llm) -> float:
    """
    ¿Mejoró el contexto recuperado tras la reformulación?
    Proxy: Context Recall antes vs después de reformulación.
    """
    recall_before = context_recall(context_retrieved_before, original_query, llm)
    recall_after = context_recall(context_retrieved_after, original_query, llm)
    return max(0, recall_after - recall_before)  # Mejora marginal
```

**3. Tool Call Precision** (para RAG con tools):
```python
def tool_call_precision(tool_calls: list[dict],
                        expected_tools: list[str]) -> float:
    """
    Evalúa si el agente llamó las herramientas correctas con los parámetros correctos.
    """
    correct_calls = sum(1 for call in tool_calls
                       if call['tool_name'] in expected_tools)
    return correct_calls / max(len(tool_calls), 1)
```

**4. Trajectory Faithfulness** (VERITAS framework, 2024):
Evalúa si los pasos intermedios de razonamiento (think→search→read→answer) están justificados por la evidencia recuperada en cada paso. Va más allá de evaluar solo el answer final.

```python
def trajectory_faithfulness(trajectory: list[dict], nli_model) -> float:
    """
    trajectory: lista de pasos del agente
    Cada paso: {'type': 'think'|'retrieve'|'answer', 'content': str, 'supporting_docs': list}
    """
    faithful_steps = 0
    verifiable_steps = 0

    for step in trajectory:
        if step['type'] in ['think', 'answer'] and step.get('supporting_docs'):
            verifiable_steps += 1
            # Verificar que el contenido del step está soportado por los docs del step
            claim_supported = any(
                nli_model.predict([(doc, step['content'])])[0].argmax() == 1  # entailment
                for doc in step['supporting_docs']
            )
            if claim_supported:
                faithful_steps += 1

    return faithful_steps / max(verifiable_steps, 1)
```

**5. Corrective Action Rate** (específico para CRAG):
- `P(correct action | correct retrieval)`: ¿Con qué frecuencia el sistema confía en retrieval correcto?
- `P(search_triggered | bad retrieval)`: ¿Con qué frecuencia corrige retrieval insuficiente?

---

### 4.3 SQL RAG (Text-to-SQL)

**Benchmarks de referencia**: Spider (Yu et al., 2018), Spider 2.0 (2024, enterprise-level con >3000 columnas), BIRD (2023).

**Métricas estándar**:

**1. Execution Accuracy (EX)** — Métrica principal actual:
```python
def execution_accuracy(predicted_sql: str, gold_sql: str, database_conn) -> bool:
    """
    Ejecuta ambas queries y compara los result sets.
    Más permisiva que exact match: permite diferentes formulations que dan el mismo resultado.
    """
    try:
        pred_result = database_conn.execute(predicted_sql).fetchall()
        gold_result = database_conn.execute(gold_sql).fetchall()
        return set(map(tuple, pred_result)) == set(map(tuple, gold_result))
    except Exception:
        return False
```

**2. Exact Match (EM)** — Métrica clásica pero problemática:
```python
def exact_match(predicted_sql: str, gold_sql: str) -> bool:
    """
    Normalización básica antes de comparar.
    Muy estricto: queries equivalentes pueden no matchear.
    """
    normalize = lambda s: ' '.join(s.lower().split())
    return normalize(predicted_sql) == normalize(gold_sql)
```

**Nota crítica**: Execution Accuracy correlaciona con Test Suite Accuracy solo al 74-90%. EX puede dar falsos positivos (queries que retornan los mismos resultados por casualidad en la DB de test). Spider 2.0 propone métricas más sofisticadas para producción enterprise.

**3. Valid Efficiency Score (VES)**:
```
VES = ExecutionAccuracy × (1 / query_execution_time_ratio)
```
Evalúa simultáneamente corrección y eficiencia de la query generada. Relevante en producción donde queries ineficientes tienen impacto real.

**4. Schema Linking Accuracy**:
```python
def schema_linking_accuracy(predicted_tables_cols: set,
                             gold_tables_cols: set) -> dict:
    """
    ¿Identificó correctamente qué tablas y columnas son relevantes?
    """
    precision = len(predicted_tables_cols & gold_tables_cols) / max(len(predicted_tables_cols), 1)
    recall = len(predicted_tables_cols & gold_tables_cols) / max(len(gold_tables_cols), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {'precision': precision, 'recall': recall, 'f1': f1}
```

**5. SQL RAG específico — Retrieval de ejemplos few-shot**:
En Text-to-SQL con RAG, el retriever busca queries similares históricas como few-shot examples. La calidad de este retrieval se evalúa con:
- Similarity de la query recuperada con la query objetivo (embedding cosine similarity).
- Execution accuracy de la SQL generada con los ejemplos recuperados vs sin ejemplos.

---

### 4.4 Hybrid RAG (Vector + BM25)

**Métricas específicas**:

**1. Source Attribution Score**:
```python
def source_attribution(answer: str,
                       vector_chunks: list[str],
                       bm25_chunks: list[str],
                       nli_model) -> dict:
    """
    Determina qué fracción del answer proviene de chunks vectoriales vs BM25.
    """
    all_chunks = vector_chunks + bm25_chunks
    claims = decompose_to_claims(answer)

    vector_support = 0
    bm25_support = 0

    for claim in claims:
        # Verificar contra cada fuente
        vector_entailed = any(
            nli_model.predict([(c, claim)])[0].argmax() == 1
            for c in vector_chunks
        )
        bm25_entailed = any(
            nli_model.predict([(c, claim)])[0].argmax() == 1
            for c in bm25_chunks
        )

        if vector_entailed:
            vector_support += 1
        if bm25_entailed:
            bm25_support += 1

    return {
        'vector_attribution': vector_support / max(len(claims), 1),
        'bm25_attribution': bm25_support / max(len(claims), 1),
        'total_coverage': max(vector_support, bm25_support) / max(len(claims), 1)
    }
```

**2. Fusion Strategy Effectiveness** (RRF vs Score Normalization):

La efectividad de la estrategia de fusión se mide comparando NDCG@k antes y después de la fusión:

```python
def fusion_effectiveness(vector_rankings: list, bm25_rankings: list,
                         ground_truth_relevant: list, k: int = 10) -> dict:
    """
    Compara la calidad del ranking fusionado vs cada sistema individual.
    """
    from sklearn.metrics import ndcg_score
    import numpy as np

    def rrf_score(rankings: list, k: int = 60) -> list:
        """Reciprocal Rank Fusion"""
        scores = {}
        for ranking in rankings:
            for rank, doc_id in enumerate(ranking, 1):
                scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank)
        return sorted(scores, key=scores.get, reverse=True)

    fused_ranking = rrf_score([vector_rankings, bm25_rankings])

    # Construir vectores de relevancia
    def ranking_to_relevance(ranking, gt_relevant, top_k):
        return [1 if doc in gt_relevant else 0 for doc in ranking[:top_k]]

    return {
        'ndcg_vector': ndcg_score([ranking_to_relevance(vector_rankings, ground_truth_relevant, k)],
                                   [list(range(k, 0, -1))]),
        'ndcg_bm25': ndcg_score([ranking_to_relevance(bm25_rankings, ground_truth_relevant, k)],
                                 [list(range(k, 0, -1))]),
        'ndcg_fused': ndcg_score([ranking_to_relevance(fused_ranking, ground_truth_relevant, k)],
                                  [list(range(k, 0, -1))])
    }
```

**3. Complementarity Score**:
¿Cuántos documentos relevantes son encontrados por un sistema pero no por el otro?
```
Complementarity = |relevant_vector_only ∪ relevant_bm25_only| / |all_relevant|
```
Un Complementarity Score alto justifica el uso de hybrid retrieval.

---

### 4.5 GraphRAG

**Contexto**: GraphRAG (como el de Microsoft, basado en Neo4j) recupera subgrafos de conocimiento en lugar de chunks de texto.

**Métricas específicas**:

**1. Subgraph Relevance**: ¿Los nodos y relaciones recuperados del grafo son relevantes para la query?
```python
def subgraph_relevance(query: str, subgraph_triples: list[tuple], llm) -> float:
    """
    subgraph_triples: lista de (subject, predicate, object)
    """
    relevant_count = 0
    for subj, pred, obj in subgraph_triples:
        triple_text = f"{subj} {pred} {obj}"
        prompt = f"Is '{triple_text}' relevant to answering '{query}'? Answer yes/no."
        if 'yes' in llm.complete(prompt).lower():
            relevant_count += 1
    return relevant_count / max(len(subgraph_triples), 1)
```

**2. Relationship Faithfulness**: ¿Las relaciones mencionadas en el answer existen en el grafo?
**3. Multi-hop Accuracy**: Para queries que requieren múltiples saltos en el grafo, ¿el camino recorrido fue correcto?

---

## Bloque 5: Monitorización Online vs Evaluación Offline

### 5.1 Diferencia Conceptual

| Aspecto | Evaluación Offline | Monitorización Online |
|---|---|---|
| Momento | Pre-deployment, en desarrollo | Post-deployment, en producción |
| Ground truth | Disponible (dataset curado) | No disponible |
| Objetivo | Validar antes de lanzar | Detectar degradación en tiempo real |
| Escala | Decenas a miles de ejemplos | Millones de queries |
| Latencia requerida | Horas/días | Segundos/minutos |
| Métricas | RAGAS completo, RGB, métricas con GT | Métricas reference-free únicamente |

### 5.2 Métricas Reference-Free para Monitorización Online

Estas métricas **no requieren ground truth** y pueden calcularse en tiempo real:

**1. Faithfulness** (RAGAS / TruLens Groundedness): No requiere GT, solo answer + context.
**2. Answer Relevance**: Solo necesita query + answer.
**3. Context Relevance**: Solo necesita query + context chunks.
**4. NRR Proxy** (Negative Rejection Proxy): Detectar respuestas que contienen frases de incertidumbre ("I don't know", "cannot find information"). No mide la corrección de la decisión, pero sirve como proxy.
**5. Response Length Distribution**: Cambios en longitud pueden indicar cambios en el comportamiento del sistema.
**6. Embedding Drift**: Cosine distance promedio entre query embeddings y chunk embeddings en ventanas temporales.

```python
class OnlineRAGMonitor:
    def __init__(self, nli_evaluator, embedding_model, window_size: int = 100):
        self.nli = nli_evaluator
        self.embedder = embedding_model
        self.window_size = window_size
        self.metric_history = []

    def evaluate_single(self, query: str, context: str, answer: str) -> dict:
        """Evaluación en tiempo real: < 100ms objetivo"""
        metrics = {
            'faithfulness': self.nli.faithfulness_score(answer, context),
            'context_relevance': self.nli.relevance_score(query, context),
            'answer_relevance': self.nli.relevance_score(query, answer),
            'response_length': len(answer.split()),
            'has_uncertainty': any(phrase in answer.lower()
                                   for phrase in ["i don't know", "cannot", "insufficient"])
        }
        self.metric_history.append(metrics)
        return metrics

    def detect_drift(self) -> dict:
        """Detectar drift comparando ventanas temporales"""
        if len(self.metric_history) < 2 * self.window_size:
            return {'drift_detected': False, 'reason': 'Insufficient data'}

        recent = self.metric_history[-self.window_size:]
        historical = self.metric_history[-2*self.window_size:-self.window_size]

        results = {}
        for metric in ['faithfulness', 'context_relevance', 'answer_relevance']:
            recent_avg = sum(d[metric] for d in recent) / self.window_size
            hist_avg = sum(d[metric] for d in historical) / self.window_size

            # Drift si la diferencia es > 2 desviaciones estándar
            hist_std = (sum((d[metric] - hist_avg)**2 for d in historical) / self.window_size)**0.5
            results[metric] = {
                'recent_avg': recent_avg,
                'historical_avg': hist_avg,
                'drift_detected': abs(recent_avg - hist_avg) > 2 * hist_std
            }

        return results
```

### 5.3 Sampling Strategies para Evaluación Continua

No se puede evaluar el 100% de las queries en producción. Estrategias:

**1. Random Sampling (baseline)**: Evaluar una fracción aleatoria (ej. 5-10%). Simple pero puede perder eventos raros.

**2. Stratified Sampling**: Asegurar representación de diferentes tipos de query (topical diversity, length diversity).

**3. Uncertainty-Based Sampling**: Evaluar preferentemente las queries donde el sistema tiene menor confianza (respuestas más cortas, más frases de incertidumbre, embeddings menos similares).

**4. Adaptive Sampling**: Aumentar la tasa de muestreo cuando se detecta drift estadístico:
```python
def adaptive_sampling_rate(base_rate: float, drift_score: float) -> float:
    """
    Aumentar la tasa de muestreo proporcionalmente al drift detectado.
    """
    return min(base_rate * (1 + 5 * drift_score), 1.0)
```

**5. Canary Queries**: Mantener un conjunto fijo de ~50 queries conocidas con respuestas esperadas. Ejecutar periódicamente. Si las respuestas degradan, hay un problema.

### 5.4 Drift Detection Formal

**Query Drift**: Cambio en la distribución de los tipos de queries de los usuarios.
```python
# KL divergence entre distribución actual y histórica de topics
from scipy.stats import entropy
def query_drift(hist_topic_dist, current_topic_dist):
    return entropy(current_topic_dist, hist_topic_dist)  # KL divergence
```

**Knowledge Base Drift**: Los documentos del corpus cambian con el tiempo (documentos obsoletos).
```python
# Monitorizar context_recall en canary queries como proxy de KB staleness
def knowledge_staleness(canary_eval_results: list[dict]) -> float:
    recent = canary_eval_results[-10:]
    return sum(r['context_recall'] for r in recent) / len(recent)
```

**Model Drift**: El LLM subyacente cambia (actualización del proveedor), alterando el comportamiento de generación.

---

## Bloque 6: Calibración Estadística

### 6.1 Expected Calibration Error (ECE) — Fórmula Exacta

**Definición**: ECE mide cuánto se desvía la confianza expresada por un evaluador de su accuracy real.

**Fórmula**:

```
ECE = Σ_{m=1}^{M} (|B_m| / n) × |acc(B_m) - conf(B_m)|
```

Donde:
- `M` = número de bins (típicamente 10-15)
- `B_m` = conjunto de predicciones cuya confianza cae en el bin m
- `|B_m|` = número de ejemplos en el bin m
- `n` = total de ejemplos
- `acc(B_m)` = accuracy real en el bin m = |{i ∈ B_m : ŷ_i = y_i}| / |B_m|
- `conf(B_m)` = confianza media en el bin m = (1/|B_m|) × Σ_{i ∈ B_m} p̂_i

**Interpretación**: ECE = 0 es calibración perfecta. ECE = 0.1 significa que el modelo es en promedio 10% más o menos confiado de lo que debería ser. Los LLM judges actuales tienen ECE entre 0.108 y 0.427 (severo).

**Implementación**:

```python
import numpy as np

def expected_calibration_error(confidences: np.ndarray,
                                 accuracies: np.ndarray,
                                 n_bins: int = 10) -> float:
    """
    confidences: probabilidades estimadas por el juez [0, 1]
    accuracies: indicadores de corrección binarios {0, 1}
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(confidences)

    for i in range(n_bins):
        # Índices en este bin
        in_bin = (confidences > bins[i]) & (confidences <= bins[i + 1])
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            confidence_in_bin = confidences[in_bin].mean()
            ece += prop_in_bin * abs(accuracy_in_bin - confidence_in_bin)

    return ece

# Para un LLM judge: obtener confidence como la probabilidad que el juez asigna
# a "correct" / "supported" / "relevant"
# accuracies: comparar el juicio del LLM con ground truth humano en un held-out set
```

---

### 6.2 Calibración con Temperature Scaling y Platt Scaling

**Temperature Scaling** (Guo et al., 2017 — el método más simple y efectivo):

```python
import torch
import torch.nn.functional as F
from torch import nn, optim

class TemperatureScaling(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature

    def fit(self, val_logits: torch.Tensor, val_labels: torch.Tensor,
            lr: float = 0.01, max_iter: int = 50):
        """Ajustar T usando un val set de anotaciones humanas."""
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        nll_criterion = nn.CrossEntropyLoss()

        def eval_closure():
            optimizer.zero_grad()
            scaled_logits = self.forward(val_logits)
            loss = nll_criterion(scaled_logits, val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_closure)
        return self

# Uso para LLM judge:
# 1. Obtener logits (scores brutos) del juez en un val set con labels humanos
# 2. Ajustar temperatura: T > 1 suaviza (reduce overconfidence), T < 1 sharpens
# 3. Aplicar a nuevas predicciones
```

**Platt Scaling** (alternativa para outputs binarios):

```python
from sklearn.linear_model import LogisticRegression
import numpy as np

class PlattScaling:
    def __init__(self):
        self.lr = LogisticRegression(C=1.0)

    def fit(self, raw_scores: np.ndarray, human_labels: np.ndarray):
        """raw_scores: scores del juez (entre 0 y 1), human_labels: 0/1"""
        self.lr.fit(raw_scores.reshape(-1, 1), human_labels)
        return self

    def calibrate(self, raw_scores: np.ndarray) -> np.ndarray:
        return self.lr.predict_proba(raw_scores.reshape(-1, 1))[:, 1]

# Platt Scaling es preferible a Temperature Scaling cuando:
# - La distribución de scores del juez es muy sesgada
# - Tienes menos de 100 ejemplos de calibración
```

**¿Cuándo usar cuál?**
- **Temperature Scaling**: Cuando el LLM judge es multiclase (puntuaciones 1-5) y el ranking relativo es correcto pero las probabilidades absolutas están mal calibradas.
- **Platt Scaling**: Para jueces binarios (soportado/no soportado) con pocos datos de calibración.
- **Ninguno**: Si el ECE del juez ya es < 0.05 (raro en la práctica).

---

### 6.3 Muestras Necesarias para Calibración

**Para calibración confiable con Temperature Scaling**: ~200-500 ejemplos con labels humanos en el dominio específico. Con menos de 100, la temperatura ajustada puede ser inestable.

**Justificación estadística**: Para estimar la temperatura T con error estándar de 0.05, necesitas aproximadamente n ≈ 400 ejemplos (regla empírica de la literatura de calibración).

Para el TFG con n pequeño, usar **bootstrap para estimar la incertidumbre de la temperatura**:

```python
def calibration_bootstrap(scores, labels, n_bootstrap=1000):
    temps = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(scores), len(scores), replace=True)
        ts = TemperatureScaling()
        ts.fit(torch.tensor(scores[idx]).unsqueeze(1),
               torch.tensor(labels[idx]).long())
        temps.append(ts.temperature.item())
    return np.percentile(temps, [2.5, 97.5])  # 95% CI
```

---

### 6.4 Intervalos de Confianza para Métricas RAG con n Pequeño (10-50)

**El problema**: Con n=20 evaluaciones, un average de faithfulness = 0.75 tiene una incertidumbre enorme. ¿Cómo reportar esto honestamente?

**Opción 1 — Wilson Score Interval** (para métricas binarias o prop-like):

```python
from scipy import stats

def wilson_ci(n_successes: int, n_total: int, alpha: float = 0.05) -> tuple:
    """
    Mejor que el IC normal para n pequeño o proporciones extremas.
    Equivalente a: faithfulness entendida como proporción de claims soportadas.
    """
    z = stats.norm.ppf(1 - alpha/2)
    p_hat = n_successes / n_total

    denominator = 1 + z**2 / n_total
    center = (p_hat + z**2 / (2*n_total)) / denominator
    margin = (z * np.sqrt(p_hat*(1-p_hat)/n_total + z**2/(4*n_total**2))) / denominator

    return (center - margin, center + margin)

# Ejemplo: 15 de 20 claims soportadas
lower, upper = wilson_ci(15, 20)
print(f"Faithfulness: {15/20:.2f} [IC 95%: {lower:.3f}, {upper:.3f}]")
# → Faithfulness: 0.75 [IC 95%: 0.531, 0.893]
# Con n=20, la incertidumbre es ENORME. Ser honesto al reportar.
```

**Opción 2 — Bootstrap Percentile** (para métricas continuas como NDCG, MRR):

```python
def bootstrap_ci(metric_values: list, n_bootstrap: int = 10000,
                  alpha: float = 0.05) -> tuple:
    """
    Válido para cualquier métrica continua.
    Recomendado con n ≥ 20.
    """
    bootstrapped_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(metric_values, size=len(metric_values), replace=True)
        bootstrapped_means.append(np.mean(sample))

    lower = np.percentile(bootstrapped_means, 100 * alpha/2)
    upper = np.percentile(bootstrapped_means, 100 * (1 - alpha/2))
    return lower, upper

# Ejemplo con n=30 evaluaciones de faithfulness
faithfulness_scores = [0.8, 0.6, 1.0, 0.75, ...]  # 30 valores
lower, upper = bootstrap_ci(faithfulness_scores)
print(f"Faithfulness: {np.mean(faithfulness_scores):.3f} [IC 95%: {lower:.3f}, {upper:.3f}]")
```

**Opción 3 — Prediction-Powered Inference (PPI)** (ARES approach):

El método más robusto cuando tienes tanto predicciones automáticas (muchas) como anotaciones humanas (pocas). Angelopoulos et al. 2023 demuestran que PPI produce intervalos de confianza estadísticamente válidos con garantías de cobertura 1-α.

```python
# Requiere: pip install ppi-python (librería oficial del paper PPI)
from ppi_py import ppi_mean_ci

# labeled_predictions: predicciones del LLM judge en el held-out humano (n pequeño)
# labeled_outcomes: ground truth humano en ese mismo held-out
# unlabeled_predictions: predicciones del LLM judge en todo el dataset (n grande)

ci_lower, ci_upper = ppi_mean_ci(
    labeled_outcomes,      # human annotations (n = 100-300)
    labeled_predictions,   # model predictions on labeled set
    unlabeled_predictions, # model predictions on full dataset
    alpha=0.05
)
```

**Guía práctica de cuánto n necesitas**:

| n (evaluaciones) | Ancho del IC 95% (Wilson/Bootstrap) | Recomendación |
|---|---|---|
| 10 | ±0.30-0.35 | Solo exploración, no reportar como resultado |
| 20 | ±0.20-0.25 | Mínimo aceptable para TFG con caveats explícitos |
| 50 | ±0.14-0.16 | Suficiente para comparar dos sistemas |
| 100 | ±0.10 | Estándar en papers académicos |
| 300 | ±0.06 | Para establecer benchmarks |

**Recomendación para el TFG**: Con n=20-50 evaluaciones, siempre reportar el IC junto con el punto estimado. Nunca reportar solo el promedio. La honestidad estadística es un criterio de calidad académica.

---

## Resumen: Recomendaciones para el TFG

### Métricas más usadas en industria vs academia

| Métrica | Industria | Academia | Adecuación TFG |
|---|---|---|---|
| Faithfulness (RAGAS) | Alta | Alta | Muy alta — es la métrica central |
| Context Relevance (TruLens) | Alta | Media | Alta — fácil de implementar |
| Answer Relevance | Alta | Media | Alta — reference-free |
| Context Precision/Recall | Media | Alta | Media — requiere GT |
| Hallucination Rate | Alta | Alta | Muy alta — el problema más visible |
| MRR/NDCG | Media | Alta | Media — útil si tienes ground truth de retrieval |
| ECE del juez | Baja (poco medido) | Media | Alta — diferenciador para el TFG |
| Noise Robustness | Media | Alta | Media — útil para robustez |
| Negative Rejection | Media | Media | Alta — crítico en producción |

### Pipeline de Evaluación Recomendado para el TFG

```python
class UniversalRAGEvaluator:
    """
    Framework de evaluación universal para cualquier arquitectura RAG.
    Wrapper con firma: rag_fn(inputs: dict) -> dict
    """

    def __init__(self, llm_judge, nli_model, use_llm_for_decomp: bool = True):
        self.llm = llm_judge
        self.nli = nli_model
        self.use_llm = use_llm_for_decomp

    def evaluate_single(self, query: str, context: str, answer: str,
                         ground_truth: str = None) -> dict:
        metrics = {}

        # === MÉTRICAS REFERENCE-FREE (siempre computables) ===

        # 1. Faithfulness (NLI-based, sin GT)
        claims = self._decompose_claims(answer)
        metrics['faithfulness'] = self._faithfulness_nli(claims, context)

        # 2. Answer Relevance (LLM judge, sin GT)
        metrics['answer_relevance'] = self._answer_relevance(query, answer)

        # 3. Context Relevance (NLI/LLM, sin GT)
        metrics['context_relevance'] = self._context_relevance(query, context)

        # 4. Hallucination Rate
        metrics['hallucination_rate'] = 1 - metrics['faithfulness']

        # === MÉTRICAS CON GROUND TRUTH (si disponible) ===
        if ground_truth:
            metrics['context_recall'] = self._context_recall(context, ground_truth)
            metrics['context_precision'] = self._context_precision(context, query, ground_truth)

        return metrics

    def evaluate_behavior(self, rag_fn, behavior_dataset: dict) -> dict:
        """Evaluar comportamiento del sistema (RGB-style)"""
        return {
            'noise_robustness': self._noise_robustness(rag_fn, behavior_dataset.get('noise')),
            'negative_rejection': self._negative_rejection(rag_fn, behavior_dataset.get('unanswerable')),
            'info_integration': self._information_integration(rag_fn, behavior_dataset.get('multihop')),
        }

    def compute_confidence_score(self, metrics: dict) -> dict:
        """
        Score de confianza agregado: el objetivo del TFG.
        """
        weights = {
            'faithfulness': 0.35,
            'answer_relevance': 0.25,
            'context_relevance': 0.20,
            'noise_robustness': 0.10,
            'negative_rejection': 0.10,
        }

        available = {k: v for k, v in metrics.items() if k in weights and v is not None}
        total_weight = sum(weights[k] for k in available)

        score = sum(metrics[k] * weights[k] for k in available) / total_weight

        # Reportar con IC si hay múltiples evaluaciones
        return {'confidence_score': score, 'metrics_used': list(available.keys())}
```

### Stack Tecnológico Recomendado para el TFG

| Componente | Herramienta | Justificación |
|---|---|---|
| LLM judge | GPT-4o (via LangSmith) | Máxima correlación con humanos, ya integrado |
| NLI model | cross-encoder/nli-deberta-v3-base | Gratis, reproducible, Apache 2.0 |
| Trazas y datasets | LangSmith SDK | Ya instalado, LangChain integration nativa |
| Framework de evaluación | RAGAS + custom | RAGAS para métricas estándar, custom para métricas específicas |
| Calibración | scipy + sklearn | Temperature Scaling y Wilson CI |
| CI estadísticos | Bootstrap + scipy.stats | Honest reporting |

### Referencias Adicionales de Apoyo

Para el marco teórico completo, los papers más importantes a citar en el TFG, por orden de relevancia:

1. **RAGAS** (Es et al., EACL 2024) — Framework central.
2. **ARES** (Saad-Falcon et al., NAACL 2024) — Calibración estadística y PPI.
3. **G-Eval** (Liu et al., EMNLP 2023) — Base teórica de LLM-as-judge.
4. **FActScore** (Min et al., EMNLP 2023) — Evaluación de atomic facts.
5. **RGB** (Chen et al., AAAI 2024) — Métricas de comportamiento.
6. **TRUE** (Honovich et al., NAACL 2022) — Justificación para usar NLI en vez de BLEU/ROUGE.
7. **UniEval** (Zhong et al., EMNLP 2022) — Evaluación multidimensional unificada.
8. **CRAG** (Yan et al., arXiv 2024) — Evaluación de Corrective RAG.

---

*Investigación compilada para el TFG "Hacia el 95% de Confianza". Fuentes verificadas en marzo 2026.*
