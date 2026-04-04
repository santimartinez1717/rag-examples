# PLANNING — Framework de Evaluación RAG

**TFG:** "Hacia el 95% de Confianza: Monitorización y Calibración de Agentes de IA basados en RAG"
**Estado:** F4 activa (24 mar – 13 abr)

---

## Objetivo del framework (recordatorio)

Tomar **cualquier RAG existente** (de GitHub, producción, Stratesys), envolverlo en `fn(dict) -> dict`,
y producir:
1. Métricas individuales (correctness, faithfulness, etc.)
2. Un **score único calibrado 0–100%** comparable entre arquitecturas distintas

La aportación central no es construir RAGs — es evaluarlos de forma comparable.

---

## Preguntas estructurales resueltas

### ¿Cómo almaceno los RAGs que evalúo?
No se "almacenan". Cada RAG es un **wrapper** en `rag_eval/wrappers/` que implementa la firma:
```python
def mi_rag(inputs: dict) -> dict:
    return {"answer": str, "context": str | list}
```
El wrapper es el código. La configuración va en `.env` o en el propio wrapper.

### ¿Cómo almaceno los datasets?
Dos capas:
- **Local**: `rag_eval/datasets/` como archivos Python (importables, versionados en git)
- **LangSmith**: cada dataset se sube con nombre único (`client.create_dataset`) y persiste ahí
LangSmith ES el almacén de referencia. No duplicar resultados en CSV locales.

### ¿Cómo almaceno los resultados/métricas?
En **LangSmith como experiments**. Cada `client.evaluate()` crea un experimento con nombre,
métricas por ejemplo, y trazas completas. Se comparan visualmente en la UI o programáticamente.

### ¿Tengo que correrlo en un notebook?
No necesariamente. La estructura ideal:
- `notebooks/` → exploración, presentación, análisis visual
- `scripts/run_eval.py` → ejecución reproducible desde CLI (para comparar arquitecturas)
- Los dos usan exactamente el mismo código de `rag_eval/`

### ¿Tengo que almacenar la arquitectura del RAG?
Sí, como **metadata en LangSmith**. Cada experimento tiene un campo `metadata` donde
pones `{"architecture": "GraphRAG-LangChain", "llm": "gpt-4.1", "dataset": "northwind"}`.
Eso permite filtrar y comparar por arquitectura en la UI.

---

## Epics y User Stories

---

### EPIC 1 — GraphRAG: Espectro de calidad demostrable

**Objetivo:** Demostrar que las métricas tienen poder discriminativo mostrando que un RAG
deliberadamente malo puntúa bajo y uno bueno puntúa alto.

**Por qué es necesario:** Main vs Naive da correctness=0.581 en ambos — no demuestra nada.
Necesitamos RAGs donde sepamos de antemano cuál es mejor.

---

**US-G1** *(alta prioridad)*
> Como evaluador, quiero un wrapper `graphrag_no_context` que ignora los resultados de Neo4j
> y responde solo con conocimiento del LLM, para validar que `faithfulness_nli` y
> `hallucination_rate` detectan correctamente la alucinación.

**Criterio de aceptación:** `hallucination_rate` ≥ 0.7 en este wrapper vs ≤ 0.3 en el principal.

---

**US-G2** *(alta prioridad)*
> Como evaluador, quiero un wrapper `graphrag_always_refuse` que siempre devuelve
> "I don't have information about that", para validar que `negative_rejection` y
> `correctness` detectan un sistema que nunca responde correctamente.

**Criterio de aceptación:** `correctness` ≈ 0.0, `negative_rejection` ≈ 1.0 en preguntas sin respuesta.

---

**US-G3** *(media prioridad)*
> Como investigador, quiero evaluar `neo4j-graphrag-python` (la librería oficial de Neo4j,
> distinta de LangChain) sobre el dataset Northwind, para comparar dos implementaciones
> distintas del mismo patrón arquitectónico.

**Criterio de aceptación:** Wrapper funcionando, experimento en LangSmith con metadata
`{"architecture": "GraphRAG-Neo4j-native"}`.

---

**US-G4** *(media prioridad)*
> Como investigador, quiero un wrapper basado en LangGraph (agente con nodo de
> generación Cypher + nodo de síntesis) en vez de GraphCypherQAChain, para comparar
> una chain fija contra un agente con control de flujo explícito.

---

**US-G5** *(media prioridad)*
> Como investigador, quiero un segundo dataset sobre el grafo de **películas** (Movies)
> que viene de ejemplo con Neo4j, para demostrar que el framework evalúa GraphRAG
> independientemente del dominio.

**Criterio de aceptación:** 20+ preguntas verificadas contra el grafo Movies, experimento
en LangSmith separado del Northwind.

---

### EPIC 2 — Métricas: Validación y refinamiento

**Objetivo:** Que cada métrica esté justificada empíricamente, no solo teóricamente.

---

**US-M1** *(alta prioridad)*
> Como investigador, quiero una tabla de "discriminative power" que muestre los scores
> de cada métrica en los distintos wrappers (bueno, malo, roto), para tener evidencia
> empírica de qué métricas sirven y cuáles no.

**Formato esperado:**
```
                    | faithfulness | correctness | hallucination | negative_rejection
--------------------|--------------|-------------|---------------|-------------------
graphrag_main       |    0.85      |    0.58     |     0.15      |       1.0
graphrag_naive      |    0.70      |    0.58     |     0.30      |       1.0
graphrag_no_context |    0.10      |    0.05     |     0.90      |       0.0
graphrag_refuse     |    0.50      |    0.00     |     0.50      |       1.0
```

---

**US-M2** *(alta prioridad)*
> Como investigador, quiero reemplazar `correctness` binario (0/1) por un score continuo
> [0,1] que capture respuestas parcialmente correctas, porque con 31 preguntas la
> varianza binaria es demasiado alta.

**Criterio de aceptación:** LLM-judge devuelve score 0.0/0.5/1.0 en vez de bool.
Correlación con binario > 0.8.

---

**US-M3** *(media prioridad)*
> Como investigador, quiero validar los scores del LLM-judge contra anotación humana
> en 15 ejemplos seleccionados del dataset Northwind, usando LangSmith Annotation Queues,
> para calcular el ECE real y reportarlo en la memoria.

**Criterio de aceptación:** Anotación de 15 ejemplos, ECE calculado, temperatura
óptima aplicada.

---

**US-M4** *(baja prioridad)*
> Como investigador, quiero una métrica `cypher_execution_success_rate` que mida
> qué porcentaje de los Cypher generados se ejecutan sin error (independientemente
> del resultado), como indicador de robustez sintáctica.

---

### EPIC 3 — Confidence Score: Único calibrado comparable

**Objetivo:** Un solo número 0–100% que tenga significado real y sea comparable
entre arquitecturas distintas. Ésta es la aportación central del TFG.

---

**US-C1** *(alta prioridad)*
> Como investigador, quiero redefinir el confidence score usando pesos aprendidos
> (regresión logística sobre las métricas) en vez de pesos arbitrarios, para que
> sea defendible académicamente.

**Criterio de aceptación:** ECE del nuevo score < ECE del score actual.
Entrenado sobre los experimentos disponibles con `correctness` como label.

---

**US-C2** *(alta prioridad)*
> Como usuario del framework, quiero que el confidence score sea el mismo cálculo
> para GraphRAG, Agentic RAG y cualquier otra arquitectura, para que los números
> sean comparables entre sí.

**Criterio de aceptación:** Mismo `confidence_score_universal()` produce scores
para todos los wrappers. Tabla comparativa entre arquitecturas en LangSmith.

---

**US-C3** *(media prioridad)*
> Como investigador, quiero aplicar temperature scaling al confidence score y
> reportar ECE antes y después en la memoria, para demostrar que la calibración
> mejora la fiabilidad del score.

---

### EPIC 4 — Estructura del proyecto

**Objetivo:** Un repo que cualquier developer pueda clonar, entender en 5 minutos
y usar para evaluar su propio RAG.

---

**US-E1** *(alta prioridad)*
> Como developer externo, quiero un script `scripts/run_eval.py` que tome un
> wrapper y un dataset como argumentos y ejecute la evaluación completa,
> para poder evaluar sin abrir un notebook.

```bash
python scripts/run_eval.py \
  --wrapper rag_eval.wrappers.graphrag_neo4j \
  --dataset rag_eval.datasets.northwind \
  --preset default \
  --experiment-name "graphrag-main-v1"
```

---

**US-E2** *(alta prioridad)*
> Como developer, quiero que cada experimento en LangSmith tenga metadata
> estructurada `{architecture, wrapper, dataset, llm, preset, timestamp}`,
> para poder filtrar y comparar experimentos por arquitectura.

---

**US-E3** *(media prioridad)*
> Como investigador, quiero un notebook `notebooks/04_comparison.ipynb` que
> cargue múltiples experimentos de LangSmith y genere la tabla comparativa
> entre arquitecturas automáticamente.

---

**US-E4** *(media prioridad)*
> Como investigador, quiero que los datasets en LangSmith tengan versiones
> explícitas (`northwind-v1`, `northwind-v2`) para poder reproducir experimentos
> pasados exactamente.

---

**US-E5** *(baja prioridad)*
> Como developer, quiero un `Makefile` con targets `eval-graphrag`, `eval-agentic`,
> `compare`, `calibrate` para ejecutar los flujos más comunes sin recordar los
> comandos exactos.

---

### EPIC 5 — LangSmith: Aprovechar todo lo disponible

**Objetivo:** LangSmith como núcleo real del framework, no solo como log viewer.

---

**US-L1** *(alta prioridad)*
> Como investigador, quiero usar LangSmith `compare_experiments` para mostrar
> side-by-side los scores de todas las arquitecturas en una sola llamada,
> en vez de calcular manualmente.

---

**US-L2** *(alta prioridad)*
> Como investigador, quiero crear una **Annotation Queue** en LangSmith con
> 15 ejemplos del experimento GraphRAG, asignarla a mí mismo, y anotar
> manualmente si la respuesta es correcta, para validar el LLM-judge.

---

**US-L3** *(media prioridad)*
> Como investigador, quiero guardar los prompts de los evaluadores LLM en el
> **LangSmith Prompt Hub**, para versionarlos y poder referenciarlos en la memoria.

---

**US-L4** *(media prioridad)*
> Como investigador, quiero configurar **online monitoring** en LangSmith
> (trazas en tiempo real con alertas si hallucination_rate > 0.3), para
> demostrar el uso del framework en producción.

---

**US-L5** *(baja prioridad)*
> Como investigador, quiero crear un dataset en LangSmith **a partir de trazas**
> reales (ejemplos donde el sistema falló), para iterar sobre casos difíciles.

---

## Orden de implementación recomendado

```
Semana 1 (ahora):
  US-G1  → wrapper graphrag_no_context      (valida faithfulness/hallucination)
  US-G2  → wrapper graphrag_always_refuse   (valida negative_rejection/correctness)
  US-M1  → tabla discriminative power       (evidencia empírica inmediata)
  US-E2  → metadata en experimentos         (fácil, mejora todo lo siguiente)

Semana 2:
  US-M2  → correctness continuo             (mejora métrica core)
  US-C1  → confidence score con pesos aprendidos
  US-E1  → script run_eval.py

Semana 3:
  US-G3  → neo4j-graphrag-python wrapper
  US-L1  → compare_experiments LangSmith
  US-E3  → notebook comparación

Semana 4 (F5 — Stratesys):
  US-M3  → annotation queues + ECE real
  US-C3  → temperature scaling reportado
  US-L4  → online monitoring
```

---

## Lo que NO hacer

- No construir RAGs desde cero — tomar implementaciones existentes
- No duplicar resultados en CSV locales — LangSmith es el storage
- No añadir más métricas sin antes validar las que hay (US-M1 primero)
- No sobre-ingenierizar la estructura — wrapper + dataset + LangSmith es suficiente
