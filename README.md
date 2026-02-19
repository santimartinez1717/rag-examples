# Framework de Evaluación de Sistemas RAG

Framework para analizar, evaluar, monitorizar y calibrar diferentes arquitecturas RAG construido sobre LangSmith.

## 🎯 Objetivo

Evaluar y comparar diferentes arquitecturas RAG (básicas, GraphRAG, multiagente, rerank, SQL, etc.) usando evaluadores universales y específicos por arquitectura.

## 📁 Estructura Actual

```
rag_example/
├── rag_evaluator.py                  # ⭐ Evaluador universal (FUNCIONA)
├── graphrag_wrapper_standalone.py    # ⭐ Wrapper GraphRAG standalone (FUNCIONA)
├── 01_agentic_rag.ipynb              # ✅ RAG Agéntico (EVALUADO)
├── 03_evaluacion_neo4j_rag.ipynb     # ✅ GraphRAG Neo4j (EVALUADO)
├── neo4j_graphrag_tutorial/          # Código del tutorial GraphRAG + datos
├── simple_rag_example.py             # RAG simple de prueba
└── run_notebook.py                   # Script para ejecutar notebooks
```

## ✅ Lo que YA funciona

### 1. Evaluador Universal ([rag_evaluator.py](rag_evaluator.py))

Evaluador agnóstico que funciona con cualquier arquitectura RAG:

**Métricas incluidas:**
- ✅ `correctness` - Exactitud vs ground truth
- ✅ `relevance` - Relevancia de la respuesta
- ✅ `groundedness` - Fidelidad al contexto recuperado
- ✅ `retrieval_relevance` - Calidad de los documentos recuperados

**Uso:**

```python
from rag_evaluator import evaluate_rag

# Tu función RAG (debe recibir dict y devolver dict)
def mi_rag(inputs: dict) -> dict:
    # ... tu lógica ...
    return {"answer": "...", "documents": [...]}

# Dataset de evaluación
dataset = [
    {
        "inputs": {"question": "¿Qué es un LLM?"},
        "outputs": {"answer": "Un modelo de lenguaje grande..."}
    },
    # ... más ejemplos ...
]

# Evaluar
results = evaluate_rag(
    mi_rag,
    dataset,
    dataset_name="mi-dataset",
    project="mi-proyecto"
)
```

### 2. RAG Agéntico ([01_agentic_rag.ipynb](01_agentic_rag.ipynb))

**Estado:** ✅ **FUNCIONA PERFECTAMENTE**

- Implementa un Agentic RAG con LangGraph
- Incluye: retrieve → grade → rewrite → generate
- Documentos: 3 blog posts de Lilian Weng sobre LLMs
- **Ya integrado con el evaluador universal** (ver última celda del notebook)
- Evaluadores custom adicionales: `answer_not_empty`, `conciseness`, `has_context`, `answer_relevance`, `faithfulness`

**Cómo ejecutar:**
1. Abre el notebook `01_agentic_rag.ipynb`
2. Ejecuta todas las celdas
3. La última celda ejecuta la evaluación automáticamente

### 3. GraphRAG Neo4j ([03_evaluacion_neo4j_rag.ipynb](03_evaluacion_neo4j_rag.ipynb))

**Estado:** ✅ **FUNCIONA PERFECTAMENTE**

- Implementa GraphRAG usando Neo4j como base de datos de grafos
- Genera queries Cypher automáticamente usando LLMs
- Datos: Dataset Northwind (empleados, productos, órdenes)
- **Ya integrado con el evaluador universal**
- Wrapper standalone que evita conflictos de dependencias

**Configuración (ya completada):**
1. ✅ Neo4j Desktop instalado con APOC plugin
2. ✅ Datos cargados (9 empleados, productos, órdenes, etc.)
3. ✅ Credenciales configuradas en `.env`
4. ✅ Wrapper standalone funcionando

**Cómo ejecutar:**
1. Asegúrate de que Neo4j esté corriendo (debe estar en "Active")
2. Ejecuta las celdas del notebook
3. La evaluación corre automáticamente con 4 métricas universales

## 🚀 Próximos pasos

### Arquitecturas RAG a implementar:

- [ ] **MultiAgent RAG** - Coordinación de agentes especializados
- [ ] **Rerank RAG** - Mejora del retrieval con reranking
- [ ] **SQL RAG** - Text-to-SQL sobre bases de datos
- [ ] **Hybrid RAG** - Combinación de vector + keyword search

### Mejoras del framework:

- [ ] Métricas específicas por arquitectura (graph quality, agent coordination, etc.)
- [ ] Dashboard de monitorización en tiempo real
- [ ] Sistema de calibración y optimización automática
- [ ] Comparación automática entre arquitecturas
- [ ] Exportación de reportes

## 📊 Resultados de Evaluación

Todos los resultados se guardan en LangSmith: https://smith.langchain.com

Para ver tus evaluaciones:
1. Ve a https://smith.langchain.com
2. Busca el proyecto configurado en `LANGCHAIN_PROJECT`
3. Explora datasets y experimentos

## 🔧 Debugging

Si algo no funciona:

```bash
python debug_graphrag.py
```

Este script verifica:
- Variables de entorno
- Imports del wrapper
- Conexión a Neo4j
- Evaluadores

## 📝 Notas

- El evaluador universal ya funciona con el RAG agéntico
- GraphRAG requiere Neo4j configurado
- Todas las API keys están en `.env`
- LangSmith tracing está activado para todas las ejecuciones

## 🎓 Referencias

- LangSmith RAG Evaluation: https://docs.langchain.com/langsmith/evaluate-rag-tutorial
- LangGraph: https://langchain-ai.github.io/langgraph/
- Neo4j Graph RAG: https://neo4j.com/labs/genai-ecosystem/langchain/
