# rag-eval — Framework de Evaluación Universal para Sistemas RAG

**TFG:** "Hacia el 95% de Confianza: Monitorización y Calibración de Agentes de IA basados en RAG"
**Universidad:** ICAI, Grado en Ingeniería Matemática e IA
**Colaboración:** Stratesys

---

## ¿Qué es esto?

`rag-eval` **no** construye RAGs. Es una herramienta de evaluación externa: tomas cualquier sistema RAG existente, lo envuelves en una función `fn(inputs: dict) -> dict`, y el framework lo evalúa con métricas académicas, captura trazas en LangSmith, y da un score de confianza calibrado.

```
Tu RAG ──► wrapper ──► rag-eval ──► métricas + LangSmith + calibración ECE
```

## Arquitecturas soportadas

| Arquitectura | Wrapper disponible | Evaluadores específicos |
|---|---|---|
| Agentic RAG (LangGraph) | `notebooks/01_agentic_rag.ipynb` | loop efficiency, query reformulation |
| GraphRAG (Neo4j + Cypher) | `rag_eval/wrappers/graphrag_neo4j.py` | cypher complexity, schema adherence, multihop |
| GraphRAG naive (baseline) | `rag_eval/wrappers/graphrag_naive.py` | mismo esquema, sin validación |
| Cualquier RAG | firma estándar `fn(dict) -> dict` | evaluadores universales |

## Métricas implementadas

### Universales (`rag_eval/evaluators/universal.py`)

| Métrica | Tipo | Paper de referencia |
|---|---|---|
| `faithfulness_nli` | NLI DeBERTa, sin LLM | TRUE (NAACL 2022), RAGAS (EACL 2024) |
| `hallucination_rate` | derivada (1 − faithfulness) | — |
| `atomic_fact_precision` | LLM decompose + NLI verify | FActScore (EMNLP 2023) |
| `context_precision_at_k` | ranking ponderado de chunks | RAGAS (EACL 2024) |
| `context_recall` | cobertura del GT en contexto | RAGAS (EACL 2024) |
| `context_relevance` | relevancia chunks vs query | TruLens RAG Triad |
| `answer_relevance_universal` | LLM-judge G-Eval style | G-Eval (EMNLP 2023) |
| `correctness_universal` | LLM-judge vs ground truth | G-Eval (EMNLP 2023) |
| `negative_rejection` | anti-alucinación RGB | RGB Benchmark (AAAI 2024) |
| `confidence_score_universal` | score compuesto calibrado | ARES (NAACL 2024) |

### GraphRAG-específicas (`rag_eval/evaluators/graphrag.py`)

`cypher_generated`, `cypher_result_nonempty`, `empty_context_hallucination`, `schema_adherence`, `cypher_complexity_score`, `relationship_direction_score`, `multihop_required_detector`, `multihop_execution_score`, `failure_mode_classifier`, `answer_completeness`, `confidence_score_v2`

### Calibración

`compute_ece` (ECE), `temperature_scaling`, `find_optimal_temperature`, `compute_calibration_report`

## Estructura del repo

```
rag-eval/
├── rag_eval/                        # Paquete principal
│   ├── evaluators/
│   │   ├── universal.py             # Evaluadores universales (10 métricas + calibración)
│   │   ├── graphrag.py              # Evaluadores GraphRAG avanzados (14 métricas)
│   │   └── base.py                  # Evaluadores base GraphRAG (9 métricas)
│   ├── wrappers/
│   │   ├── graphrag_neo4j.py        # Wrapper Neo4j (validate_cypher=True)
│   │   └── graphrag_naive.py        # Wrapper naive (baseline, sin validación)
│   └── datasets/
│       └── northwind.py             # 31 preguntas Northwind en 7 categorías
│
├── notebooks/
│   ├── 01_agentic_rag.ipynb         # Agentic RAG (LangGraph) + evaluación
│   └── 03_evaluacion_neo4j_rag.ipynb # GraphRAG Neo4j completo + comparación + calibración
│
├── docs/
│   ├── research_universal_rag_evaluation.md   # 9 papers + 12 métricas con fórmulas
│   ├── research_graphrag_evaluation.md        # Evaluación específica GraphRAG
│   └── research_evaluadores_graphrag_avanzados.md
│
├── neo4j_graphrag_tutorial/         # Tutorial externo (referencia, datos CSV)
├── tests/
├── pyproject.toml
├── requirements.txt
└── .env.example
```

## Quickstart

```bash
# 1. Clonar e instalar
git clone <repo>
cd rag-eval
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configurar credenciales
cp .env.example .env
# Editar .env con OPENAI_API_KEY, LANGCHAIN_API_KEY, NEO4J_PASSWORD

# 3. Evaluar cualquier RAG
python - << 'EOF'
from rag_eval.evaluators.universal import evaluate_rag_universal
from rag_eval.datasets.northwind import DATASET_NORTHWIND

def mi_rag(inputs: dict) -> dict:
    return {"answer": "...", "context": "..."}

results = evaluate_rag_universal(
    rag_fn=mi_rag,
    dataset=DATASET_NORTHWIND,
    dataset_name="mi-primera-eval",
    project="01-agentic-rag",
    preset="default",        # "nli_only" (sin LLM) | "default" | "full"
)
EOF
```

## Dataset Northwind

31 preguntas verificadas contra Neo4j en 7 categorías:

| Categoría | N | Descripción |
|---|---|---|
| A. Lookups directos | 6 | 1-hop, baseline |
| B. Agregaciones | 8 | COUNT, AVG, MAX |
| C. Multi-hop 2 saltos | 5 | 2 relaciones |
| D. Multi-hop 3+ saltos | 3 | máxima complejidad |
| E. Sin respuesta | 4 | test anti-alucinación |
| F. Filtros complejos | 3 | múltiples WHERE |
| G. Jerarquía | 2 | REPORTS_TO recursivo |

## Resultados experimentales (GraphRAG Principal vs Naive)

| Métrica | Principal | Naive | Delta |
|---|---|---|---|
| Sin alucinación silenciosa | 1.000 | 0.839 | +0.161 |
| Resultados no vacíos | 0.839 | 0.774 | +0.065 |
| Schema adherence | 1.000 | 0.989 | +0.011 |
| Score estructural | 0.892 | 0.819 | +0.073 |
| Correctness | 0.581 | 0.581 | = |
| Groundedness | 0.806 | 0.710 | +0.097 |

**Hallazgo:** correctness igual en ambos (0.581) — el bottleneck es el razonamiento multi-hop, no la validación de Cypher.

## Notas técnicas

- `bolt://localhost:7687` para Neo4j (no `neo4j://` — causa routing error)
- `max_concurrency=1` obligatorio (rate limit OpenAI 30k TPM)
- NLI model: `cross-encoder/nli-deberta-v3-base` — retorna **logits**, usar softmax (no sum)
- Los archivos raíz (`rag_evaluator.py`, `universal_rag_evaluator.py`, etc.) son shims de compatibilidad para los notebooks originales

## Referencias académicas clave

- **RAGAS** — Es et al., EACL 2024 · arXiv:2309.15217
- **ARES** — Saad-Falcon et al., NAACL 2024 · arXiv:2311.09476
- **RGB Benchmark** — Chen et al., AAAI 2024 · arXiv:2309.01431
- **FActScore** — Min et al., EMNLP 2023 · arXiv:2305.14251
- **G-Eval** — Liu et al., EMNLP 2023 · arXiv:2303.16634
- **TRUE** — Honovich et al., NAACL 2022 · arXiv:2204.04991
