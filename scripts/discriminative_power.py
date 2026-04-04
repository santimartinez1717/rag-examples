"""
US-M1: Tabla de Poder Discriminativo

Ejecuta los 4 wrappers GraphRAG sobre el dataset Northwind con el preset
DISCRIMINATIVE_EVALUATORS y genera una tabla comparativa que demuestra
que las métricas distinguen correctamente entre RAGs buenos y malos.

Wrappers evaluados:
  graphrag_main         → validate_cypher=True (el mejor)
  graphrag_naive        → validate_cypher=False (baseline)
  graphrag_no_context   → ignora Neo4j, alucina (US-G1)
  graphrag_always_refuse→ siempre rechaza (US-G2)

Resultado esperado (del PLANNING.md US-M1):
                    | faithfulness | correctness | hallucination | negative_rejection
--------------------|--------------|-------------|---------------|-------------------
graphrag_main       |    0.85      |    0.58     |     0.15      |       1.0
graphrag_naive      |    0.70      |    0.58     |     0.30      |       1.0
graphrag_no_context |    0.10      |    0.05     |     0.90      |       0.0
graphrag_always_refuse|  0.50      |    0.00     |     0.50      |       1.0

Uso:
    python scripts/discriminative_power.py
    python scripts/discriminative_power.py --subset 10  # solo 10 ejemplos (test rápido)
"""

import sys
import os
import argparse
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")


def run_wrapper_eval(wrapper_fn, wrapper_name: str, dataset: list, dataset_name: str,
                     architecture: str, llm: str):
    """Ejecuta la evaluación de un wrapper y devuelve el objeto results."""
    from rag_eval.evaluators.universal import evaluate_rag_universal, DISCRIMINATIVE_EVALUATORS

    print(f"\n{'─' * 50}")
    print(f"  Evaluando: {wrapper_name}")
    print(f"{'─' * 50}")

    results = evaluate_rag_universal(
        rag_fn=wrapper_fn,
        dataset=dataset,
        dataset_name=dataset_name,
        project="graphrag-discriminative-power",
        evaluators=DISCRIMINATIVE_EVALUATORS,
        experiment_name=f"discriminative-{wrapper_name}",
        metadata={
            "architecture": architecture,
            "wrapper":      wrapper_name,
            "dataset":      dataset_name,
            "llm":          llm,
            "experiment_type": "discriminative_power_table",
        },
        max_concurrency=1,
    )
    return results


def extract_metrics(results) -> dict:
    """Extrae métricas promedio del objeto ExperimentResults."""
    try:
        df = results.to_pandas()
        metrics = {}
        score_cols = [c for c in df.columns if c.startswith("feedback.")]
        for col in score_cols:
            metric_name = col.replace("feedback.", "")
            valid = df[col].dropna()
            if len(valid) > 0:
                metrics[metric_name] = round(float(valid.mean()), 3)
        return metrics
    except Exception as e:
        print(f"⚠️  No se pudieron extraer métricas: {e}")
        return {}


def print_table(all_metrics: dict):
    """Imprime la tabla de poder discriminativo."""
    KEY_METRICS = [
        "faithfulness_nli",
        "hallucination_rate",
        "correctness_continuous",
        "negative_rejection",
    ]

    wrappers = list(all_metrics.keys())

    # Header
    col_w = 22
    header = f"{'Wrapper':<25}" + "".join(f"{m:<{col_w}}" for m in KEY_METRICS)
    print(f"\n{'═' * (25 + col_w * len(KEY_METRICS))}")
    print(f"  TABLA DE PODER DISCRIMINATIVO")
    print(f"{'═' * (25 + col_w * len(KEY_METRICS))}")
    print(header)
    print("─" * (25 + col_w * len(KEY_METRICS)))

    for wrapper_name, metrics in all_metrics.items():
        row = f"{wrapper_name:<25}"
        for m in KEY_METRICS:
            val = metrics.get(m)
            if val is None:
                row += f"{'N/A':<{col_w}}"
            else:
                row += f"{val:<{col_w}.3f}"
        print(row)

    print(f"{'═' * (25 + col_w * len(KEY_METRICS))}")
    print()

    # Diagnóstico
    print("  Diagnóstico de poder discriminativo:")
    print("  ─────────────────────────────────────")

    main = all_metrics.get("graphrag_main", {})
    no_ctx = all_metrics.get("graphrag_no_context", {})
    refuse = all_metrics.get("graphrag_always_refuse", {})

    if main and no_ctx:
        hall_main = main.get("hallucination_rate", None)
        hall_noctx = no_ctx.get("hallucination_rate", None)
        if hall_main is not None and hall_noctx is not None:
            diff = hall_noctx - hall_main
            status = "✅" if diff >= 0.3 else "⚠️ "
            print(f"  {status} hallucination_rate: main={hall_main:.3f} vs no_context={hall_noctx:.3f} (delta={diff:+.3f}, umbral≥0.30)")

    if refuse:
        corr_refuse = refuse.get("correctness_continuous", None)
        neg_refuse = refuse.get("negative_rejection", None)
        if corr_refuse is not None:
            status = "✅" if corr_refuse <= 0.1 else "⚠️ "
            print(f"  {status} correctness always_refuse: {corr_refuse:.3f} (esperado ≈ 0.0)")
        if neg_refuse is not None:
            status = "✅" if neg_refuse >= 0.9 else "⚠️ "
            print(f"  {status} negative_rejection always_refuse: {neg_refuse:.3f} (esperado ≈ 1.0)")

    print()


def main():
    parser = argparse.ArgumentParser(description="Genera tabla de poder discriminativo.")
    parser.add_argument("--subset", type=int, default=None,
                        help="Limitar a N ejemplos del dataset (default: todos)")
    parser.add_argument("--skip-no-context", action="store_true",
                        help="Saltar graphrag_no_context (requiere Neo4j activo)")
    parser.add_argument("--only", nargs="+",
                        choices=["graphrag_main", "graphrag_naive", "graphrag_no_context", "graphrag_always_refuse"],
                        help="Evaluar solo estos wrappers")
    args = parser.parse_args()

    # ── Cargar dataset ───────────────────────────────────────────────────────
    from rag_eval.datasets.northwind import DATASET_NORTHWIND
    dataset = DATASET_NORTHWIND
    if args.subset:
        dataset = dataset[:args.subset]
    dataset_name = f"northwind-discriminative-{'full' if not args.subset else str(args.subset)}"

    # ── Definir wrappers a evaluar ───────────────────────────────────────────
    from rag_eval.wrappers.graphrag_neo4j import neo4j_graphrag_wrapper_standalone
    from rag_eval.wrappers.graphrag_naive import neo4j_graphrag_naive
    from rag_eval.wrappers.graphrag_no_context import graphrag_no_context
    from rag_eval.wrappers.graphrag_always_refuse import graphrag_always_refuse

    wrappers = [
        {
            "fn":           neo4j_graphrag_wrapper_standalone,
            "name":         "graphrag_main",
            "architecture": "GraphRAG-LangChain-Main",
            "llm":          "gpt-3.5-turbo",
        },
        {
            "fn":           neo4j_graphrag_naive,
            "name":         "graphrag_naive",
            "architecture": "GraphRAG-LangChain-Naive",
            "llm":          "gpt-3.5-turbo",
        },
        {
            "fn":           graphrag_no_context,
            "name":         "graphrag_no_context",
            "architecture": "GraphRAG-NoContext-LLMOnly",
            "llm":          "gpt-3.5-turbo",
        },
        {
            "fn":           graphrag_always_refuse,
            "name":         "graphrag_always_refuse",
            "architecture": "GraphRAG-AlwaysRefuse",
            "llm":          "none",
        },
    ]

    # Filtrar si se especificó --only
    if args.only:
        wrappers = [w for w in wrappers if w["name"] in args.only]
    if args.skip_no_context:
        wrappers = [w for w in wrappers if w["name"] != "graphrag_no_context"]

    print(f"\n{'═' * 60}")
    print(f"  US-M1: Tabla de Poder Discriminativo")
    print(f"  Dataset: {dataset_name} ({len(dataset)} ejemplos)")
    print(f"  Wrappers: {[w['name'] for w in wrappers]}")
    print(f"{'═' * 60}")

    # ── Ejecutar evaluaciones ────────────────────────────────────────────────
    all_metrics = {}
    all_results = {}

    for w in wrappers:
        try:
            results = run_wrapper_eval(
                wrapper_fn=w["fn"],
                wrapper_name=w["name"],
                dataset=dataset,
                dataset_name=dataset_name,
                architecture=w["architecture"],
                llm=w["llm"],
            )
            metrics = extract_metrics(results)
            all_metrics[w["name"]] = metrics
            all_results[w["name"]] = results
            print(f"✅ {w['name']}: {metrics}")
        except Exception as e:
            print(f"❌ Error en {w['name']}: {e}")
            all_metrics[w["name"]] = {}

    # ── Imprimir tabla ───────────────────────────────────────────────────────
    print_table(all_metrics)

    return all_metrics, all_results


if __name__ == "__main__":
    main()
