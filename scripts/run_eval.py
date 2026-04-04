"""
US-E1: scripts/run_eval.py — CLI para ejecutar evaluaciones reproducibles.

Uso:
    python scripts/run_eval.py \\
        --wrapper rag_eval.wrappers.graphrag_neo4j \\
        --dataset rag_eval.datasets.northwind \\
        --preset default \\
        --experiment-name "graphrag-main-v1"

    python scripts/run_eval.py \\
        --wrapper graphrag_no_context \\
        --dataset northwind \\
        --preset nli_only \\
        --architecture "GraphRAG-NoContext" \\
        --llm "gpt-3.5-turbo"

Wrappers disponibles (shorthand):
    graphrag_main         → graphrag_neo4j (validate_cypher=True)
    graphrag_naive        → graphrag_naive  (validate_cypher=False)
    graphrag_no_context   → graphrag_no_context (US-G1, sin contexto)
    graphrag_always_refuse→ graphrag_always_refuse (US-G2, siempre rechaza)

Datasets disponibles (shorthand):
    northwind             → rag_eval.datasets.northwind.DATASET_NORTHWIND
"""

import argparse
import importlib
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Añadir el directorio raíz al path para importar rag_eval
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")


# ── Mapas de shorthands ──────────────────────────────────────────────────────

WRAPPER_ALIASES = {
    "graphrag_main":          ("rag_eval.wrappers.graphrag_neo4j", "neo4j_graphrag_wrapper_standalone"),
    "graphrag_naive":         ("rag_eval.wrappers.graphrag_naive",  "neo4j_graphrag_naive"),
    "graphrag_no_context":    ("rag_eval.wrappers.graphrag_no_context",    "graphrag_no_context"),
    "graphrag_always_refuse": ("rag_eval.wrappers.graphrag_always_refuse", "graphrag_always_refuse"),
}

DATASET_ALIASES = {
    "northwind": ("rag_eval.datasets.northwind", "DATASET_NORTHWIND"),
}


def resolve_wrapper(spec: str):
    """Importa la función wrapper dado un alias o 'module.path:fn_name'."""
    if spec in WRAPPER_ALIASES:
        module_path, fn_name = WRAPPER_ALIASES[spec]
    elif ":" in spec:
        module_path, fn_name = spec.rsplit(":", 1)
    else:
        # Intentar inferir: el último componente es el nombre de la función
        parts = spec.rsplit(".", 1)
        if len(parts) == 2:
            module_path, fn_name = parts
        else:
            raise ValueError(
                f"No se puede resolver el wrapper '{spec}'. "
                f"Usa un alias ({list(WRAPPER_ALIASES.keys())}) "
                f"o el formato 'module.path:fn_name'."
            )

    mod = importlib.import_module(module_path)
    fn = getattr(mod, fn_name)
    return fn, fn_name


def resolve_dataset(spec: str):
    """Importa el dataset dado un alias o 'module.path:VAR_NAME'."""
    if spec in DATASET_ALIASES:
        module_path, var_name = DATASET_ALIASES[spec]
    elif ":" in spec:
        module_path, var_name = spec.rsplit(":", 1)
    else:
        parts = spec.rsplit(".", 1)
        if len(parts) == 2:
            module_path, var_name = parts
        else:
            raise ValueError(
                f"No se puede resolver el dataset '{spec}'. "
                f"Usa un alias ({list(DATASET_ALIASES.keys())}) "
                f"o el formato 'module.path:VAR_NAME'."
            )

    mod = importlib.import_module(module_path)
    dataset = getattr(mod, var_name)
    return dataset, var_name


def main():
    parser = argparse.ArgumentParser(
        description="Evalúa un wrapper RAG sobre un dataset usando LangSmith.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--wrapper", required=True,
        help="Wrapper a evaluar. Alias: graphrag_main, graphrag_naive, graphrag_no_context, "
             "graphrag_always_refuse. O formato 'rag_eval.wrappers.X:fn_name'.",
    )
    parser.add_argument(
        "--dataset", default="northwind",
        help="Dataset a usar. Alias: northwind. O formato 'rag_eval.datasets.X:VAR'. (default: northwind)",
    )
    parser.add_argument(
        "--preset", default="default", choices=["default", "full", "nli_only"],
        help="Preset de evaluadores: default | full | nli_only (default: default)",
    )
    parser.add_argument(
        "--experiment-name", default=None,
        help="Nombre del experimento en LangSmith (auto-generado si no se especifica)",
    )
    parser.add_argument(
        "--dataset-name", default=None,
        help="Nombre del dataset en LangSmith (default: alias del dataset)",
    )
    parser.add_argument(
        "--project", default="rag-eval",
        help="Proyecto LangSmith (default: rag-eval)",
    )
    parser.add_argument(
        "--architecture", default=None,
        help="Metadata: nombre de la arquitectura (ej. 'GraphRAG-LangChain')",
    )
    parser.add_argument(
        "--llm", default=None,
        help="Metadata: modelo LLM usado (ej. 'gpt-3.5-turbo')",
    )
    parser.add_argument(
        "--max-concurrency", type=int, default=1,
        help="Concurrencia máxima para la evaluación (default: 1, recomendado por rate limits)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Mostrar configuración sin ejecutar la evaluación",
    )

    args = parser.parse_args()

    # ── Resolver wrapper y dataset ───────────────────────────────────────────
    try:
        rag_fn, wrapper_name = resolve_wrapper(args.wrapper)
    except (ValueError, ImportError, AttributeError) as e:
        print(f"❌ Error al cargar wrapper '{args.wrapper}': {e}", file=sys.stderr)
        sys.exit(1)

    try:
        dataset, dataset_var = resolve_dataset(args.dataset)
    except (ValueError, ImportError, AttributeError) as e:
        print(f"❌ Error al cargar dataset '{args.dataset}': {e}", file=sys.stderr)
        sys.exit(1)

    dataset_name = args.dataset_name or args.dataset

    # ── Construir metadata US-E2 ─────────────────────────────────────────────
    metadata = {
        "wrapper":      wrapper_name,
        "dataset":      dataset_name,
        "preset":       args.preset,
    }
    if args.architecture:
        metadata["architecture"] = args.architecture
    if args.llm:
        metadata["llm"] = args.llm

    # ── Mostrar configuración ────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"  run_eval.py — Configuración")
    print(f"{'═' * 60}")
    print(f"  Wrapper    : {wrapper_name} ({args.wrapper})")
    print(f"  Dataset    : {dataset_name} ({len(dataset)} ejemplos)")
    print(f"  Preset     : {args.preset}")
    print(f"  Experiment : {args.experiment_name or '(auto)'}")
    print(f"  Project    : {args.project}")
    print(f"  Metadata   : {metadata}")
    print(f"  Concurrency: {args.max_concurrency}")
    print(f"{'═' * 60}\n")

    if args.dry_run:
        print("⚠️  --dry-run activo, no se ejecuta la evaluación.")
        return

    # ── Ejecutar evaluación ──────────────────────────────────────────────────
    from rag_eval.evaluators.universal import evaluate_rag_universal

    results = evaluate_rag_universal(
        rag_fn=rag_fn,
        dataset=dataset,
        dataset_name=dataset_name,
        project=args.project,
        preset=args.preset,
        experiment_name=args.experiment_name,
        metadata=metadata,
        max_concurrency=args.max_concurrency,
    )

    # ── Mostrar resumen ──────────────────────────────────────────────────────
    from rag_eval.evaluators.universal import print_universal_summary
    print_universal_summary(results, title=f"Results — {wrapper_name}")

    return results


if __name__ == "__main__":
    main()
