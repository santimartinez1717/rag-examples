"""
Ejecuta evaluaciones multi-dataset con un solo comando.

Por defecto intenta: northwind, recommendations, metaqa.
Si un dataset no existe (JSON pendiente), lo salta y continúa.

Uso:
  python scripts/eval_multi_datasets.py \
    --wrapper graphrag_main \
    --preset full \
    --project graphrag-neo4j-evaluation
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")


def main():
    parser = argparse.ArgumentParser(description="Evalúa un wrapper en múltiples datasets")
    parser.add_argument("--wrapper", required=True, help="Alias o module:fn del wrapper")
    parser.add_argument("--preset", default="full", choices=["default", "full", "nli_only", "discriminative"])
    parser.add_argument("--project", default="graphrag-neo4j-evaluation")
    parser.add_argument("--datasets", nargs="+", default=["northwind", "recommendations", "metaqa"])
    parser.add_argument("--max-concurrency", type=int, default=1)
    args = parser.parse_args()

    from scripts.run_eval import resolve_wrapper, resolve_dataset
    from rag_eval.evaluators.universal import evaluate_rag_universal, print_universal_summary

    rag_fn, wrapper_name = resolve_wrapper(args.wrapper)
    print(f"Wrapper: {wrapper_name} ({args.wrapper})")
    print(f"Datasets: {args.datasets}")

    results_by_dataset = {}

    for ds_name in args.datasets:
        print(f"\n{'─' * 70}")
        print(f"Dataset: {ds_name}")
        print(f"{'─' * 70}")

        try:
            dataset, _ = resolve_dataset(ds_name)
            if not dataset:
                print(f"⚠️  Dataset '{ds_name}' vacío. Se omite.")
                continue
        except Exception as e:
            print(f"⚠️  No se pudo cargar '{ds_name}': {e}")
            continue

        experiment_name = f"{wrapper_name}-{ds_name}"
        metadata = {
            "wrapper": wrapper_name,
            "dataset": ds_name,
            "preset": args.preset,
        }

        res = evaluate_rag_universal(
            rag_fn=rag_fn,
            dataset=dataset,
            dataset_name=ds_name,
            project=args.project,
            preset=args.preset,
            experiment_name=experiment_name,
            metadata=metadata,
            max_concurrency=args.max_concurrency,
        )
        print_universal_summary(res, title=f"Results — {wrapper_name} — {ds_name}")
        results_by_dataset[ds_name] = res

    print(f"\n✅ Finalizado. Datasets evaluados: {list(results_by_dataset.keys())}")


if __name__ == "__main__":
    main()
