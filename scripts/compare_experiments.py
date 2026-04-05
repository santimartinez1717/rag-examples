"""
US-L1: Comparar experimentos LangSmith side-by-side.

Carga múltiples experimentos de LangSmith y genera una tabla comparativa
automáticamente. Alternativa programática a la UI de LangSmith.

Uso:
    # Comparar todos los experimentos discriminativos
    python scripts/compare_experiments.py \\
        --experiments "discriminative-graphrag_main" \\
                      "discriminative-graphrag_naive" \\
                      "discriminative-graphrag_no_context" \\
                      "discriminative-graphrag_always_refuse"

    # Listar experimentos disponibles
    python scripts/compare_experiments.py --list

    # Comparar por prefijo
    python scripts/compare_experiments.py --prefix "discriminative-"

    # Exportar a CSV
    python scripts/compare_experiments.py --prefix "discriminative-" --csv results/comparison.csv
"""

import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, List
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")


def list_experiments(prefix: Optional[str] = None, limit: int = 50):
    """Lista experimentos LangSmith disponibles."""
    from langsmith import Client
    client = Client()

    kwargs = {}
    if prefix:
        kwargs["project_name"] = prefix  # LangSmith filtra por prefijo si no es exact match

    projects = list(client.list_projects(**kwargs))[:limit]

    print(f"\n{'─' * 60}")
    print(f"  Experimentos LangSmith disponibles{' (prefijo: ' + prefix + ')' if prefix else ''}:")
    print(f"{'─' * 60}")
    for p in projects:
        print(f"  {p.name}")
    print(f"{'─' * 60}")
    print(f"  Total: {len(projects)}")
    return [p.name for p in projects]


def load_experiment_metrics(experiment_name: str) -> dict:
    """
    Carga métricas promedio de un experimento LangSmith.

    Returns:
        dict con métricas {"faithfulness_nli": 0.85, ...}
    """
    from langsmith import Client
    client = Client()

    try:
        runs = list(client.list_runs(
            project_name=experiment_name,
            run_type="chain",
            limit=200,
        ))

        if not runs:
            return {}

        # Agregar feedback por métrica
        metric_scores: Dict[str, List[float]] = {}

        for run in runs:
            if run.feedback_stats:
                for key, stats in run.feedback_stats.items():
                    score = stats.get("avg", stats.get("mean", None))
                    if score is not None:
                        if key not in metric_scores:
                            metric_scores[key] = []
                        metric_scores[key].append(score)

        # Calcular promedios
        metrics = {}
        for key, scores in metric_scores.items():
            valid = [s for s in scores if s is not None]
            if valid:
                metrics[key] = round(sum(valid) / len(valid), 3)

        return metrics

    except Exception as e:
        print(f"  ⚠️  Error cargando '{experiment_name}': {e}")
        return {}


def load_experiment_metadata(experiment_name: str) -> dict:
    """Carga metadata del experimento (architecture, wrapper, etc.)."""
    from langsmith import Client
    client = Client()

    try:
        projects = list(client.list_projects(name=experiment_name))
        if projects:
            return projects[0].metadata or {}
    except Exception:
        pass
    return {}


def print_comparison_table(all_data: dict, key_metrics: list = None):
    """Imprime tabla comparativa de experimentos."""
    if key_metrics is None:
        key_metrics = [
            "faithfulness_nli",
            "hallucination_rate",
            "correctness_continuous",
            "correctness_universal",
            "negative_rejection",
            "confidence_score_universal",
        ]

    # Filtrar solo métricas que aparecen en al menos un experimento
    present_metrics = []
    for m in key_metrics:
        if any(m in data.get("metrics", {}) for data in all_data.values()):
            present_metrics.append(m)

    if not present_metrics:
        # Mostrar todas las métricas disponibles
        all_metric_keys = set()
        for data in all_data.values():
            all_metric_keys.update(data.get("metrics", {}).keys())
        present_metrics = sorted(all_metric_keys)

    col_w = 20
    exp_col = 40

    # Header
    print(f"\n{'═' * (exp_col + col_w * len(present_metrics))}")
    print(f"  COMPARATIVA DE EXPERIMENTOS LANGSMITH (US-L1)")
    print(f"{'═' * (exp_col + col_w * len(present_metrics))}")

    header = f"{'Experimento':<{exp_col}}" + "".join(f"{m[:col_w-2]:<{col_w}}" for m in present_metrics)
    print(header)
    print("─" * (exp_col + col_w * len(present_metrics)))

    for exp_name, data in all_data.items():
        metrics = data.get("metrics", {})
        short_name = exp_name[:exp_col-1]
        row = f"{short_name:<{exp_col}}"
        for m in present_metrics:
            val = metrics.get(m)
            if val is None:
                row += f"{'—':<{col_w}}"
            else:
                row += f"{val:<{col_w}.3f}"
        print(row)

    print(f"{'═' * (exp_col + col_w * len(present_metrics))}")

    # Resumen: mejor experimento por métrica
    print("\n  Mejor por métrica:")
    for m in present_metrics:
        scores = {
            name: data["metrics"][m]
            for name, data in all_data.items()
            if m in data.get("metrics", {})
        }
        if not scores:
            continue
        # Para hallucination_rate: menor es mejor
        if "hallucination" in m or "error" in m:
            best = min(scores, key=lambda k: scores[k])
        else:
            best = max(scores, key=lambda k: scores[k])
        print(f"    {m:<40} {best} ({scores[best]:.3f})")
    print()


def export_csv(all_data: dict, output_path: str):
    """Exporta tabla de comparación a CSV."""
    import csv

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Recopilar todas las métricas
    all_metrics = set()
    for data in all_data.values():
        all_metrics.update(data.get("metrics", {}).keys())
    all_metrics = sorted(all_metrics)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        # Header
        writer.writerow(["experiment"] + list(all_metrics))
        # Rows
        for exp_name, data in all_data.items():
            row = [exp_name] + [data.get("metrics", {}).get(m, "") for m in all_metrics]
            writer.writerow(row)

    print(f"  ✅ CSV exportado: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compara experimentos LangSmith side-by-side.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--experiments", nargs="+", default=[],
                        help="Nombres exactos de experimentos LangSmith")
    parser.add_argument("--prefix", default=None,
                        help="Filtrar experimentos por prefijo")
    parser.add_argument("--list", action="store_true",
                        help="Listar experimentos disponibles y salir")
    parser.add_argument("--metrics", nargs="+", default=None,
                        help="Métricas a mostrar en la tabla (default: las 6 principales)")
    parser.add_argument("--csv", default=None,
                        help="Exportar tabla a CSV en esta ruta")
    args = parser.parse_args()

    if args.list:
        list_experiments(prefix=args.prefix)
        return

    if not args.experiments and not args.prefix:
        parser.print_help()
        sys.exit(1)

    # ── Resolver nombres de experimentos ─────────────────────────────────────
    experiment_names = list(args.experiments)
    if args.prefix:
        available = list_experiments(prefix=args.prefix)
        prefix_str: str = args.prefix
        experiment_names = [n for n in available if n is not None and n.startswith(prefix_str)]
        if not experiment_names:
            print(f"⚠️  No se encontraron experimentos con prefijo '{args.prefix}'")
            sys.exit(1)

    print(f"\n  Cargando {len(experiment_names)} experimentos de LangSmith...")

    # ── Cargar datos ──────────────────────────────────────────────────────────
    all_data = {}
    for name in experiment_names:
        if not name:
            continue
        print(f"  → {name}...")
        metrics = load_experiment_metrics(str(name))
        metadata = load_experiment_metadata(str(name))
        all_data[name] = {"metrics": metrics, "metadata": metadata}
        if metrics:
            print(f"    Métricas encontradas: {list(metrics.keys())}")
        else:
            print(f"    ⚠️  Sin métricas (experimento puede estar vacío o en otro proyecto)")

    # ── Imprimir tabla ────────────────────────────────────────────────────────
    print_comparison_table(all_data, key_metrics=args.metrics)

    # ── Exportar CSV ──────────────────────────────────────────────────────────
    if args.csv:
        export_csv(all_data, args.csv)

    return all_data


if __name__ == "__main__":
    main()
