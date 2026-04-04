"""
US-C1: Calibrar confidence score con pesos aprendidos (regresión logística).

Este script:
1. Descarga resultados de experimentos LangSmith (o acepta un CSV de métricas)
2. Extrae métricas y labels de correctness
3. Entrena regresión logística → pesos aprendidos
4. Reporta ECE antes y después
5. Guarda el modelo en models/confidence_model.json

Uso:
    # Desde experimentos LangSmith
    python scripts/calibrate_confidence.py \\
        --experiments "discriminative-graphrag_main" "discriminative-graphrag_naive" \\
        --output models/confidence_model.json

    # Test rápido con datos sintéticos
    python scripts/calibrate_confidence.py --synthetic

Criterio de aceptación (US-C1):
    ECE del score aprendido < ECE del score heurístico actual
"""

import sys
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")


def load_experiment_data(experiment_names: list) -> tuple:
    """
    Descarga métricas y labels de experimentos LangSmith.

    Returns:
        (metrics_list, labels) donde:
            metrics_list: List[Dict[str, float]] — métricas por ejemplo
            labels: List[float] — correctness binario (0/1)
    """
    from langsmith import Client
    client = Client()

    metrics_list = []
    labels = []

    for exp_name in experiment_names:
        print(f"  Cargando experimento: {exp_name}...")
        try:
            # Buscar experimento por nombre (prefix match)
            experiments = list(client.list_projects(name=exp_name))
            if not experiments:
                # Intentar buscar como dataset experiment
                runs = list(client.list_runs(project_name=exp_name))
                if not runs:
                    print(f"  ⚠️  Experimento '{exp_name}' no encontrado, saltando.")
                    continue
            else:
                runs = list(client.list_runs(project_name=exp_name))

            for run in runs:
                if run.feedback_stats:
                    metrics = {}
                    for key, stats in run.feedback_stats.items():
                        if "avg" in stats:
                            metrics[key] = stats["avg"]
                        elif "mean" in stats:
                            metrics[key] = stats["mean"]

                    # Label: correctness_continuous o correctness_universal
                    label = None
                    for k in ["correctness_continuous", "correctness_universal"]:
                        if k in metrics:
                            label = metrics[k]
                            break

                    if label is not None and metrics:
                        metrics_list.append(metrics)
                        labels.append(label)

        except Exception as e:
            print(f"  ❌ Error cargando '{exp_name}': {e}")

    print(f"  Total ejemplos cargados: {len(metrics_list)}")
    return metrics_list, labels


def synthetic_data():
    """Genera datos sintéticos para probar el entrenamiento."""
    import random
    random.seed(42)

    metrics_list = []
    labels = []

    # 30 ejemplos sintéticos con correlación realista
    for i in range(30):
        faith = random.uniform(0.1, 1.0)
        corr = random.uniform(0.0, 1.0)
        # Label: 1 si faith > 0.6 y corr > 0.5, con algo de ruido
        label = 1.0 if (faith > 0.6 and corr > 0.5) else 0.0
        if random.random() < 0.1:
            label = 1.0 - label  # 10% ruido

        metrics_list.append({
            "faithfulness_nli":      faith,
            "hallucination_rate":    1.0 - faith,
            "correctness_continuous": corr,
            "negative_rejection":    1.0 if random.random() > 0.3 else 0.0,
        })
        labels.append(label)

    return metrics_list, labels


def main():
    parser = argparse.ArgumentParser(description="Calibra confidence score con regresión logística.")
    parser.add_argument("--experiments", nargs="+", default=[],
                        help="Nombres de experimentos LangSmith de donde extraer datos")
    parser.add_argument("--output", default="models/confidence_model.json",
                        help="Ruta de salida del modelo (default: models/confidence_model.json)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Usar datos sintéticos para probar el pipeline")
    parser.add_argument("--features", nargs="+",
                        default=["faithfulness_nli", "hallucination_rate",
                                 "correctness_continuous", "negative_rejection"],
                        help="Métricas a usar como features")
    args = parser.parse_args()

    from rag_eval.evaluators.universal import (
        train_confidence_weights,
        predict_confidence_score,
        compute_ece,
    )

    print(f"\n{'═' * 60}")
    print(f"  US-C1: Calibración de Confidence Score")
    print(f"{'═' * 60}")

    # ── Cargar datos ─────────────────────────────────────────────────────────
    if args.synthetic:
        print("\n  ⚠️  Usando datos sintéticos para test.")
        metrics_list, labels = synthetic_data()
    elif args.experiments:
        print(f"\n  Cargando datos de {len(args.experiments)} experimentos LangSmith...")
        metrics_list, labels = load_experiment_data(args.experiments)
    else:
        print("  ❌ Especifica --experiments o --synthetic")
        sys.exit(1)

    if len(metrics_list) < 5:
        print(f"  ❌ Solo {len(metrics_list)} ejemplos. Se necesitan ≥ 5.")
        sys.exit(1)

    print(f"  Datos: {len(metrics_list)} ejemplos, {sum(labels)} positivos ({100*sum(labels)/len(labels):.1f}%)")

    # ── ECE del score heurístico actual ──────────────────────────────────────
    heuristic_scores = []
    feature_keys = args.features
    for metrics in metrics_list:
        f = metrics.get("faithfulness_nli", 0.5)
        c = metrics.get("correctness_continuous", metrics.get("correctness_universal", 0.5))
        ar = metrics.get("answer_relevance_universal", 0.5)
        cr = metrics.get("context_relevance", 0.5)
        # Fórmula heurística actual
        heuristic_scores.append(0.35 * f + 0.25 * c + 0.20 * ar + 0.20 * cr)

    binary_labels = [int(l >= 0.5) for l in labels]
    ece_heuristic = compute_ece(heuristic_scores, binary_labels)["ece"]
    print(f"\n  ECE score heurístico (pesos fijos): {ece_heuristic:.4f}")

    # ── Entrenar modelo logístico ─────────────────────────────────────────────
    print(f"\n  Entrenando regresión logística sobre features: {feature_keys}...")
    try:
        model = train_confidence_weights(
            metrics_matrix=metrics_list,
            correctness_labels=labels,
            feature_keys=feature_keys,
        )
    except ValueError as e:
        print(f"  ❌ {e}")
        sys.exit(1)

    ece_learned = model["ece_logistic"]
    print(f"  ECE score aprendido (logistic regression): {ece_learned:.4f}")

    improvement = ece_heuristic - ece_learned
    status = "✅ MEJORA" if improvement > 0 else "⚠️  NO MEJORA"
    print(f"\n  {status}: delta ECE = {improvement:+.4f}")
    print(f"\n  Pesos aprendidos (normalizados):")
    for feat, weight in model["weights"].items():
        bar = "█" * int(abs(weight) * 20)
        sign = "+" if weight >= 0 else "-"
        print(f"    {feat:<35} {sign}{abs(weight):.4f} {bar}")
    print(f"    bias: {model['bias']:.4f}")

    # ── Guardar modelo ────────────────────────────────────────────────────────
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import datetime
    model_save = {
        **model,
        "ece_heuristic": ece_heuristic,
        "ece_improvement": improvement,
        "n_train": len(metrics_list),
        "trained_at": datetime.datetime.utcnow().isoformat() + "Z",
    }

    with open(output_path, "w") as f:
        json.dump(model_save, f, indent=2)
    print(f"\n  ✅ Modelo guardado en: {output_path}")

    # ── Resumen ───────────────────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"  Criterio de aceptación US-C1:")
    print(f"  ECE heurístico : {ece_heuristic:.4f}")
    print(f"  ECE aprendido  : {ece_learned:.4f}")
    print(f"  Cumple (aprendido < heurístico): {'✅ SÍ' if ece_learned < ece_heuristic else '❌ NO'}")
    print(f"{'═' * 60}\n")

    print("  Próximo paso:")
    print("    from rag_eval.evaluators.universal import confidence_score_learned_factory")
    print("    import json")
    print(f"    model = json.load(open('{args.output}'))")
    print("    learned_evaluator = confidence_score_learned_factory(model)")
    print("    # Usar learned_evaluator en client.evaluate(..., evaluators=[..., learned_evaluator])")

    return model


if __name__ == "__main__":
    main()
