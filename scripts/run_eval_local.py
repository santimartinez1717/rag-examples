"""
run_eval_local.py — Evaluación local sin LangSmith trace upload.

Calcula las mismas métricas NLI que evaluate_rag_universal pero guarda
resultados en CSV/JSON local. Útil cuando se alcanza el límite mensual
de LangSmith.

Uso:
    python scripts/run_eval_local.py \
        --wrapper sqlrag_langchain \
        --dataset northwind_sql \
        --output results/sql_local.csv

    python scripts/run_eval_local.py \
        --wrapper sqlrag_agent \
        --dataset northwind_sql

Métricas calculadas:
    faithfulness_nli, hallucination_rate, correctness_continuous, negative_rejection
"""

import argparse
import json
import sys
import os
import csv
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Load .env BEFORE setting tracing overrides so they take effect
load_dotenv(ROOT / ".env")

# Override after load_dotenv to prevent LangSmith background thread rate-limit hangs
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"


def run_local_eval(wrapper_fn, dataset: List[Dict], verbose: bool = True) -> List[Dict]:
    """
    Corre el wrapper sobre el dataset y calcula métricas NLI localmente.

    Returns: lista de dicts con {question, expected, answer, context,
                                  faithfulness_nli, hallucination_rate,
                                  correctness_continuous, negative_rejection}
    """
    from rag_eval.evaluators.universal import (
        faithfulness_nli,
        hallucination_rate,
        correctness_continuous,
        negative_rejection,
    )

    results = []
    n = len(dataset)

    for i, example in enumerate(dataset):
        inputs = example["inputs"]
        expected_output = example["outputs"]
        question = inputs["question"]

        if verbose:
            print(f"[{i+1}/{n}] {question[:70]}...", end="\r", flush=True)

        # Run wrapper
        try:
            output = wrapper_fn(inputs)
        except Exception as e:
            output = {"answer": f"ERROR: {e}", "context": ""}

        answer = output.get("answer", "")
        context = output.get("context", "")
        expected = expected_output.get("answer", "")

        # Compute metrics using actual function signatures
        scores = {}
        try:
            r = faithfulness_nli(inputs, output)
            scores["faithfulness_nli"] = round(r["score"], 4)
        except Exception:
            scores["faithfulness_nli"] = None

        try:
            r = hallucination_rate(inputs, output)
            scores["hallucination_rate"] = round(r["score"], 4)
        except Exception:
            scores["hallucination_rate"] = None

        try:
            r = correctness_continuous(inputs, output, expected_output)
            scores["correctness_continuous"] = round(r["score"], 4)
        except Exception:
            scores["correctness_continuous"] = None

        try:
            r = negative_rejection(inputs, output, expected_output)
            score = r.get("score")
            scores["negative_rejection"] = round(score, 4) if score is not None else None
        except Exception:
            scores["negative_rejection"] = None

        row = {
            "question": question,
            "expected": expected,
            "answer": answer,
            "context": context[:200] if context else "",
            **scores,
        }
        results.append(row)

    if verbose:
        print()  # clear \r line

    return results


def print_summary(results: List[Dict], title: str = "Local Evaluation Results"):
    """Imprime resumen de métricas."""
    metrics = ["faithfulness_nli", "hallucination_rate", "correctness_continuous", "negative_rejection"]

    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  N = {len(results)} questions")
    print()

    for m in metrics:
        vals = [r[m] for r in results if r.get(m) is not None]
        if vals:
            avg = sum(vals) / len(vals)
            print(f"  {m:<28}: {avg:.3f}")
    print(f"{'='*60}\n")


def save_csv(results: List[Dict], path: str):
    """Guarda resultados en CSV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not results:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to: {path}")


def save_json(results: List[Dict], path: str):
    """Guarda resultados en JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluación local sin LangSmith")
    parser.add_argument("--wrapper", required=True, help="Alias del wrapper (e.g. sqlrag_langchain)")
    parser.add_argument("--dataset", required=True, help="Alias del dataset (e.g. northwind_sql)")
    parser.add_argument("--output", help="Ruta CSV de salida (default: results/<wrapper>-<dataset>.csv)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    from scripts.run_eval import resolve_wrapper, resolve_dataset

    wrapper_fn, wrapper_name = resolve_wrapper(args.wrapper)
    dataset, dataset_name = resolve_dataset(args.dataset)

    if not dataset:
        print(f"ERROR: dataset '{args.dataset}' vacío o no encontrado")
        sys.exit(1)

    print(f"Wrapper:  {wrapper_name}")
    print(f"Dataset:  {dataset_name} ({len(dataset)} questions)")
    print(f"Metrics:  faithfulness_nli, hallucination_rate, correctness_continuous, negative_rejection")
    print()

    results = run_local_eval(wrapper_fn, dataset, verbose=not args.quiet)
    print_summary(results, title=f"{wrapper_name} × {dataset_name}")

    output_path = args.output or f"results/{wrapper_name}-{dataset_name}.csv"
    save_csv(results, output_path)
    save_json(results, output_path.replace(".csv", ".json"))
