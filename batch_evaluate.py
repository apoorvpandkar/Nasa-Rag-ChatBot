#!/usr/bin/env python3
"""
Batch Evaluation Script for NASA RAG System

Loads evaluation_dataset.json, runs each question through the full RAG pipeline
(retrieve context -> generate answer -> evaluate with RAGAS), and outputs
per-question results and aggregate metric summaries.

Usage:
    export OPENAI_API_KEY="your-key"
    export OPENAI_API_BASE="https://openai.vocareum.com/v1"  # if using proxy
    python batch_evaluate.py [--dataset evaluation_dataset.json] [--n-docs 3]
"""

import json
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List

import rag_client
import llm_client
import ragas_evaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("batch_evaluation.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def load_evaluation_dataset(dataset_path: str) -> List[Dict]:
    """Load evaluation questions from JSON file."""
    path = Path(dataset_path)
    if not path.exists():
        logger.error(f"Dataset file not found: {dataset_path}")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    logger.info(f"Loaded {len(dataset)} questions from {dataset_path}")
    return dataset


def run_batch_evaluation(
    dataset: List[Dict],
    chroma_dir: str,
    collection_name: str,
    openai_key: str,
    n_docs: int = 3,
    model: str = "gpt-4o-mini",
) -> List[Dict]:
    """
    Run full RAG pipeline + evaluation for each question in the dataset.

    Returns:
        List of per-question result dicts with question, answer, contexts, and scores.
    """
    # Initialize RAG system
    collection, success, error = rag_client.initialize_rag_system(chroma_dir, collection_name)
    if not success:
        logger.error(f"Failed to initialize RAG system: {error}")
        sys.exit(1)

    logger.info(f"RAG system initialized: {chroma_dir}/{collection_name}")

    results = []
    for i, item in enumerate(dataset, 1):
        question = item["question"]
        category = item.get("category", "unknown")
        expected = item.get("expected_response", "")

        logger.info(f"[{i}/{len(dataset)}] Evaluating: {question[:60]}...")

        # Step 1: Retrieve context
        docs_result = rag_client.retrieve_documents(collection, question, n_docs)
        contexts = []
        if docs_result and docs_result.get("documents"):
            contexts = docs_result["documents"][0]

        context_str = rag_client.format_context(
            docs_result["documents"][0], docs_result["metadatas"][0]
        ) if docs_result and docs_result.get("documents") else ""

        # Step 2: Generate answer
        answer = llm_client.generate_response(openai_key, question, context_str, [], model)

        # Step 3: Evaluate with RAGAS
        scores = ragas_evaluator.evaluate_response_quality(question, answer, contexts)

        result = {
            "question": question,
            "category": category,
            "answer": answer,
            "expected_response": expected,
            "retrieved_contexts": contexts,
            "scores": scores,
        }
        results.append(result)

        # Log per-question scores
        if "error" in scores:
            logger.warning(f"  Evaluation error: {scores['error']}")
        else:
            score_str = ", ".join(f"{k}: {v:.3f}" if isinstance(v, (int, float)) else f"{k}: {v}" for k, v in scores.items())
            logger.info(f"  Scores: {score_str}")

    return results


def compute_aggregate_metrics(results: List[Dict]) -> Dict[str, Dict]:
    """
    Compute aggregate statistics (mean, min, max) for each metric across all questions.
    """
    from collections import defaultdict

    metric_values = defaultdict(list)

    for r in results:
        scores = r.get("scores", {})
        if "error" in scores:
            continue
        for metric_name, value in scores.items():
            if isinstance(value, (int, float)) and value == value:  # exclude NaN
                metric_values[metric_name].append(value)

    aggregates = {}
    for metric_name, values in metric_values.items():
        if values:
            aggregates[metric_name] = {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "count": len(values),
            }

    return aggregates


def print_summary(results: List[Dict], aggregates: Dict[str, Dict]):
    """Print a formatted summary of per-question results and aggregate metrics."""
    print("\n" + "=" * 80)
    print("BATCH EVALUATION RESULTS")
    print("=" * 80)

    for i, r in enumerate(results, 1):
        print(f"\n--- Question {i} [{r['category']}] ---")
        print(f"Q: {r['question']}")
        print(f"A: {r['answer'][:200]}...")
        scores = r.get("scores", {})
        if "error" in scores:
            print(f"  EVALUATION ERROR: {scores['error']}")
        else:
            for metric, value in scores.items():
                if isinstance(value, (int, float)):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")

    print("\n" + "=" * 80)
    print("AGGREGATE METRICS")
    print("=" * 80)

    if not aggregates:
        print("No valid metric scores to aggregate.")
        return

    print(f"\n{'Metric':<45} {'Mean':>8} {'Min':>8} {'Max':>8} {'Count':>6}")
    print("-" * 80)
    for metric_name, stats in aggregates.items():
        print(
            f"{metric_name:<45} {stats['mean']:>8.4f} {stats['min']:>8.4f} "
            f"{stats['max']:>8.4f} {stats['count']:>6}"
        )

    print("\n" + "=" * 80)


def save_results(results: List[Dict], aggregates: Dict[str, Dict], output_path: str):
    """Save full results and aggregates to a JSON file."""
    output = {
        "per_question_results": results,
        "aggregate_metrics": aggregates,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Batch evaluation of NASA RAG system")
    parser.add_argument(
        "--dataset",
        default="evaluation_dataset.json",
        help="Path to evaluation dataset JSON file",
    )
    parser.add_argument(
        "--chroma-dir",
        default="./chroma_db_openai",
        help="ChromaDB persist directory",
    )
    parser.add_argument(
        "--collection-name",
        default="nasa_space_missions_text",
        help="ChromaDB collection name",
    )
    parser.add_argument("--n-docs", type=int, default=3, help="Number of docs to retrieve")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model for answer generation")
    parser.add_argument(
        "--output",
        default="batch_evaluation_results.json",
        help="Output file for results",
    )

    args = parser.parse_args()

    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        logger.error("OPENAI_API_KEY environment variable is not set.")
        sys.exit(1)

    # Load dataset
    dataset = load_evaluation_dataset(args.dataset)

    # Run evaluation
    results = run_batch_evaluation(
        dataset=dataset,
        chroma_dir=args.chroma_dir,
        collection_name=args.collection_name,
        openai_key=openai_key,
        n_docs=args.n_docs,
        model=args.model,
    )

    # Compute aggregates
    aggregates = compute_aggregate_metrics(results)

    # Print summary
    print_summary(results, aggregates)

    # Save results
    save_results(results, aggregates, args.output)


if __name__ == "__main__":
    main()
