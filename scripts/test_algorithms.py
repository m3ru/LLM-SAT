#!/usr/bin/env python3
"""
End-to-end testing script for SAT solver algorithm evaluation.

Workflow:
1. Parse algorithms from GPT-4 generated JSONL
2. Generate C code using Qwen2.5-7B-Instruct (non-fine-tuned)
3. Build solvers with generated code
4. Submit to SLURM for benchmark evaluation
5. Collect PAR2 scores

Usage:
    # Full pipeline (all algorithms)
    python scripts/test_algorithms.py --jsonl data/gpt_out_algorithm.jsonl

    # Test with limited number of algorithms
    python scripts/test_algorithms.py --jsonl data/gpt_out_algorithm.jsonl --limit 5

    # Generate code only (no evaluation)
    python scripts/test_algorithms.py --jsonl data/gpt_out_algorithm.jsonl --code-only

    # Evaluate already generated code
    python scripts/test_algorithms.py --evaluate-codes <code_id1> <code_id2> ...
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llmsat.data import parse_algorithm_jsonl, extract_algorithm_name
from llmsat.evaluation.coder import Coder, CoderConfig
from llmsat.llmsat import setup_logging, get_logger
from llmsat.pipelines.evaluation import EvaluationPipeline
from llmsat.utils.aws import update_code_result, update_algorithm_result
from llmsat.llmsat import AlgorithmResult

logger = get_logger(__name__)


def generate_codes_for_algorithms(
    algorithms_jsonl: str,
    output_dir: str = "data/generated_codes",
    limit: Optional[int] = None,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    device: str = "auto"
) -> List[dict]:
    """
    Generate C code for algorithms using Coder model.

    Args:
        algorithms_jsonl: Path to JSONL file with algorithm descriptions
        output_dir: Directory to save generated codes
        limit: Optional limit on number of algorithms to process
        model_name: HuggingFace model name or local path
        device: Device to use ("cuda", "cpu", or "auto")

    Returns:
        List of dicts with algorithm_id, code_id, code, and metadata
    """
    logger.info(f"Parsing algorithms from {algorithms_jsonl}")
    algorithms = parse_algorithm_jsonl(algorithms_jsonl, limit=limit)
    logger.info(f"Parsed {len(algorithms)} algorithms")

    # Initialize Coder
    if device == "auto":
        device = "cuda"  # Default to CUDA if available

    config = CoderConfig(model_name=model_name, device=device)
    coder = Coder(config)
    coder.load_model()

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results = []
    for i, algo_data in enumerate(algorithms):
        algo_payload = algo_data.payload
        algorithm_id = algo_payload["id"]
        algorithm_text = algo_payload["algorithm"]
        custom_id = algo_payload.get("custom_id", f"algo-{i}")

        logger.info(f"\n{'='*80}")
        logger.info(f"Processing algorithm {i+1}/{len(algorithms)}: {custom_id}")
        logger.info(f"Algorithm ID: {algorithm_id[:16]}...")

        # Extract algorithm name for logging
        algo_name = extract_algorithm_name(algorithm_text)
        logger.info(f"Algorithm name: {algo_name}")

        # Generate code
        code_result = coder.generate_code(
            algorithm=algorithm_text,
            algorithm_id=algorithm_id,
            solver_id="kissat-qwen25-7b"
        )

        if code_result.status == "completed":
            # Save code to file
            code_filename = f"{custom_id}_{code_result.task_id[:8]}.c"
            code_path = Path(output_dir) / code_filename
            coder.save_code(code_result.code, str(code_path))

            # Save metadata
            result_data = {
                "algorithm_id": algorithm_id,
                "algorithm_name": algo_name,
                "custom_id": custom_id,
                "code_id": code_result.task_id,
                "code_path": str(code_path),
                "code_length": len(code_result.code),
                "status": code_result.status
            }
            results.append(result_data)

            # Optionally store in database
            try:
                update_code_result(code_result)
                logger.info(f"Code result stored in database")
            except Exception as e:
                logger.warning(f"Could not store in database: {e}")

        else:
            logger.error(f"Code generation failed for {custom_id}")

    # Save summary
    summary_path = Path(output_dir) / "generation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nGeneration summary saved to {summary_path}")

    logger.info(f"\n{'='*80}")
    logger.info(f"Code generation complete!")
    logger.info(f"Successfully generated: {len(results)}/{len(algorithms)} codes")

    return results


def evaluate_codes(
    code_ids: List[str],
    benchmark_path: Optional[str] = None
) -> None:
    """
    Evaluate generated codes by building solvers and submitting to SLURM.

    Args:
        code_ids: List of code IDs to evaluate
        benchmark_path: Path to benchmark CNF files (uses default if None)
    """
    pipeline = EvaluationPipeline()

    for code_id in code_ids:
        logger.info(f"\n{'='*80}")
        logger.info(f"Evaluating code: {code_id[:16]}...")

        try:
            # This will:
            # 1. Build the solver
            # 2. Submit SLURM jobs
            # 3. Evaluation pipeline will collect results later
            pipeline.run_single_solver(code_id)
            logger.info(f"Evaluation submitted for code {code_id[:16]}")

        except Exception as e:
            logger.error(f"Evaluation failed for {code_id}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Test SAT solver algorithms end-to-end",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Input options
    parser.add_argument(
        "--jsonl",
        type=str,
        default="data/gpt_out_algorithm.jsonl",
        help="Path to algorithm JSONL file"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of algorithms to process"
    )

    # Model options
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model name or local path"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for model inference"
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/generated_codes",
        help="Directory to save generated codes"
    )

    # Pipeline control
    parser.add_argument(
        "--code-only",
        action="store_true",
        help="Only generate code, don't evaluate"
    )
    parser.add_argument(
        "--evaluate-codes",
        nargs="+",
        help="Evaluate specific code IDs (skip generation)"
    )
    parser.add_argument(
        "--benchmark-path",
        type=str,
        default=None,
        help="Path to benchmark CNF files"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging()
    logger.info("Starting algorithm testing pipeline")
    logger.info(f"Arguments: {vars(args)}")

    # Mode 1: Evaluate existing codes
    if args.evaluate_codes:
        logger.info("Mode: Evaluate existing codes")
        evaluate_codes(args.evaluate_codes, args.benchmark_path)
        return

    # Mode 2: Generate codes (and optionally evaluate)
    logger.info("Mode: Generate codes from algorithms")
    results = generate_codes_for_algorithms(
        algorithms_jsonl=args.jsonl,
        output_dir=args.output_dir,
        limit=args.limit,
        model_name=args.model,
        device=args.device
    )

    # If not code-only mode, proceed to evaluation
    if not args.code_only and results:
        logger.info("\nProceeding to evaluation phase...")
        code_ids = [r["code_id"] for r in results]
        evaluate_codes(code_ids, args.benchmark_path)
    elif args.code_only:
        logger.info("\nCode-only mode: Skipping evaluation")
        logger.info(f"Generated codes are in: {args.output_dir}")
        logger.info("To evaluate later, run:")
        for r in results:
            logger.info(f"  python scripts/test_algorithms.py --evaluate-codes {r['code_id']}")

    logger.info("\nPipeline complete!")


if __name__ == "__main__":
    main()
