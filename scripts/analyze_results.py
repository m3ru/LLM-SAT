#!/usr/bin/env python3
"""
Analyze PAR2 scores from SAT solver evaluations.

Usage:
    # Analyze all results from database
    python scripts/analyze_results.py

    # Analyze specific algorithms
    python scripts/analyze_results.py --algorithm-ids <id1> <id2> ...

    # Compare against baseline
    python scripts/analyze_results.py --baseline-par2 500.0
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llmsat.llmsat import setup_logging, get_logger
from llmsat.utils.aws import get_all_tasks, get_algorithm_result, get_code_result

logger = get_logger(__name__)


def load_solving_times(algorithm_id: str, code_id: str) -> Optional[Dict[str, float]]:
    """Load solving times from JSON file."""
    from llmsat.utils.paths import get_solver_solving_times_path

    try:
        path = get_solver_solving_times_path(algorithm_id, code_id)
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def analyze_results(
    algorithm_ids: Optional[List[str]] = None,
    baseline_par2: Optional[float] = None
) -> None:
    """
    Analyze evaluation results and print summary.

    Args:
        algorithm_ids: Optional list of specific algorithm IDs to analyze
        baseline_par2: Optional baseline PAR2 score for comparison
    """
    logger.info("Analyzing evaluation results...\n")

    # Load all algorithm results
    if algorithm_ids:
        algorithms = [get_algorithm_result(aid) for aid in algorithm_ids]
        algorithms = [a for a in algorithms if a is not None]
    else:
        # Get all from database
        try:
            all_tasks = get_all_tasks()
            algorithm_ids = [task[0] for task in all_tasks]  # Assuming first column is ID
            algorithms = [get_algorithm_result(aid) for aid in algorithm_ids]
            algorithms = [a for a in algorithms if a is not None]
        except Exception as e:
            logger.error(f"Could not load from database: {e}")
            logger.info("Trying to load from local files...")
            algorithms = load_from_local_files()

    if not algorithms:
        logger.error("No results found to analyze")
        return

    logger.info(f"Found {len(algorithms)} algorithms with results\n")

    # Sort by PAR2 score
    algorithms_with_scores = [(a, getattr(a, 'par2', float('inf'))) for a in algorithms]
    algorithms_with_scores.sort(key=lambda x: x[1])

    # Print top performers
    print("=" * 80)
    print("TOP 10 BEST PERFORMING ALGORITHMS")
    print("=" * 80)

    for i, (algo, par2) in enumerate(algorithms_with_scores[:10], 1):
        algo_id = getattr(algo, 'id', 'unknown')
        algo_text = getattr(algo, 'algorithm', '')[:100]

        print(f"\n{i}. Algorithm ID: {algo_id[:16]}...")
        print(f"   PAR2 Score: {par2:.2f}")

        if baseline_par2:
            improvement = ((baseline_par2 - par2) / baseline_par2) * 100
            print(f"   Improvement over baseline: {improvement:+.2f}%")

        # Extract algorithm name if possible
        if algo_text:
            lines = algo_text.split('\n')
            for line in lines:
                if 'Algorithm Name' in line or '**' in line:
                    print(f"   Name: {line.strip()}")
                    break

    # Print worst performers
    print("\n" + "=" * 80)
    print("BOTTOM 5 WORST PERFORMING ALGORITHMS")
    print("=" * 80)

    for i, (algo, par2) in enumerate(algorithms_with_scores[-5:], 1):
        algo_id = getattr(algo, 'id', 'unknown')
        print(f"\n{i}. Algorithm ID: {algo_id[:16]}...")
        print(f"   PAR2 Score: {par2:.2f}")

    # Statistics
    print("\n" + "=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)

    par2_scores = [score for _, score in algorithms_with_scores if score != float('inf')]

    if par2_scores:
        avg_par2 = sum(par2_scores) / len(par2_scores)
        min_par2 = min(par2_scores)
        max_par2 = max(par2_scores)

        print(f"\nTotal algorithms evaluated: {len(par2_scores)}")
        print(f"Average PAR2: {avg_par2:.2f}")
        print(f"Best PAR2: {min_par2:.2f}")
        print(f"Worst PAR2: {max_par2:.2f}")

        if baseline_par2:
            better_than_baseline = sum(1 for s in par2_scores if s < baseline_par2)
            print(f"\nBaseline PAR2: {baseline_par2:.2f}")
            print(f"Algorithms better than baseline: {better_than_baseline}/{len(par2_scores)}")
            print(f"Percentage improvement: {(better_than_baseline/len(par2_scores)*100):.1f}%")

    # Save detailed results
    output_file = "data/analysis_results.json"
    results_data = {
        "total_algorithms": len(algorithms_with_scores),
        "baseline_par2": baseline_par2,
        "statistics": {
            "average": avg_par2 if par2_scores else None,
            "min": min_par2 if par2_scores else None,
            "max": max_par2 if par2_scores else None,
        },
        "top_10": [
            {
                "algorithm_id": getattr(algo, 'id', 'unknown'),
                "par2": score
            }
            for algo, score in algorithms_with_scores[:10]
        ]
    }

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"\nDetailed results saved to: {output_file}")


def load_from_local_files() -> List:
    """Fallback: load results from local solving_times.json files."""
    results = []
    solvers_dir = Path("solvers")

    if not solvers_dir.exists():
        return results

    for algo_dir in solvers_dir.glob("algorithm_*"):
        for code_dir in algo_dir.glob("code_*"):
            times_file = code_dir / "solving_times.json"
            if times_file.exists():
                with open(times_file, 'r') as f:
                    times = json.load(f)
                    if times:
                        par2 = sum(times.values()) / len(times)
                        # Create simple result object
                        class SimpleResult:
                            def __init__(self, aid, par2_score):
                                self.id = aid
                                self.par2 = par2_score
                                self.algorithm = ""

                        algo_id = algo_dir.name.replace("algorithm_", "")
                        results.append(SimpleResult(algo_id, par2))

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Analyze SAT solver evaluation results"
    )
    parser.add_argument(
        "--algorithm-ids",
        nargs="+",
        help="Specific algorithm IDs to analyze"
    )
    parser.add_argument(
        "--baseline-par2",
        type=float,
        help="Baseline PAR2 score for comparison"
    )

    args = parser.parse_args()

    setup_logging()

    analyze_results(
        algorithm_ids=args.algorithm_ids,
        baseline_par2=args.baseline_par2
    )


if __name__ == "__main__":
    main()
