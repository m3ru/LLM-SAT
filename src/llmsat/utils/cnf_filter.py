"""Filter for AIGER CNF benchmarks."""
from pathlib import Path
from typing import List


def get_aiger_benchmarks(benchmark_dir: str) -> List[str]:
    """
    Get all AIGER CNF files from benchmark directory.

    Args:
        benchmark_dir: Path to directory with CNF files

    Returns:
        List of AIGER CNF file paths
    """
    benchmark_path = Path(benchmark_dir)

    # Find all .cnf files
    all_cnf = list(benchmark_path.rglob("*.cnf"))

    # Filter for AIGER (contains "aig", "aiger", or "hwmcc" in filename)
    aiger_files = [
        str(f) for f in all_cnf
        if any(x in f.name.lower() for x in ["aig", "aiger", "hwmcc"])
    ]

    return aiger_files
