from __future__ import annotations

from dataclasses import dataclass

from ..evaluation.algorithm_evaluator import AlgorithmEvaluator
from ..evaluation.code_evaluator import CodeEvaluator


@dataclass
class EvaluationPipeline:
    """Unified evaluation entry point for designer and coder models."""

    algo_eval: AlgorithmEvaluator
    code_eval: CodeEvaluator

    def run(self) -> None:  # pragma: no cover - declaration only
        """Run evaluation for configured components."""
        raise NotImplementedError

