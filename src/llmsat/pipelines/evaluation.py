from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from typing import Dict, List

from llmsat.llmsat import (
    AlgorithmResult,
    BASE_SOLVER_PATH,
    CodeResult,
    SAT2025_BENCHMARK_PATH,
    get_logger,
)
from llmsat.utils.aws import (
    get_algorithm_result,
    get_code_result,
    update_algorithm_result,
    update_code_result,
)
from llmsat.utils.paths import get_solver_dir, get_solver_solving_times_path
from llmsat.utils.utils import wrap_command_to_slurm, get_activate_python_path
from llmsat.evaluation.coder import Coder
from ..evaluation.algorithm_evaluator import AlgorithmEvaluator
from ..evaluation.code_evaluator import CodeEvaluator

logger = get_logger(__name__)


@dataclass
class SolverEvaluationResult:
    solver_path: str
    solver_id: str
    build_success: bool
    par2_score: float
    other_metrics: Dict[str, float]

    def get_reward(self):
        # compute the final scalar / vector reward
        pass

def parse_solving_time(content: str) -> float:
    # parse the solving time from the content
    for line in content.split("\n"):
        if "solve time" in line:
            return float(line.split(" ")[-1])
    return None

@dataclass
class EvaluationPipeline:
    """Unified evaluation entry point for designer and coder models."""
    def __init__(self):
        pass

    def clean_solving_logs(self, algorithm_id: str, code_id: str) -> None:
        # clean the solving logs
        solver_dir = get_solver_dir(algorithm_id, code_id)
        for file in os.listdir(solver_dir):
            if file.endswith(".solving.log"):
                os.remove(f"{solver_dir}/{file}")
            if file.endswith(".slurm.log"):
                os.remove(f"{solver_dir}/{file}")

    def clean_solver(self, algorithm_id: str, code_id: str) -> None:
        # clean the solver
        solver_dir = get_solver_dir(algorithm_id, code_id)
        shutil.rmtree(solver_dir)

    def collect_results(self, algorithm_id: str, code_id: str) -> float:
        # collect the results from the solver
        solver_dir = get_solver_dir(algorithm_id, code_id)
        solving_times = {}
        for file in os.listdir(solver_dir):
            if file.endswith(".solving.log"):
                instance_name = file.split(".")[0]
                with open(f"{solver_dir}/{file}", "r") as f:
                    content = f.read()
                time_value = parse_solving_time(content)
                if time_value is not None:
                    solving_times[instance_name] = time_value

        # Compute PAR2 score (average of solving times, with penalty for timeouts)
        if solving_times:
            par2 = sum(solving_times.values()) / len(solving_times)
        else:
            par2 = float('inf')  # No results means infinite PAR2

        # update the code result and algorithm result
        code_result = get_code_result(code_id)
        if code_result:
            code_result.par2 = par2
            code_result.solving_time = json.dumps(solving_times)
            update_code_result(code_result)

        algorithm_result = get_algorithm_result(algorithm_id)
        if algorithm_result:
            algorithm_result.par2 = par2
            update_algorithm_result(algorithm_result)

        # Save solving times to file
        with open(get_solver_solving_times_path(algorithm_id, code_id), "w") as f:
            json.dump(solving_times, f, indent=2)

        return par2

    def slurm_colloct_result(self, slurm_ids: List[str], code_id: str) -> None:
        activate_python_path = get_activate_python_path()
        pass

    def build_solver(self, code_result: CodeResult) -> Optional[str]:
        """
        Build solver with modified restart.c file.

        Args:
            code_result: CodeResult containing the complete restart.c file content

        Returns:
            Path to built solver if successful, None otherwise
        """
        code = code_result.code

        # Copy original solver to a new folder
        new_solver_path = get_solver_dir(code_result.algorithm_id, code_result.task_id)
        logger.info(f"Copying base solver to {new_solver_path}")
        shutil.copytree(BASE_SOLVER_PATH, new_solver_path)

        # Simply replace the entire restart.c file
        restart_file = f"{new_solver_path}/src/restart.c"
        logger.info(f"Replacing restart.c with generated code ({len(code)} chars)")
        with open(restart_file, "w") as f:
            f.write(code)

        # Try to compile the solver
        logger.info("Compiling solver with make...")
        try:
            output = os.popen(f"cd {new_solver_path} && make 2>&1").read()

            # Check for errors in build output
            if "error:" in output.lower() or "fatal" in output.lower():
                logger.error(f"Build failed with errors:\n{output}")
                build_success = False
            else:
                logger.info("Build succeeded")
                build_success = True
        except Exception as e:
            logger.error(f"Build exception: {e}")
            build_success = False

        # Update code_result with build status
        code_result.build_success = build_success

        if build_success:
            new_solver_bin_path = f"{new_solver_path}/build/kissat"
            if os.path.exists(new_solver_bin_path):
                logger.info(f"Solver binary created at {new_solver_bin_path}")
                return new_solver_path
            else:
                logger.error(f"Build reported success but binary not found at {new_solver_bin_path}")
                return None
        else:
            logger.error("Build failed")
            return None 

    def slurm_run_evaluate(self, solver_path: str, benchmark_path: str) -> None:
        # run the solver on the benchmark
        activate_python_path = get_activate_python_path()
        slurm_ids = []
        for benchmark_file in os.listdir(benchmark_path):
            if benchmark_file.endswith(".cnf"):
                command = f"{activate_python_path} && ./{solver_path}/kissat {benchmark_path}/{benchmark_file} > {solver_path}/{benchmark_file}.solving.log"
                slurm_log = f"{solver_path}/{benchmark_file}.slurm.log"
                slurm_cmd = wrap_command_to_slurm(command, output_file=slurm_log, job_name=f"solve_{benchmark_file}")
                
                slurm_id = os.popen(slurm_cmd).read().strip() # TODO test if this is correct, also , there might be a limit on the number of jobs that can be submitted at once
                logger.info(f"Submitted job {slurm_id} for {benchmark_file}")
                slurm_ids.append(slurm_id)
        return slurm_ids

    def run_single_solver(self, code_id: str) -> None:  # pragma: no cover - declaration only
        """Run evaluation for configured components."""
        code_result = get_code_result(code_id)
        assert code_result is not None, "Code result not found"
        solver_path = self.build_solver(code_result) # build in evaluation
        if solver_path is not None: # build successful
            print(f"Solver built successfully: {solver_path}")
            slurm_ids = slurm_run_evaluate(solver_path, SAT2025_BENCHMARK_PATH)
            self.slurm_colloct_result(slurm_ids,code_id)
        else: # build failed
            print(f"Solver build failed: {build_result.error}")
    
    def read_current_progress(self) -> None:
        # read the current progress from the progress file
        pass

    def read_algorithm(self, algorithm_id: str) -> None:
        algorithm_result = get_algorithm_result(algorithm_id)
        assert algorithm_result is not None, "Algorithm result not found"
        return algorithm_result

    def run_all_solvers(self, algorithm_id: str) -> None:  # pragma: no cover - declaration only
        """Run evaluation for all configured components."""
        algorithm = self.read_algorithm(algorithm_id)
        os.makedirs(f"solvers/algorithm_{algorithm_id}", exist_ok=True)
        code_id_list = self.generate_or_read_code(algorithm) # actually should only read here, generation should be done in a separate process

        assert code is not None, "Code generation failed"
        
        for code_id in code_id_list:
            self.run_single_solver(code_id)
    
    def generate_or_read_code(self, algorithm: AlgorithmResult) -> None: #
        # generate the code for the solver
        for code_id in algorithm_result.code_id_list:
            code_result = get_code_result(code_id)
            if code_result is not None:
                return code_result
        

def main():
    pass

if __name__ == "__main__":
    main()