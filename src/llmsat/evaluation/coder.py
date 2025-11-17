"""Coder class for generating SAT solver code from algorithm descriptions."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmsat.llmsat import CodeResult, TaskStatus, get_logger

logger = get_logger(__name__)


@dataclass
class CoderConfig:
    """Configuration for the Coder model."""

    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    model_path: Optional[str] = None  # If provided, load from local path instead
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True


class Coder:
    """
    Generates C code implementations from natural language algorithm descriptions.

    This class only handles code generation. Evaluation (building, SLURM submission,
    benchmark running) is handled by the EvaluationPipeline.
    """

    def __init__(self, config: Optional[CoderConfig] = None):
        """
        Initialize the Coder with model configuration.

        Args:
            config: CoderConfig object. If None, uses default configuration.
        """
        self.config = config or CoderConfig()
        self.model = None
        self.tokenizer = None
        self.device = self.config.device

        logger.info(f"Coder initialized with model: {self.config.model_name}")
        logger.info(f"Using device: {self.device}")

    def load_model(self) -> None:
        """Load the language model and tokenizer."""
        logger.info(f"Loading model: {self.config.model_name}")

        try:
            # Load tokenizer
            model_path = self.config.model_path or self.config.model_name
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )

            if self.device == "cpu":
                self.model = self.model.to(self.device)

            self.model.eval()
            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def create_prompt(self, algorithm: str, restart_c_content: str) -> str:
        """
        Create a prompt for modifying restart.c based on algorithm description.

        Args:
            algorithm: Natural language description of the SAT solver heuristic
            restart_c_content: Complete content of the base restart.c file

        Returns:
            Formatted prompt string for the model
        """
        prompt = f"""You are an expert SAT solver developer. Your task is to modify the restart.c file from the Kissat SAT solver to implement a new branching heuristic.

# Algorithm to Implement:
{algorithm}

# Current restart.c File:
```c
{restart_c_content}
```

# Instructions:

1. Modify the branching/decision heuristic functions in restart.c to implement the algorithm described above
2. Focus on modifying the variable selection logic (how variables are scored and chosen)
3. You can use these available data structures from Kissat:
   - `stab[x]`: EVSIDS activity score for variable x
   - `btab[x]`: Move-to-front timestamp for variable x
   - `stats.conflicts`: Global conflict counter
   - `phases.target[x]`: Target phase for variable x
   - `phases.saved[x]`: Saved phase for variable x
   - `stable`: Boolean indicating if solver is in stable mode

4. Keep the implementation efficient (O(1) or O(log n) operations per variable)
5. Preserve all necessary includes, function signatures, and overall file structure
6. Only modify the parts needed to implement the new heuristic
7. Output the COMPLETE modified restart.c file

# Output Format:
Provide the entire modified restart.c file inside a code block. Do not provide explanations, just the complete C file.

```c
// Complete modified restart.c file here
```"""

        return prompt

    def parse_response(self, response: str) -> str:
        """
        Parse and extract C code from model response.

        Args:
            response: Raw text response from the model

        Returns:
            Extracted C code string
        """
        # Try to extract code from markdown code blocks
        code_block_pattern = r"```(?:c|cpp|C)?\s*(.*?)\s*```"
        matches = re.findall(code_block_pattern, response, re.DOTALL)

        if matches:
            # Use the first (or largest) code block
            code = max(matches, key=len)
            logger.info("Extracted code from markdown block")
            return code.strip()

        # If no code blocks found, try to extract lines that look like C code
        lines = response.split('\n')
        code_lines = []
        in_code = False

        for line in lines:
            # Start collecting when we see C-like patterns
            if any(keyword in line for keyword in ['int ', 'double ', 'float ', 'for(', 'if(', '{', '}']):
                in_code = True

            if in_code:
                code_lines.append(line)

        if code_lines:
            logger.info("Extracted code from response heuristically")
            return '\n'.join(code_lines).strip()

        # Fallback: return the whole response
        logger.warning("Could not identify code blocks, returning full response")
        return response.strip()

    def generate_code(
        self,
        algorithm: str,
        algorithm_id: str,
        solver_id: Optional[str] = None,
        base_solver_path: str = "solvers/base"
    ) -> CodeResult:
        """
        Generate complete restart.c file from algorithm description.

        This method ONLY generates code. It does NOT:
        - Build the solver
        - Submit SLURM jobs
        - Run benchmarks

        Those steps are handled by EvaluationPipeline.

        Args:
            algorithm: Natural language algorithm description
            algorithm_id: Unique identifier for the algorithm
            solver_id: Optional identifier for the solver variant
            base_solver_path: Path to base Kissat solver (default: "solvers/base")

        Returns:
            CodeResult object containing the complete modified restart.c file
        """
        if self.model is None or self.tokenizer is None:
            logger.info("Model not loaded, loading now...")
            self.load_model()

        logger.info(f"Generating code for algorithm: {algorithm_id[:8]}...")

        try:
            # Load the base restart.c file
            from pathlib import Path
            restart_c_path = Path(base_solver_path) / "src" / "restart.c"

            if not restart_c_path.exists():
                raise FileNotFoundError(f"Base restart.c not found at: {restart_c_path}")

            with open(restart_c_path, 'r') as f:
                restart_c_content = f.read()

            logger.info(f"Loaded restart.c ({len(restart_c_content)} chars) from {restart_c_path}")

            # Create prompt with full restart.c content
            prompt = self.create_prompt(algorithm, restart_c_content)

            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096
            ).to(self.device)

            # Generate
            logger.info("Running model inference...")
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=self.config.do_sample,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Decode
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract just the generated part (after the prompt)
            response = full_response[len(prompt):].strip()

            # Parse code from response
            code = self.parse_response(response)

            logger.info(f"Generated {len(code)} characters of code")

            # Generate code ID from the code content
            code_id = hashlib.sha256(code.encode('utf-8')).hexdigest()

            # Create CodeResult
            # Note: build_success will be updated later by EvaluationPipeline
            code_result = CodeResult(
                task_id=code_id,
                status=TaskStatus.Completed,
                created_at=datetime.now().isoformat(),
                algorithm_id=algorithm_id,
                code=code,
                solver_id=solver_id or "kissat-base",
                build_success=False  # Will be updated after build attempt in EvaluationPipeline
            )

            return code_result

        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            # Return failed result
            task_id = hashlib.sha256(f"{algorithm_id}_{datetime.now()}".encode()).hexdigest()
            return CodeResult(
                task_id=task_id,
                status=TaskStatus.Failed,
                created_at=datetime.now().isoformat(),
                algorithm_id=algorithm_id,
                code="",
                solver_id=solver_id or "kissat-base",
                build_success=False
            )

    def save_code(self, code: str, output_path: str) -> None:
        """
        Save generated code to file.

        Args:
            code: C code string
            output_path: Path where code should be saved
        """
        with open(output_path, 'w') as f:
            f.write(code)
        logger.info(f"Code saved to: {output_path}")
