"""
Data generation pipeline using diff-based LLM outputs.

This module generates algorithm variations and corresponding code diffs
for parameter tuning in SAT solvers.
"""

import os
from pathlib import Path
from typing import Any, Dict
from llmsat.data.create_prompt import build_record, write_jsonl, render_custom_id
from llmsat.data.algorithm_parse import parse_kissat_restart_policy_json
from llmsat.utils.chatgpt_helper import (
    submit_batch_input as helper_submit_batch_input,
    block_until_completion as helper_block_until_completion,
    download_batch_outputs as helper_download_batch_outputs,
)
from llmsat.utils.paths import get_batch_output_dir, get_generation_output_dir, get_solver_solving_times_path
from llmsat.utils.aws import get_ids_from_router_table, update_router_table, get_algorithm_result, get_code_result, clear_router_table, get_algorithms_by_prompt, remove_code_result, remove_algorithm_result
from llmsat.llmsat import CHATGPT_DATA_GENERATION_TABLE, AlgorithmStatus, AlgorithmResult, ALGORITHM, CodeResult, CodeStatus, BASE_SOLVER_PATH
from datetime import datetime
import json
from llmsat.llmsat import get_id
from llmsat.utils.aws import update_algorithm_result
from llmsat.llmsat import NOT_INITIALIZED
from llmsat.utils.aws import update_code_result, add_par2_to_code_results_table
from llmsat.llmsat import setup_logging, get_logger
from llmsat.utils.paths import get_algorithm_dir
import logging

setup_logging(level=logging.INFO)
logger = get_logger(__name__)


def read_algorithm_prompt_file(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


def create_batch_input_file(prompt: str, output_path: str, n_requests: int = 10, model: str = "gpt-4.1"):
    logger.info(f"Creating batch input file for {n_requests} requests")
    # Reuse JSONL record structure from llmsat.data.create_prompt
    system_message = os.environ.get("LLMSAT_SYSTEM_MESSAGE", "You are an AI researcher specialising in SAT solver heuristics.")
    model = os.environ.get("OPENAI_MODEL", "gpt-4.1")
    try:
        temperature = float(os.environ.get("OPENAI_TEMPERATURE", "0.7"))
    except Exception:
        temperature = 0.7
    method = "POST"
    url = "/v1/responses"

    records = []
    for i in range(1, int(n_requests) + 1):
        custom_id = render_custom_id("req-{index:04d}", i, stem=None)
        records.append(
            build_record(
                system_message=system_message,
                user_prompt=prompt,
                model=model,
                temperature=temperature,
                method=method,
                url=url,
                custom_id=custom_id,
            )
        )
    write_jsonl(records, Path(output_path))


def submit_batch_input(file_path: str, block: bool = False, poll_interval_seconds: int = 60, timeout_seconds: int = 24 * 60 * 60) -> str:
    batch_id = helper_submit_batch_input(file_path, block=block, poll_interval_seconds=poll_interval_seconds, timeout_seconds=timeout_seconds)
    return batch_id


def block_until_completion(batch_id: str, poll_interval_seconds: int = 60, timeout_seconds: int = 24 * 60 * 60) -> str:
    return helper_block_until_completion(batch_id, poll_interval_seconds=poll_interval_seconds, timeout_seconds=timeout_seconds)


def download_batch_outputs(batch_id: str, output_path: str) -> str:
    result_path = helper_download_batch_outputs(batch_id, Path(output_path))
    return str(result_path)


def read_code_prompt_template(path: str) -> str:
    return read_algorithm_prompt_file(path)


def generate_code_prompt(template: str, algorithm: str) -> str:
    try:
        return template.format(algorithm=algorithm)
    except Exception:
        return f"{template}\n\nAlgorithm:\n{algorithm}"


def parse_diff_response(response: Dict[str, Any]) -> str:
    # Try to extract assistant text from OpenAI batch Responses output
    if not isinstance(response, dict):
        full_text = str(response)
        # Attempt to extract <diff>...</diff>
        start = full_text.find("<diff>")
        end = full_text.find("</diff>")
        if start != -1 and end != -1 and end > start:
            return full_text[start + len("<diff>"):end].strip()
        return full_text

    resp_obj = response.get("response") or response
    text = resp_obj.get("output_text") if isinstance(resp_obj, dict) else None
    if text:
        full_text = text
        start = full_text.find("<diff>")
        end = full_text.find("</diff>")
        if start != -1 and end != -1 and end > start:
            return full_text[start + len("<diff>"):end].strip()
        return full_text

    outputs = resp_obj.get("output") or resp_obj.get("outputs") if isinstance(resp_obj, dict) else None
    if isinstance(outputs, list):
        for item in outputs:
            content = item.get("content") if isinstance(item, dict) else None
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "output_text":
                        maybe = part.get("text")
                        if maybe:
                            full_text = maybe
                            start = full_text.find("<diff>")
                            end = full_text.find("</diff>")
                            if start != -1 and end != -1 and end > start:
                                return full_text[start + len("<diff>"):end].strip()
                            return full_text

    # Fallbacks
    if isinstance(resp_obj, dict) and "content" in resp_obj:
        full_text = str(resp_obj["content"])
        start = full_text.find("<diff>")
        end = full_text.find("</diff>")
        if start != -1 and end != -1 and end > start:
            return full_text[start + len("<diff>"):end].strip()
        return full_text

    full_text = json.dumps(response, ensure_ascii=False)
    start = full_text.find("<diff>")
    end = full_text.find("</diff>")
    if start != -1 and end != -1 and end > start:
        return full_text[start + len("<diff>"):end].strip()
    return full_text


def parse_algorithm_response(response: Dict[str, Any]) -> str:
    """
    Parse algorithm response (same as original pipeline).
    """
    # Extract text similarly to parse_diff_response
    if not isinstance(response, dict):
        raw_text = str(response)
    else:
        resp_obj = response.get("response") or response
        text = resp_obj.get("output_text") if isinstance(resp_obj, dict) else None
        if text:
            raw_text = text
        else:
            outputs = resp_obj.get("output") or resp_obj.get("outputs") if isinstance(resp_obj, dict) else None
            extracted = None
            if isinstance(outputs, list):
                for item in outputs:
                    content = item.get("content") if isinstance(item, dict) else None
                    if isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "output_text":
                                maybe = part.get("text")
                                if maybe:
                                    extracted = maybe
                                    break
                    if extracted:
                        break
            raw_text = extracted if extracted is not None else (str(resp_obj.get("content")) if isinstance(resp_obj, dict) and "content" in resp_obj else json.dumps(response, ensure_ascii=False))

    # Parse into validated algorithm spec dict, then normalize to JSON string
    try:
        spec = parse_kissat_restart_policy_json(raw_text)
        # Drop optional fields that aren't part of the core spec if present
        if isinstance(spec, dict) and "Reason" in spec:
            spec.pop("Reason")
        return json.dumps(spec, ensure_ascii=False)
    except Exception:
        # If parsing fails, return the raw text for debugging/upstream handling
        return raw_text


def generate_data(designer_prompt_path: str, code_prompt_template_path: str, generation_tag: str, n_algorithms: int = 500, n_codes: int = 10, use_cache=False, model: str = "gpt-4o"):
    """
    Generate algorithm and diff data using LLM batch API.

    This function:
    1. Generates algorithm specifications
    2. For each algorithm, generates parameter-tuning diffs
    3. Stores results in database with same schema as original pipeline

    Args:
        designer_prompt_path: Path to algorithm generation prompt
        code_prompt_template_path: Path to diff generation prompt template
        generation_tag: Tag to group this generation run
        n_algorithms: Number of algorithms to generate
        n_codes: Number of diff variations per algorithm
        use_cache: Whether to reuse cached outputs
        model: OpenAI model to use
    """
    if generation_tag is None:
        logger.error("Generation tag is None")
        return

    designer_prompt = read_algorithm_prompt_file(designer_prompt_path)
    code_prompt = read_algorithm_prompt_file(code_prompt_template_path)
    batch_input_path = os.path.join(get_generation_output_dir(generation_tag), "designer_batch_input.txt")
    batch_id_map = {}
    batch_id_map["algorithm_batch_id"] = None

    algorithms_output_path = os.path.join(get_generation_output_dir(generation_tag), "algorithms_output.txt")

    if use_cache and os.path.exists(algorithms_output_path):
        logger.info(f"Using cached algorithms from {algorithms_output_path}")
    else:
        create_batch_input_file(designer_prompt, batch_input_path, n_requests=n_algorithms, model=model)
        algorithm_batch_id = submit_batch_input(batch_input_path)
        batch_id_map["algorithm_batch_id"] = algorithm_batch_id
        block_until_completion(algorithm_batch_id)
        algorithms_output_path = os.path.join(get_batch_output_dir(generation_tag, batch_id=algorithm_batch_id), f"algorithms_output.txt")
        download_batch_outputs(algorithm_batch_id, algorithms_output_path)

    # Collect algorithm IDs from this run
    algorithm_ids_local = []
    with open(algorithms_output_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            algorithm_response = json.loads(line)
            algorithm_str = parse_algorithm_response(algorithm_response)
            algorithm_id = get_id(algorithm_str)
            algorithm_ids_local.append(algorithm_id)
            update_router_table(CHATGPT_DATA_GENERATION_TABLE, algorithm_id, generation_tag)
            algorithm_result = AlgorithmResult(
                id=algorithm_id,
                algorithm=algorithm_str,
                status=AlgorithmStatus.Generated,
                last_updated=datetime.now(),
                prompt=designer_prompt,
                par2=NOT_INITIALIZED,
                error_rate=NOT_INITIALIZED,
                other_metrics=NOT_INITIALIZED,
                code_id_list=[],
            )
            update_algorithm_result(algorithm_result)

    algorithm_ids = algorithm_ids_local
    code_prompt_template = read_code_prompt_template(code_prompt_template_path)
    waiting_batch_ids = []
    batch_id_to_algorithm_id = {}

    # Generate diff prompts for each algorithm
    for algorithm_id in algorithm_ids:
        algorithm_result = get_algorithm_result(algorithm_id)
        code_prompt = generate_code_prompt(code_prompt_template, algorithm_result.algorithm)

        algorithm_batch_id = batch_id_map["algorithm_batch_id"]
        code_batch_input_path = os.path.join(get_batch_output_dir(generation_tag, batch_id=algorithm_batch_id), f"code_batch_input_{algorithm_id}.txt")
        create_batch_input_file(code_prompt, code_batch_input_path, n_requests=n_codes, model=model)
        batch_id = submit_batch_input(code_batch_input_path)
        waiting_batch_ids.append(batch_id)
        batch_id_to_algorithm_id[batch_id] = algorithm_id

    batch_id_map["code_batch_ids"] = waiting_batch_ids
    batch_id_map["code_batch_map"] = batch_id_to_algorithm_id
    json.dump(batch_id_map, open(os.path.join(get_batch_output_dir(generation_tag, batch_id=batch_id_map["algorithm_batch_id"]), f"batch_id_map_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"), "w"))

    # Process code generation results
    while len(waiting_batch_ids) > 0:
        next_waiting_batch_ids = []
        for batch_id in list(waiting_batch_ids):
            block_until_completion(batch_id)
            code_output_path = os.path.join(get_batch_output_dir(generation_tag, batch_id=batch_id_map["algorithm_batch_id"]), f"code_output_{batch_id}.txt")
            download_batch_outputs(batch_id, code_output_path)

            mapped_algorithm_id = batch_id_to_algorithm_id.get(batch_id)
            if mapped_algorithm_id is None:
                logger.warning(f"Unknown batch_id {batch_id} -> algorithm mapping not found; skipping")
                continue

            algorithm_result = get_algorithm_result(mapped_algorithm_id)
            if algorithm_result is None:
                logger.warning(f"Algorithm {mapped_algorithm_id} not found when processing batch {batch_id}; skipping")
                continue

            if algorithm_result.code_id_list is None:
                algorithm_result.code_id_list = []
            existing_ids = set(algorithm_result.code_id_list)

            with open(code_output_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    code_response = json.loads(line)
                    # Parse diff instead of code
                    diff_str = parse_diff_response(code_response)
                    code_result = CodeResult(
                        id=get_id(diff_str),
                        algorithm_id=mapped_algorithm_id,
                        code=diff_str,  # Store diff as code field
                        status=CodeStatus.Generated,
                        par2=None,
                        last_updated=datetime.now(),
                        build_success=NOT_INITIALIZED,
                    )
                    update_code_result(code_result)
                    if code_result.id not in existing_ids:
                        algorithm_result.code_id_list.append(code_result.id)
                        existing_ids.add(code_result.id)
                    update_algorithm_result(algorithm_result)

        waiting_batch_ids = next_waiting_batch_ids


def print_generation_result(generation_tag: str):
    """Print summary of generated data."""
    algorithm_ids = get_ids_from_router_table(CHATGPT_DATA_GENERATION_TABLE, generation_tag)
    print(f"Number of algorithms: {len(algorithm_ids)}")
    for algorithm_id in algorithm_ids:
        algorithm_result = get_algorithm_result(algorithm_id)
        code_ids = algorithm_result.code_id_list
        print(f"algorithm_id: {algorithm_id}, Number of codes: {len(code_ids)}")
        for code_id in code_ids:
            code_result = get_code_result(code_id)
            print(f"code_id: {code_id}, algorithm_id: {code_result.algorithm_id}, status: {code_result.status}, build_success: {code_result.build_success}, par2: {code_result.par2}")


def main():
    generate_data(
        generation_tag="diff_testing",
        designer_prompt_path="./data/prompts/kissat_mab.txt",
        code_prompt_template_path="./data/prompts/kissat_mab_code_diff.txt",
        n_algorithms=10,
        n_codes=5,
        model="gpt-4o",
    )


if __name__ == "__main__":
    main()
