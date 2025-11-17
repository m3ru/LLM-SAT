"""Parser for algorithm data from JSONL files."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List, Optional

from .base import AlgorithmData


def _extract_text_from_output(output_obj: dict) -> str:
    """
    Extract text content from OpenAI Responses API output object.

    Handles the structure:
    {"type": "message", "content": [{"type": "output_text", "text": "..."}, ...]}
    """
    if not isinstance(output_obj, dict):
        return ""

    parts = []
    content = output_obj.get("content") or []
    for item in content:
        if isinstance(item, dict):
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text)

    return "\n".join(parts).strip()


def generate_algorithm_id(text: str) -> str:
    """Generate SHA-256 hash ID for algorithm text."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def parse_algorithm_jsonl(
    jsonl_path: str,
    limit: Optional[int] = None
) -> List[AlgorithmData]:
    """
    Parse algorithms from JSONL file in OpenAI Responses API format.

    Args:
        jsonl_path: Path to JSONL file containing algorithm descriptions
        limit: Optional limit on number of algorithms to parse

    Returns:
        List of AlgorithmData objects with algorithm text and metadata

    Example JSONL structure:
        {
          "id": "batch_req_...",
          "custom_id": "req-heuristic-0001",
          "response": {
            "body": {
              "output": [
                {
                  "type": "message",
                  "content": [{"type": "output_text", "text": "...algorithm..."}]
                }
              ]
            }
          }
        }
    """
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    algorithms = []

    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON at line {idx + 1}: {e}")
                continue

            # Extract custom_id or fallback to batch id
            custom_id = obj.get("custom_id") or obj.get("id") or f"algorithm-{idx}"

            # Extract algorithm text from response body
            response = obj.get("response", {})
            body = response.get("body", {})
            outputs = body.get("output", [])

            # Combine all outputs
            text_parts = []
            for output in outputs:
                text = _extract_text_from_output(output)
                if text:
                    text_parts.append(text)

            full_text = "\n\n".join(text_parts).strip()

            if not full_text:
                print(f"Warning: No text found in entry {custom_id}")
                continue

            # Create AlgorithmData with payload
            algorithm_id = generate_algorithm_id(full_text)
            algorithm_data = AlgorithmData(
                payload={
                    "id": algorithm_id,
                    "custom_id": custom_id,
                    "algorithm": full_text,
                    "source": "gpt-4",
                    "index": idx
                }
            )

            algorithms.append(algorithm_data)

            if limit is not None and len(algorithms) >= limit:
                break

    print(f"Parsed {len(algorithms)} algorithms from {jsonl_path}")
    return algorithms


def extract_algorithm_name(text: str) -> str:
    """
    Extract algorithm name from the text.

    Looks for patterns like:
    - "1. **Algorithm Name**"
    - "**Algorithm Name:** ..."
    - First meaningful line
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return "Unknown Algorithm"

    # Look for "Algorithm Name" section
    for i, line in enumerate(lines[:-1]):
        if "algorithm" in line.lower() and "name" in line.lower():
            # Next line likely contains the name
            name_line = lines[i + 1]
            # Remove markdown formatting
            name_line = name_line.replace("**", "").replace("*", "").replace("#", "").strip()
            if name_line:
                return name_line

    # Fallback: return first non-numeric line
    for line in lines:
        cleaned = line.replace("*", "").replace("#", "").replace("-", "").strip()
        if cleaned and not cleaned[0].isdigit():
            return cleaned[:100]  # Limit length

    return lines[0][:100]
