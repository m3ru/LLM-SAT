"""
Utilities for parsing and applying unified diffs to source code files.
Used for LLM-generated parameter tuning diffs.
"""

from typing import Optional
import logging
import subprocess

logger = logging.getLogger(__name__)


def apply_diff(baseline_content: str, diff_text: str) -> Optional[str]:
    """
    Apply a unified diff to baseline content using GNU patch with fuzzy matching.

    Args:
        baseline_content: Original file content as string
        diff_text: Unified diff text (with --- +++ @@ headers)

    Returns:
        Modified content as string, or None if diff fails to apply
    """
    try:
        # Apply patch with fuzzy matching via stdin/stdout
        # -F 3: Allow up to 3 lines of fuzz (context mismatch)
        # -o -: Output to stdout
        result = subprocess.run(
            ["patch", "-F", "3", "-o", "-"],
            input=f"{diff_text}\n{baseline_content}",
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            logger.error(f"GNU patch failed with return code {result.returncode}")
            logger.error(f"Stderr: {result.stderr}")
            logger.debug(f"Stdout: {result.stdout}")
            logger.debug(f"Diff text:\n{diff_text}")
            return None

        # Log any warnings from patch
        if result.stderr and "offset" in result.stderr.lower():
            logger.info(f"Patch applied with offset: {result.stderr}")

        patched_content = result.stdout
        logger.info("Successfully applied diff using GNU patch")
        return patched_content

    except FileNotFoundError:
        logger.error("GNU patch command not found. Please install patch utility.")
        return None
    except Exception as e:
        logger.error(f"Failed to apply diff: {e}")
        logger.debug(f"Diff text:\n{diff_text}")
        return None


def extract_diff_from_tags(text: str) -> Optional[str]:
    """
    Extract diff content from <diff>...</diff> tags.

    Args:
        text: Raw text potentially containing <diff> tags

    Returns:
        Extracted diff text, or None if tags not found
    """
    start_tag = "<diff>"
    end_tag = "</diff>"

    start_idx = text.find(start_tag)
    end_idx = text.find(end_tag)

    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        logger.warning("Could not find <diff>...</diff> tags in text")
        return None

    diff_text = text[start_idx + len(start_tag):end_idx].strip()
    return diff_text


def validate_diff_format(diff_text: str) -> bool:
    """
    Validate that diff text appears to be in unified diff format.

    Args:
        diff_text: Diff text to validate

    Returns:
        True if text contains diff markers
    """
    # Basic validation: check for common diff markers
    has_header = '---' in diff_text or '+++' in diff_text
    has_hunks = '@@' in diff_text

    if has_header and has_hunks:
        return True

    logger.warning(f"Diff validation failed. Has header: {has_header}, Has hunks: {has_hunks}")
    return False
