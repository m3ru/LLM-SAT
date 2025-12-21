"""
Utilities for parsing and applying unified diffs to source code files.
Used for LLM-generated parameter tuning diffs.
"""

from typing import Optional
import logging
from unidiff import PatchSet

logger = logging.getLogger(__name__)


def apply_diff(baseline_content: str, diff_text: str) -> Optional[str]:
    """
    Apply a unified diff to baseline content.

    Args:
        baseline_content: Original file content as string
        diff_text: Unified diff text (with --- +++ @@ headers)

    Returns:
        Modified content as string, or None if diff fails to apply
    """
    try:
        # Parse the diff using unidiff
        patch = PatchSet(diff_text)

        if not patch:
            logger.error("No patches found in diff text")
            return None

        # Split baseline into lines for processing
        lines = baseline_content.split('\n')

        # Apply each patch (typically just one for restart.c)
        for patched_file in patch:
            logger.debug(f"Applying patch to {patched_file.path}")

            # Apply hunks in reverse order to avoid line number shifts
            for hunk in reversed(patched_file):
                lines = apply_hunk(lines, hunk)

        return '\n'.join(lines)

    except Exception as e:
        logger.error(f"Failed to apply diff: {e}")
        logger.debug(f"Diff text:\n{diff_text}")
        return None


def apply_hunk(lines: list, hunk) -> list:
    """
    Apply a single hunk to a list of lines.

    Args:
        lines: List of lines from the file
        hunk: unidiff.Hunk object

    Returns:
        Modified list of lines
    """
    # Hunk line numbers are 1-indexed
    # source_start is where the hunk begins in the original file
    source_start = hunk.source_start - 1  # Convert to 0-indexed

    # Build the modified section
    new_lines = []
    source_pos = 0

    for line in hunk:
        if line.is_added:
            # Add new line (strip the leading '+')
            new_lines.append(line.value.rstrip('\n'))
        elif line.is_removed:
            # Skip removed lines (they won't be in output)
            source_pos += 1
        elif line.is_context:
            # Context line - keep as is
            new_lines.append(line.value.rstrip('\n'))
            source_pos += 1

    # Calculate how many source lines this hunk affects
    source_length = hunk.source_length

    # Replace the affected section
    result = (
        lines[:source_start] +           # Before hunk
        new_lines +                       # Modified section
        lines[source_start + source_length:]  # After hunk
    )

    return result


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
    Validate that diff text is in proper unified diff format.

    Args:
        diff_text: Diff text to validate

    Returns:
        True if valid unified diff format
    """
    try:
        patch = PatchSet(diff_text)
        return len(patch) > 0
    except Exception as e:
        logger.warning(f"Invalid diff format: {e}")
        return False
