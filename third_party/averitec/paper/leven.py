"""Small compatibility implementation for the unmaintained ``leven`` package."""

from __future__ import annotations


def levenshtein(source: str, target: str) -> int:
    """Return the unit-cost Levenshtein edit distance between two strings."""

    if len(source) < len(target):
        return levenshtein(target, source)
    previous = list(range(len(target) + 1))
    for source_index, source_char in enumerate(source, start=1):
        current = [source_index]
        for target_index, target_char in enumerate(target, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[target_index] + 1,
                    previous[target_index - 1] + (source_char != target_char),
                )
            )
        previous = current
    return previous[-1]
