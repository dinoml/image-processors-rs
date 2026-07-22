"""Shared acceptance metadata for deterministic parity fixtures."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


def verify_clean_git_checkout(
    source: Path, *, expected_commit: str, project: str
) -> Path:
    """Return a resolved checkout pinned to `expected_commit` with no changes."""
    source = source.resolve()
    try:
        commit = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(source), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"unable to inspect {project} checkout at {source}") from error
    if commit != expected_commit:
        raise RuntimeError(f"expected {project} {expected_commit}, found {commit}")
    if status:
        raise RuntimeError(f"{project} checkout is not clean: {source}")
    return source


def comparison_policy(
    absolute_tolerance: float,
    *,
    relative_tolerance: float = 0.0,
    statistics_absolute_tolerance: float | None = None,
) -> dict[str, Any]:
    statistics_absolute_tolerance = (
        absolute_tolerance
        if statistics_absolute_tolerance is None
        else statistics_absolute_tolerance
    )
    return {
        "full_payload": {
            "absolute_tolerance": absolute_tolerance,
            "relative_tolerance": relative_tolerance,
        },
        "statistics": {
            "absolute_tolerance": statistics_absolute_tolerance,
            "relative_tolerance": relative_tolerance,
        },
        "exact": {
            "shape": True,
            "dtype": True,
            "integer_payload": True,
            "metadata": True,
        },
    }


def acceptance_contract(
    *,
    input_locations: list[str],
    input_description: str,
    config_locations: list[str],
    absolute_tolerance: float,
    relative_tolerance: float = 0.0,
    statistics_absolute_tolerance: float | None = None,
) -> dict[str, Any]:
    return {
        "deterministic_input": {
            "locations": input_locations,
            "description": input_description,
        },
        "processor_config": {
            "locations": config_locations,
            "explicit": True,
        },
        "comparison": comparison_policy(
            absolute_tolerance,
            relative_tolerance=relative_tolerance,
            statistics_absolute_tolerance=statistics_absolute_tolerance,
        ),
    }
