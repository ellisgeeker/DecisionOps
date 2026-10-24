"""Pytest bootstrap.

GitHub Actions sets ``GITHUB_ACTIONS``, which makes Typer's Rich helper force
colour even under ``CliRunner``. The option highlighter then splits a flag such
as ``--cal-ratio`` into ``-cal`` and ``-ratio``, so substring asserts on the
unbroken name fail. Disable colour before any test imports Typer.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from typing import Any

import pytest

# Must run at import time, before ``decisionops.cli.main`` pulls in Typer.
os.environ["NO_COLOR"] = "1"
os.environ["_TYPER_FORCE_DISABLE_TERMINAL"] = "1"

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


@pytest.fixture
def plain_output() -> Callable[[Any], str]:
    """CLI capture with ANSI colour codes stripped."""

    def _plain(result: Any) -> str:
        return _ANSI_ESCAPE.sub("", result.output)

    return _plain
