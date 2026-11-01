"""Build the real evaluator explicitly; unit adapters select their own backends."""

import subprocess

import pytest

from inferatlas.verify.golden import Evaluator
from inferatlas.verify.io import ROOT


@pytest.fixture(scope="session")
def node_backend():
    for package in ("contracts", "expr"):
        subprocess.run(
            ["pnpm", "--filter", f"@inferatlas/{package}", "build"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )


@pytest.fixture
def runtime(node_backend, monkeypatch):
    evaluator = Evaluator(ROOT)
    monkeypatch.setattr(evaluator, "backend", "node")
    return evaluator
