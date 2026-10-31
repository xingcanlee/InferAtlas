"""The thin shell wrappers preserve caller arguments, command failures and cwd."""

import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "script,command",
    [
        ("scan", "extract"),
        ("triage", "lower"),
        ("verify", "verify"),
        ("golden_diff", "golden"),
        ("preview", "site"),
    ],
)
def test_wrapper_invocation(tmp_path, script, command):
    fake_uv = tmp_path / "uv"
    fake_uv.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        'print(json.dumps({"cwd": os.getcwd(), "args": sys.argv[1:]}))\n'
        'sys.exit(int(os.environ.get("FAKE_EXIT", "0")))\n'
    )
    fake_uv.chmod(0o755)
    wrapper = ROOT / f"skills/inferatlas-maintain/scripts/{script}.sh"
    env = {**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"]}
    for status in [0, 7]:
        result = subprocess.run(
            [str(wrapper), "--model", "a model with spaces", "--flag=literal;$value"],
            cwd=tmp_path,
            env={**env, "FAKE_EXIT": str(status)},
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == status
        assert json.loads(result.stdout) == {
            "cwd": str(ROOT / "generator"),
            "args": [
                "run",
                "inferatlas",
                command,
                "--model",
                "a model with spaces",
                "--flag=literal;$value",
            ],
        }
