"""Enforce the agent's file boundary before a separate credential publishes its draft."""

import json
import os
import subprocess
import sys
from pathlib import Path, PurePosixPath

from jsonschema import validate
from ruamel.yaml import YAML


def prepare() -> None:
    pairs = set()
    paths = sorted(Path(".cache/incoming/triage").rglob("triage.json"))
    if not paths:
        raise ValueError("No triage files in the build artifact")
    schema = json.loads(Path("schemas/triage.schema.json").read_text())
    for path in paths:
        triage = json.loads(path.read_text())
        validate(triage, schema)
        if not triage["items"]:
            raise ValueError("Maintenance artifacts must contain unresolved triage")
        pairs.add(tuple(triage["snapshot"][name]["commit"] for name in ("vllm", "ascend")))
    if len(pairs) != 1:
        raise ValueError("A build artifact must describe exactly one upstream pair")
    vllm, ascend = pairs.pop()
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as stream:
        stream.write(f"vllm={vllm}\nascend={ascend}\n")


def changed_paths() -> list[str]:
    changed = subprocess.check_output(["git", "diff", "HEAD", "--name-only", "--no-renames", "-z"])
    untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "-z"])
    return sorted({path.decode() for path in (changed + untracked).split(b"\0") if path})


def allowed(path: str) -> bool:
    parts = PurePosixPath(path).parts
    return (
        len(parts) >= 2
        and parts[0] in {"rules", "annotations"}
        and ".." not in parts
        and PurePosixPath(path).suffix == ".yaml"
    )


def check() -> list[str]:
    paths = changed_paths()
    for name in paths:
        path = Path(name)
        if not allowed(name) or path.is_symlink():
            raise ValueError(f"Agent changed a forbidden path: {name}")
        if path.exists():
            schema = "rules" if path.parts[0] == "rules" else "annotations"
            validate(
                YAML(typ="safe").load(path.read_text()),
                json.loads(Path(f"schemas/{schema}.schema.json").read_text()),
            )
    return paths


def publish() -> None:
    paths = check()
    if not paths:
        print("No rule/annotation changes; no PR to open. Inspect the action's triage response.")
        return
    run_id = os.environ["SOURCE_RUN_ID"]
    if not run_id.isdecimal():
        raise ValueError("Invalid source run id")
    branch = f"maintenance/triage-{run_id}"
    base = os.environ["DEFAULT_BRANCH"]
    # Do not overwrite an earlier draft or its human edits on a retry.
    existing = subprocess.run(
        ["git", "ls-remote", "--exit-code", "--heads", "origin", branch],
        check=False,
    )
    if existing.returncode not in {0, 2}:
        raise RuntimeError("Could not query existing draft branch")
    if existing.returncode == 2:
        subprocess.run(["git", "checkout", "-b", branch], check=True)
        subprocess.run(["git", "config", "user.name", "inferatlas-maintenance[bot]"], check=True)
        subprocess.run(
            ["git", "config", "user.email", "inferatlas-maintenance@users.noreply.github.com"],
            check=True,
        )
        subprocess.run(["git", "add", "--", *paths], check=True)
        subprocess.run(
            ["git", "commit", "-m", "fix(rules): draft upstream triage resolutions"], check=True
        )
        subprocess.run(["git", "push", "origin", branch], check=True)
    prs = subprocess.check_output(
        ["gh", "pr", "list", "--head", branch, "--state", "all", "--json", "number"],
        text=True,
    )
    if json.loads(prs):
        print(f"A PR for {branch} already exists; leaving it for review.")
        return
    subprocess.run(
        ["gh", "label", "create", "needs-review", "--color", "D4C5F9", "--force"], check=True
    )
    body = Path(".cache/maintenance-pr.md")
    body.write_text(
        "Draft rules and annotations for upstream triage. Human review is required before merge.\n\n"
        f"Source: {os.environ['SOURCE_SHA']}\n\n"
        f"Evidence: https://github.com/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{run_id}\n\n"
        "Validation: schemas, lint, Python tests excluding extraction, Node tests, golden check.\n"
        "Review formulas, conditions and source anchors; rerun the pinned build before publication.\n"
    )
    subprocess.run(
        [
            "gh",
            "pr",
            "create",
            "--draft",
            "--base",
            base,
            "--head",
            branch,
            "--label",
            "needs-review",
            "--title",
            "fix(rules): review upstream triage resolutions",
            "--body-file",
            str(body),
        ],
        check=True,
    )


if __name__ == "__main__":
    {"prepare": prepare, "check": check, "publish": publish}[sys.argv[1]]()
