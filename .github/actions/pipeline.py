"""Small deterministic adapters between Actions, the catalog, and the public CLI."""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from jsonschema import validate
from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / ".cache"
SHA = re.compile(r"[0-9a-f]{40}")
SLUG = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def output(name: str, value: object) -> None:
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as stream:
        stream.write(f"{name}={value}\n")


def pins() -> dict[str, str]:
    result = {
        "ascend": os.environ["INFERATLAS_ASCEND_COMMIT"],
        "vllm": os.environ["INFERATLAS_VLLM_COMMIT"],
    }
    if not all(SHA.fullmatch(value) for value in result.values()):
        raise ValueError("Both upstream revisions must be full, lowercase commit SHAs")
    return result


def pair_key(pair: dict[str, object]) -> str:
    return hashlib.sha256(canonical(pair).encode()).hexdigest()


def snapshot_id(pair: dict[str, str]) -> str:
    paths = (
        "generator",  # Includes uv.lock.
        "rules",
        "annotations",
        "catalog.yaml",
        "packages",
        "web",
        "schemas",
        "pnpm-lock.yaml",
        ".github/actions",
    )
    trees = dict.fromkeys(paths)
    entries = subprocess.run(
        ["git", "ls-tree", "-z", "HEAD", "--", *paths],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    for entry in entries.split("\0"):
        if entry:
            metadata, path = entry.split("\t", 1)
            trees[path] = metadata.split()[2]
    return "snapshot-" + pair_key({**pair, "trees": trees})


def catalog_matrix(catalog: dict) -> list[dict]:
    models = catalog["models"]
    devices = catalog["devices"]
    if not models or not devices:
        raise ValueError("Catalog must contain models and devices")
    if len({model["id"] for model in models}) != len(models):
        raise ValueError("Duplicate model id")
    if len({device["id"] for device in devices}) != len(devices):
        raise ValueError("Duplicate device id")
    cells = []
    for model in sorted(models, key=lambda item: item["id"]):
        if not SLUG.fullmatch(model["id"]) or not model["hf_id"] or not model["config_rev"]:
            raise ValueError("Invalid model identity")
        for device in sorted(devices, key=lambda item: item["id"]):
            if device["id"] not in {"A2", "A3", "A5", "310P", "H100"}:
                raise ValueError("Unknown device")
            cells.append(
                {
                    "id": model["id"],
                    "model": model["hf_id"],
                    "config_rev": model["config_rev"],
                    "device": device["id"],
                }
            )
    if len(cells) > 256:
        raise ValueError("Catalog exceeds GitHub's matrix limit; shard the workflow")
    return cells


def plan() -> None:
    pair = pins()
    catalog = YAML(typ="safe").load((ROOT / "catalog.yaml").read_text())
    output("matrix", canonical({"include": catalog_matrix(catalog)}))
    output("snapshot", snapshot_id(pair))


def load_contract(path: Path, schema: str) -> dict:
    value = json.loads(path.read_text())
    validate(value, json.loads((ROOT / f"schemas/{schema}.schema.json").read_text()))
    return value


def check_identity(snapshot: dict, cell: dict, pair: dict) -> None:
    actual = {name: snapshot[name]["commit"] for name in ("ascend", "vllm")}
    if (
        actual != pair
        or snapshot["device"] != cell["device"]
        or snapshot["model"]
        != {
            "hf_id": cell["model"],
            "config_rev": cell["config_rev"],
        }
    ):
        raise ValueError(
            "Build identity differs from pinned source/config/device; refusing fallback"
        )


def run_cli(arguments: list[str], log: Path) -> int:
    with log.open("w") as stream:
        result = subprocess.run(
            ["uv", "run", "--project", str(ROOT / "generator"), "inferatlas", *arguments],
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    print(log.read_text(), end="")
    return result.returncode


def report_passes(report: dict) -> bool:
    required = {"schema", "coverage", "source_links"}
    checks = report["checks"]
    return (
        required <= {check["id"] for check in checks}
        and all(check["status"] == "pass" for check in checks if check["id"] in required)
        and all(check["status"] != "fail" for check in checks)
        and not report["golden_diff"]
        and not report["source_link_failures"]
    )


def cell() -> None:
    current = json.loads(os.environ["CELL"])
    pair = pins()
    dest = CACHE / "cell"
    dest.mkdir(parents=True, exist_ok=True)
    # Record identity before invoking the CLI so a hard failure still has diagnostics.
    result = {**current, "status": "error"}
    write_json(dest / "result.json", result)
    code = run_cli(
        ["build", "--model", current["model"], "--device", current["device"], "--out", str(dest)],
        dest / "build.log",
    )
    triage = load_contract(dest / "triage.json", "triage")
    check_identity(triage["snapshot"], current, pair)
    if not triage["items"]:
        if code:
            raise RuntimeError("Build failed without actionable triage")
        ir = load_contract(dest / "ir.json", "ir")
        check_identity(ir["snapshot"], current, pair)
        load_contract(dest / "view.json", "view")
        if run_cli(["verify", "--input", str(dest)], dest / "verify.log"):
            raise RuntimeError("Source/coverage verification failed")
        report = load_contract(dest / "report.json", "report")
        check_identity(report["snapshot"], current, pair)
        if run_cli(["golden", "check", "--input", str(dest)], dest / "golden.log"):
            diff = (dest / "golden.log").read_text().replace(str(ROOT), "<repo>").strip()
            triage["items"].append(
                {
                    "kind": "golden_diff",
                    "model": current["model"],
                    "diff": diff
                    or "Golden check exited unsuccessfully; inspect the build artifact.",
                }
            )
            write_json(dest / "triage.json", triage)
            load_contract(dest / "triage.json", "triage")
        elif not report_passes(report):
            raise RuntimeError("Verification report lacks passing schema/coverage/source checks")
    result["status"] = "triage" if triage["items"] else "ok"
    result["snapshot"] = snapshot_id(pair)
    write_json(dest / "result.json", result)


def collect_cells(cells: Path, bundle: Path, expected: list[dict]) -> tuple[int, bool]:
    results = {}
    errors = []
    for path in sorted(cells.glob("cell-*/result.json")):
        result = json.loads(path.read_text())
        key = (result["id"], result["device"])
        if key in results:
            raise ValueError("Duplicate cell artifact")
        results[key] = (result, path.parent)
    expected_keys = {(item["id"], item["device"]) for item in expected}
    if results.keys() - expected_keys:
        raise ValueError("Unexpected cell artifact")
    blocked = {result["id"] for result, _ in results.values() if result["status"] != "ok"}
    blocked.update(item["id"] for item in expected if (item["id"], item["device"]) not in results)
    clean = 0
    has_triage = False
    for item in expected:
        key = (item["id"], item["device"])
        if key not in results:
            errors.append(f"Missing cell: {item['id']}/{item['device']}")
            continue
        result, source = results[key]
        if any(result[field] != value for field, value in item.items()):
            raise ValueError("Artifact does not match catalog matrix")
        if result["status"] == "triage":
            has_triage = True
            shutil.copytree(source, bundle / "triage" / item["id"] / item["device"])
        elif result["status"] != "ok":
            errors.append(f"Failed cell: {item['id']}/{item['device']}")
        elif item["id"] not in blocked:
            snapshot = result["snapshot"]
            if not re.fullmatch(r"snapshot-[0-9a-f]{64}", snapshot):
                raise ValueError("Invalid snapshot identity")
            dest = bundle / "data" / snapshot / item["id"] / item["device"]
            dest.mkdir(parents=True, exist_ok=True)
            for filename in ("ir.json", "view.json", "report.json"):
                shutil.copyfile(source / filename, dest / filename)
            clean += 1
    write_json(
        bundle / "summary.json",
        {
            "clean": clean,
            "errors": sorted(errors),
            "blocked_models": sorted(blocked),
        },
    )
    return clean, has_triage


def collect() -> None:
    expected = json.loads(os.environ["EXPECTED_MATRIX"])["include"]
    clean, triage = collect_cells(CACHE / "cells", CACHE / "bundle", expected)
    output("clean", clean)
    output("triage", str(triage).lower())
    summary = json.loads((CACHE / "bundle/summary.json").read_text())
    output("errors", len(summary["errors"]))


def git(*arguments: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(CACHE / "state"), *arguments],
        check=check,
        text=True,
        capture_output=True,
    )


def restore(*, merge_data: bool = True) -> None:
    state_dir = CACHE / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    git("init", "--initial-branch=state")
    git("remote", "add", "origin", f"https://github.com/{os.environ['GITHUB_REPOSITORY']}.git")
    exists = git("ls-remote", "--exit-code", "--heads", "origin", "state", check=False)
    if exists.returncode == 0:
        git("fetch", "--depth=1", "origin", "state")
        git("checkout", "-B", "state", "FETCH_HEAD")
        if not (state_dir / "state.json").is_file():
            raise ValueError("Existing state branch is missing state.json")
    elif exists.returncode != 2:
        raise RuntimeError(exists.stderr)
    else:
        write_json(state_dir / "state.json", {"pairs": {}})
    source = CACHE / "bundle/data"
    if merge_data and source.exists():
        for path in sorted(source.rglob("*.json")):
            target = state_dir / "data" / path.relative_to(source)
            if target.exists() and target.read_bytes() != path.read_bytes():
                raise ValueError(f"Immutable snapshot changed: {path.relative_to(source)}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)


def gh(*arguments: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["gh", *arguments],
        check=check,
        text=True,
        capture_output=True,
        cwd=ROOT,
    )


def release(snapshot: str, singles: Path) -> None:
    files = sorted(singles.glob("*.html"))
    if not files:
        raise ValueError("Site renderer produced no single-file HTMLs")
    manifest = singles / "SHA256SUMS"
    manifest.write_text(
        "".join(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n" for path in files)
    )
    existing = gh("release", "view", snapshot, "--json", "isDraft", check=False)
    if existing.returncode:
        # A transient lookup failure must not be mistaken for a missing release.
        listing = json.loads(
            gh(
                "api", "--paginate", "--slurp", f"repos/{os.environ['GITHUB_REPOSITORY']}/releases"
            ).stdout
        )
        if any(r["tag_name"] == snapshot for page in listing for r in page):
            raise RuntimeError(existing.stderr)
        gh(
            "release",
            "create",
            snapshot,
            "--draft",
            "--target",
            os.environ["GITHUB_SHA"],
            "--title",
            snapshot,
            "--notes",
            "Immutable InferAtlas HTML snapshots; see SHA256SUMS.",
        )
        draft = True
    else:
        draft = json.loads(existing.stdout)["isDraft"]
    if draft:
        gh("release", "upload", snapshot, *map(str, [*files, manifest]), "--clobber")
        gh("release", "edit", snapshot, "--draft=false", "--latest=false")
    else:
        previous = CACHE / "release-check"
        previous.mkdir(parents=True, exist_ok=True)
        gh("release", "download", snapshot, "--dir", str(previous))
        expected = {path.name: path.read_bytes() for path in [*files, manifest]}
        actual = {path.name: path.read_bytes() for path in previous.iterdir()}
        if actual != expected:
            raise ValueError(
                "Published release differs; bump generator revision instead of overwriting"
            )


def publish() -> None:
    summary = json.loads((CACHE / "bundle/summary.json").read_text())
    if summary["errors"]:
        raise ValueError("Cannot publish a partial infrastructure failure")
    snapshot = os.environ["SNAPSHOT_ID"]
    state_file = CACHE / "state/state.json"
    state = json.loads(state_file.read_text())
    state["pairs"][pair_key(pins())] = {
        "snapshot": snapshot,
        "generator": os.environ["GITHUB_SHA"],
        "blocked_models": summary["blocked_models"],
    }
    save_state(state, f"chore(ci): retain {snapshot}")


def record_failure() -> None:
    # Failure jobs restore only the ledger/history, never partially built artifacts.
    restore(merge_data=False)
    state = json.loads((CACHE / "state/state.json").read_text())
    generator = os.environ["GITHUB_SHA"]
    if not SHA.fullmatch(generator):
        raise ValueError("Generator revision must be a full SHA")
    state.setdefault("failures", {}).setdefault(pair_key(pins()), {})[generator] = True
    save_state(state, f"chore(ci): record failed pair at {generator}")


def save_state(state: dict, message: str) -> None:
    state_file = CACHE / "state/state.json"
    write_json(state_file, state)
    git("config", "user.name", "github-actions[bot]")
    git("config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com")
    git("add", "--all")
    if git("diff", "--cached", "--quiet", check=False).returncode:
        git("commit", "-m", message)
        git("push", "origin", "HEAD:state")


def release_snapshot() -> None:
    # Site groups single-file exports by the immutable snapshot id.
    snapshot = os.environ["SNAPSHOT_ID"]
    singles = ROOT / "dist/single" / snapshot
    expected = {
        f"{path.parent.parent.name}--{path.parent.name}.html"
        for path in (CACHE / "bundle/data" / snapshot).glob("*/*/ir.json")
    }
    if not expected or {path.name for path in singles.glob("*.html")} != expected:
        raise ValueError("Single-file exports do not cover exactly the clean catalog cells")
    release(snapshot, singles)


if __name__ == "__main__":
    commands = {
        function.__name__: function
        for function in (
            plan,
            cell,
            collect,
            restore,
            publish,
            release_snapshot,
            record_failure,
        )
    }
    commands[sys.argv[1]]()
