#!/usr/bin/env python3
"""Fetch only config.json, pin its HF commit and hashes, and update the catalog.

uv run --project generator python scripts/fetch-config.py SLUG HF_ID FAMILY
uv run --project generator python scripts/fetch-config.py --check SLUG
The mirror is tried first; redirects/fallback to huggingface.co are allowed.
Provenance is a YAML comment to preserve the shared CatalogModel contract.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[1]
HOSTS = ("https://hf-mirror.com", "https://huggingface.co")


def fetch(hf_id: str, revision: str) -> tuple[bytes, str, str]:
    errors = []
    for host in HOSTS:
        url = f"{host}/{hf_id}/resolve/{revision}/config.json"
        try:
            with urlopen(url, timeout=45) as response:
                raw = response.read(1_000_001)
                commit = response.headers.get("x-repo-commit")
                if not commit:
                    match = re.search(r"/([0-9a-f]{40})/config.json", response.url)
                    commit = match[1] if match else revision
            if len(raw) > 1_000_000:
                raise ValueError("config.json exceeds 1 MB")
            if not re.fullmatch(r"[0-9a-f]{40}", commit):
                raise ValueError(f"server did not identify an immutable commit: {url}")
            config = json.loads(raw)
            if not isinstance(config, dict) or not config.get("model_type"):
                raise ValueError("expected a model configuration object")
            return raw, commit, f"{host}/{hf_id}/resolve/{commit}/config.json"
        except (HTTPError, URLError, TimeoutError, ValueError) as error:
            errors.append(f"{url}: {error}")
    raise RuntimeError("\n".join(errors))


def canonical_hash(config: dict) -> str:
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def update_catalog(slug, hf_id, family, config, digest):
    from ruamel.yaml.comments import CommentedSeq

    path = ROOT / "catalog.yaml"
    before, rest = re.split(r"^models:[^\n]*\n", path.read_text(), maxsplit=1, flags=re.MULTILINE)
    _, after = re.split(r"(?=^version_pairs:)", rest, maxsplit=1, flags=re.MULTILINE)
    models = YAML(typ="safe").load(path)["models"]
    models = [model for model in models if model["id"] != slug]
    models.append(
        dict(
            config=config,
            config_rev=f"inline:sha256:{digest}",
            family=family,
            hf_id=hf_id,
            id=slug,
        )
    )
    # Rebuild comments after sorting; ruamel sequence comments are indexed by position.
    models = CommentedSeq(
        json.loads(json.dumps(model, sort_keys=True))
        for model in sorted(models, key=lambda model: model["id"])
    )
    for index, model in enumerate(models):
        source_path = ROOT / "fixtures" / model["id"] / "config.source.json"
        if source_path.is_file():
            source = json.loads(source_path.read_text())
            models.yaml_set_comment_before_after_key(
                index,
                before="\n".join(f"{key}: {source[key]}" for key in sorted(source)),
                indent=2,
            )
    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.width = 110
    stream = io.StringIO()
    yaml.dump(dict(models=models), stream)
    path.write_text(before + stream.getvalue() + after)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slug")
    parser.add_argument("hf_id", nargs="?")
    parser.add_argument("family", nargs="?", choices=("dense", "moe", "mla", "hybrid", "vlm"))
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--check", action="store_true", help="re-fetch pinned bytes and verify hashes"
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", args.slug):
        parser.error("slug must be a lowercase filesystem-safe identifier")
    directory = ROOT / "fixtures" / args.slug
    provenance_path = directory / "config.source.json"
    if args.check:
        source = json.loads(provenance_path.read_text())
        raw, revision, _ = fetch(source["hf_id"], source["revision"])
        if (revision != source["revision"] or hashlib.sha256(raw).hexdigest() != source["sha256"]
                or raw != (directory / "config.json").read_bytes()):
            raise ValueError("Pinned config verification failed")
        print(f"{args.slug}: pinned config verified")
        return
    if not args.hf_id or not args.family:
        parser.error("HF_ID and FAMILY are required when fetching")
    raw, revision, url = fetch(args.hf_id, args.revision)
    config = json.loads(raw)
    digest = canonical_hash(config)
    source = dict(
        hf_id=args.hf_id,
        revision=revision,
        sha256=hashlib.sha256(raw).hexdigest(),
        canonical_sha256=digest,
        source_url=url,
    )
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_bytes(raw)
    provenance_path.write_text(json.dumps(source, sort_keys=True, indent=2) + "\n")
    update_catalog(args.slug, args.hf_id, args.family, config, digest)
    print(f"{args.slug}: {revision} sha256:{source['sha256']}")


if __name__ == "__main__":
    main()
