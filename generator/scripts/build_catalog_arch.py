"""Build the catalog's real CPU/meta structure bundles, reusing validated extraction caches."""

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import inferatlas.cli as cli
from inferatlas.extract import extract
from inferatlas.resolve import get_model, load_catalog
from inferatlas.schema import Extract
from inferatlas.verify.io import ROOT, read_json, write_json


def cell(slug, device):
    def cached(snapshot, tp, ep=False):
        model = get_model(slug)
        path = ROOT / ".cache/S1" / slug / device / f"extract-tp{tp}-v11.json"
        if path.is_file():
            facts = Extract.model_validate(read_json(path))
            if (
                facts.snapshot != snapshot
                or facts.config != model.config
                or facts.parallel.tp != tp
            ):
                raise ValueError("Cached extraction identity mismatch")
            return facts
        facts = extract(snapshot, model, tp=tp, ep=ep, device=None if device == "H100" else device)
        write_json(path, facts)
        return facts

    cli.extract_stage = cached
    directory, code = cli.build_pipeline(
        slug, device, "main-verified", ROOT / ".cache/S1-builds" / device / slug, [1]
    )
    print(
        f"{slug}/{device}: exit={code} structure={(directory / 'arch.json').is_file()}", flush=True
    )
    return code


def main():
    if len(sys.argv) == 3:
        raise SystemExit(cell(*sys.argv[1:]))
    cells = [(m.id, d) for m in load_catalog().models for d in ("H100", "A2")]

    def run(args):
        result = subprocess.run([sys.executable, __file__, *args], capture_output=True, text=True)
        log = ROOT / ".cache/S1" / args[0] / args[1] / "build.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(result.stdout + result.stderr)
        print(
            f"{args}: {result.returncode}\n{result.stdout[-500:]}{result.stderr[-1000:] if result.returncode else ''}",
            flush=True,
        )
        return result.returncode

    with ThreadPoolExecutor(max_workers=4) as pool:
        codes = list(pool.map(run, cells))
    raise SystemExit(int(any(codes)))


if __name__ == "__main__":
    main()
