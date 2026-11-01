"""Create the isolated extraction interpreter; never build either upstream package."""

import os
import re
import subprocess
from pathlib import Path

NPU_ONLY = {"torch-npu", "triton", "triton-ascend", "memfabric-hybrid", "memcache-hybrid"}
CPU_PACKAGES = {"torch", "torchvision", "torchaudio"}


def requirements(text: str) -> tuple[dict[str, str], list[str]]:
    pins = {}
    optional = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"([A-Za-z0-9_.-]+)", line)
        if not match:
            raise ValueError(f"Unsupported upstream requirement: {line}")
        name = match[1].lower().replace("_", "-")
        if name in CPU_PACKAGES:
            pin = re.fullmatch(r"[\w-]+==([0-9]+\.[0-9]+\.[0-9]+)(?:\+cpu)?", line)
            if not pin:
                raise ValueError(f"CPU requirement must have an exact release pin: {line}")
            pins[name] = pin[1]
        elif name not in NPU_ONLY:
            optional.append(line)
    if "torch" not in pins:
        raise ValueError("Upstream did not declare a torch pin")
    return pins, optional


def main() -> None:
    src = Path(os.environ["INFERATLAS_SRC"]).resolve()
    python = Path(os.environ["INFERATLAS_PY"]).resolve()
    pins, optional = requirements((src / "vllm-ascend/requirements.txt").read_text())
    scratch = Path(".cache/cpu-deps")
    scratch.mkdir(parents=True, exist_ok=True)
    constraints = scratch / "constraints.txt"
    constraints.write_text("".join(f"{name}=={pin}+cpu\n" for name, pin in sorted(pins.items())))
    subprocess.run(["uv", "venv", "--python", "3.12", str(python.parent.parent)], check=True)
    install = [
        "uv",
        "pip",
        "install",
        "--python",
        str(python),
        "--index-url",
        "https://pypi.org/simple/",
        "--extra-index-url",
        "https://download.pytorch.org/whl/cpu",
        "--index-strategy",
        "unsafe-best-match",
        "--constraint",
        str(constraints),
    ]
    subprocess.run(
        [*install, *[f"{name}=={pin}+cpu" for name, pin in sorted(pins.items())]], check=True
    )
    subprocess.run([*install, "-r", str(src / "vllm/requirements/common.txt")], check=True)
    # Match the experiment: NPU-adjacent optional deps may be unavailable on CPU.
    skipped = []
    for requirement in optional:
        if subprocess.run([*install, requirement], check=False).returncode:
            print(f"::warning::CPU-only extraction skipped optional dependency: {requirement}")
            skipped.append(requirement)
    subprocess.run(
        [
            *install,
            "pytest",
            "setuptools-scm==10.3.4",
            "runai-model-streamer",
            "librosa",
            "soundfile",
        ],
        check=True,
    )
    (scratch / "skipped.txt").write_text("".join(f"{item}\n" for item in sorted(skipped)))
    purelib = subprocess.check_output(
        [str(python), "-I", "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        text=True,
    ).strip()
    (Path(purelib) / "inferatlas_upstream.pth").write_text(f"{src / 'vllm'}\n")
    subprocess.run(
        [
            str(python),
            "-I",
            "-c",
            "import torch, vllm; assert torch.version.cuda is None; "
            "assert not torch.cuda.is_available(); print(torch.__version__, vllm.__version__)",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
