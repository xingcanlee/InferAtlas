"""Regenerate architecture goldens from real CPU/meta construction (never weights)."""

import hashlib
import json
import subprocess

from inferatlas.arch import emit_arch
from inferatlas.cli import annotate_stage, lower_stage
from inferatlas.extract import extract
from inferatlas.resolve import get_model, resolve
from inferatlas.schema import Extract
from inferatlas.schema.models import ExtractModule, ExtractParam, ModelRef
from inferatlas.verify.io import ROOT, read_json, write_json


def synthetic_facts():
    """A small dense→MoE MLA stack, explicitly synthetic rather than a measured model."""
    base = Extract.model_validate(read_json(ROOT / "fixtures/qwen3-30b-a3b/extract.json"))
    base.config = {
        "model_type": "synthetic",
        "architectures": ["SyntheticHeterogeneousForCausalLM"],
        "hidden_size": 16,
        "vocab_size": 32,
        "num_hidden_layers": 5,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": 8,
        "intermediate_size": 48,
        "moe_intermediate_size": 12,
        "n_routed_experts": 8,
        "num_experts_per_tok": 2,
        "n_shared_experts": 1,
        "q_lora_rank": 8,
        "kv_lora_rank": 4,
        "qk_nope_head_dim": 4,
        "qk_rope_head_dim": 4,
        "v_head_dim": 4,
        "rope_theta": 10000,
        "hidden_act": "silu",
        "torch_dtype": "bfloat16",
        "scoring_func": "sigmoid",
    }
    digest = hashlib.sha256(
        json.dumps(base.config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    base.snapshot.model = ModelRef(
        hf_id="synthetic/heterogeneous-mla", config_rev=f"inline:sha256:{digest}"
    )
    base.fixture = True
    base.ascend = None
    base.layer_types = ["full_attention"] * 5
    modules = []

    def module(path, cls, shape=None):
        modules.append(
            ExtractModule(
                path=path,
                class_name=cls,
                params=[ExtractParam(name="weight", shape=shape, dtype="bfloat16")]
                if shape
                else [],
            )
        )

    layers = "vllm.model_executor.layers."
    deepseek = "vllm.model_executor.models.deepseek_v2."
    module("", "synthetic.SyntheticHeterogeneousForCausalLM")
    module("model", "synthetic.Model")
    module(
        "model.embed_tokens", layers + "vocab_parallel_embedding.VocabParallelEmbedding", [32, 16]
    )
    module("model.layers", "torch.nn.modules.container.ModuleList")
    module("model.norm", layers + "layernorm.RMSNorm", [16])
    module("lm_head", layers + "vocab_parallel_embedding.ParallelLMHead", [32, 16])
    for index in range(5):
        root = f"model.layers.{index}"
        module(root, deepseek + "DeepseekV2DecoderLayer")
        for name in ("input_layernorm", "post_attention_layernorm"):
            module(root + "." + name, layers + "layernorm.RMSNorm", [16])
        module(root + ".self_attn", deepseek + "DeepseekV2MLAAttention")
        module(root + ".mlp", deepseek + ("DeepseekV2MLP" if index < 2 else "DeepseekV2MoE"))
        if index >= 2:
            module(root + ".mlp.shared_experts", deepseek + "DeepseekV2MLP")
    base.modules = sorted(modules, key=lambda m: m.path)
    return base


def main():
    facts = synthetic_facts()
    ir, triage = lower_stage([facts])
    arch = emit_arch(facts, ir, triage, model_id="synthetic-heterogeneous", family="hybrid")
    write_json(ROOT / "generator/tests/fixtures_arch/synthetic-heterogeneous.json", arch)
    for slug in ("qwen3-8b", "qwen3-30b-a3b"):
        model = get_model(slug)
        for device in ("A2", "H100"):
            snapshot = resolve(slug, device=device)
            cache = ROOT / ".cache/S1" / slug / device / "extract.json"
            if cache.is_file():
                facts = Extract.model_validate(read_json(cache))
            else:
                facts = extract(snapshot, slug, device=None if device == "H100" else device)
                write_json(cache, facts)
            if (
                facts.fixture
                or facts.snapshot != snapshot
                or facts.config != model.config
                or facts.parallel.tp != 1
                or facts.parallel.ep
            ):
                raise ValueError(
                    f"{slug}/{device}: cached facts disagree with the pinned catalog snapshot"
                )
            ir, triage = lower_stage([facts])
            ir, triage = annotate_stage(ir, [facts], triage)
            arch = emit_arch(facts, ir, triage, model_id=slug, family=model.family)
            write_json(ROOT / "golden" / slug / device / "arch.json", arch)
            if device == "A2":
                write_json(ROOT / "generator/tests/fixtures_arch" / f"{slug}.json", arch)
            print(
                f"{slug}/{device}: {len(facts.modules)} modules, {len(arch.notes)} notes",
                flush=True,
            )

    subprocess.run(
        ["pnpm", "exec", "biome", "format", "--write", "generator/tests/fixtures_arch"],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
