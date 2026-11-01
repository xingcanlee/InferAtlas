/** @inferatlas/arch-ui primitives: render, variants, accessibility and shape evaluation. */
import {
  ArchFigureProvider,
  Attention,
  Band,
  BlockStack,
  DecoderBlock,
  evaluateShape,
  FigureBranch,
  FigureNode,
  formatNumeric,
  formatSymbolic,
  Generic,
  MLP,
  ModelFigure,
  MoE,
  Norm,
  NoteBadge,
  PatternGroup,
  ShapeLabel,
  StatusBadge,
  TensorEdge,
} from "@inferatlas/arch-ui";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { evaluator } from "@/lib/arch";

const slot = (c: HTMLElement, name: string) => c.querySelectorAll(`[data-slot="${name}"]`);

describe("shape helpers", () => {
  const ev = evaluator({ T: 4096, H: 2048, Hq: 32, Hkv: 4, D: 128, tp: 2 });
  it("formats symbolic and evaluated shapes", () => {
    const shape = ["T", "(Hq+2*Hkv)*D/tp"];
    expect(formatSymbolic(shape)).toBe("[T, (Hq+2*Hkv)*D/tp]");
    expect(formatNumeric(evaluateShape(shape, ev) ?? [])).toBe("[4096, 2560]");
  });
  it("marks dimensions that cannot be evaluated", () => {
    expect(formatNumeric(evaluateShape(["T", "nope"], ev) ?? [])).toBe("[4096, ?]");
    expect(evaluateShape(["T"], undefined)).toBeUndefined();
  });
});

describe("ShapeLabel", () => {
  it("shows the symbolic shape and exposes the numeric one; click pins it", async () => {
    render(
      <ArchFigureProvider evaluate={evaluator({ T: 8, H: 16 })}>
        <ShapeLabel shape={["T", "H"]} name="hidden" />
      </ArchFigureProvider>,
    );
    const label = screen.getByRole("button", { name: /hidden \[T, H\] = \[8, 16\]/ });
    expect(label).toHaveAttribute("data-numeric", "[8, 16]");
    expect(screen.getByTestId("shape-text")).toHaveTextContent("[T, H]");
    expect(screen.getByRole("tooltip", { hidden: true })).toHaveTextContent("[8, 16]");
    await userEvent.click(label);
    expect(label).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByTestId("shape-text")).toHaveTextContent("[8, 16]");
  });
  it("works without an evaluator (symbolic only)", () => {
    render(<ShapeLabel shape={["T"]} />);
    expect(screen.queryByRole("tooltip", { hidden: true })).toBeNull();
  });
});

describe("components", () => {
  it("attention variants render their badge, heads and extras", () => {
    const { container, rerender } = render(
      <Attention variant="gqa" headsQ={32} headsKv={8} headDim={128} rope="RoPE θ 1e6" qkNorm />,
    );
    expect(screen.getByText("GQA")).toBeInTheDocument();
    expect(screen.getByText("heads 32 q · 8 kv · d 128")).toBeInTheDocument();
    expect(screen.getByText("QK-Norm")).toBeInTheDocument();
    expect(slot(container, "latent-path")).toHaveLength(0);
    rerender(
      <Attention
        variant="mla"
        headsQ={128}
        headsKv={128}
        latent={{ qRank: 1536, kvRank: 512, ropeDim: 64 }}
      />,
    );
    expect(screen.getByText("MLA")).toBeInTheDocument();
    expect(slot(container, "latent-path")).toHaveLength(1);
    expect(screen.getByText("c_KV 512")).toHaveAttribute("data-tone", "accent");
    expect(screen.getByText("c_Q 1,536")).toBeInTheDocument();
    // MHA-equal kv heads are not repeated.
    expect(screen.getByText("heads 128 q")).toBeInTheDocument();
    rerender(<Attention variant="sliding_window" window={4096} />);
    expect(screen.getByText("window 4,096")).toBeInTheDocument();
    rerender(<Attention variant="sparse" indexTopk={2048} />);
    expect(screen.getByText("index top-2,048")).toBeInTheDocument();
    rerender(<Attention variant="linear" detail="Gated DeltaNet" />);
    expect(screen.getByText("Linear")).toBeInTheDocument();
    expect(screen.getByText("Linear attention")).toBeInTheDocument();
  });

  it("MLP shows gate/up/down for gated variants only", () => {
    const { container, rerender } = render(<MLP variant="swiglu" intermediate={12288} />);
    expect(screen.getByText("SwiGLU")).toBeInTheDocument();
    expect(screen.getByText("intermediate 12,288")).toBeInTheDocument();
    expect(screen.getByText("gate")).toBeInTheDocument();
    rerender(<MLP variant="gelu" />);
    expect(screen.queryByText("gate")).toBeNull();
    expect(slot(container, "mlp-projections")).toHaveLength(1);
  });

  it("MoE draws router, experts and the shared expert", () => {
    const { container } = render(
      <MoE
        experts={256}
        topK={8}
        sharedExperts={1}
        scoring="sigmoid"
        groups={{ groups: 8, topkGroups: 4 }}
        expert={{ variant: "swiglu", intermediate: 2048 }}
      />,
    );
    expect(screen.getByText("256E · top-8")).toBeInTheDocument();
    expect(screen.getByText("sigmoid · top-8 · 4/8 groups")).toBeInTheDocument();
    expect(screen.getByText("S")).toHaveAttribute("data-tone", "shared");
    expect(screen.getByText("256")).toBeInTheDocument();
    expect(container.querySelectorAll("[data-active]").length).toBeGreaterThan(0);
    // The highlighted routing is marked as illustrative.
    expect(slot(container, "moe-experts")[0]).toHaveAttribute("data-illustrative", "true");
    expect(screen.getByText("e.g.")).toBeInTheDocument();
    expect(screen.getByText(/256 routed · 1 shared · SwiGLU · 2,048/)).toBeInTheDocument();
  });

  it("generic shows the class name, children and a dashed outline", () => {
    const kids = Array.from({ length: 8 }, (_, i) => ({ name: `c${i}`, class_name: "X" }));
    const { container } = render(<Generic moduleClass="FooLayer" members={kids} />);
    expect(screen.getByText("FooLayer")).toBeInTheDocument();
    expect(screen.getByText("+2")).toBeInTheDocument();
    expect(container.querySelector('[data-kind="generic"]')?.className).toMatch(/border-dashed/);
  });

  it("interactive nodes are buttons with aria-pressed and status markers", async () => {
    const onSelect = vi.fn();
    render(<Norm onSelect={onSelect} selected status="unannotated_override" />);
    const btn = screen.getByRole("button", { pressed: true });
    expect(btn).toHaveTextContent("RMSNorm");
    expect(screen.getByRole("img", { name: "status: unannotated override" })).toBeInTheDocument();
    btn.focus();
    await userEvent.keyboard("{Enter}");
    expect(onSelect).toHaveBeenCalledTimes(1);
  });

  it("ruled status draws no marker; badges carry text, not only colour", () => {
    const { container } = render(<FigureNode title="x" status="ruled" />);
    expect(slot(container, "status-badge")).toHaveLength(0);
    render(<StatusBadge status="ascend_override" />);
    expect(screen.getByText("Ascend override")).toBeInTheDocument();
    render(<NoteBadge severity="warn" count={2} />);
    expect(screen.getByRole("img", { name: "2 warnings" })).toBeInTheDocument();
  });
});

describe("DecoderBlock", () => {
  it("pre-norm: two residual branches, each closed by ⊕", () => {
    const { container } = render(
      <DecoderBlock residual="pre_norm" mixer={<Attention variant="gqa" />} ffn={<MLP />} />,
    );
    expect(slot(container, "residual-branch")).toHaveLength(2);
    expect(slot(container, "residual-add")).toHaveLength(2);
    expect(slot(container, "block-step")).toHaveLength(0);
    // Norm sits inside the branch, before the mixer.
    const first = slot(container, "residual-branch")[0] as HTMLElement;
    const kinds = [...first.querySelectorAll("[data-kind]")].map((e) =>
      e.getAttribute("data-kind"),
    );
    expect(kinds).toEqual(["norm", "attention"]);
  });

  it("post-norm: norms on the main stream after each ⊕", () => {
    const { container } = render(
      <DecoderBlock residual="post_norm" mixer={<Attention variant="mha" />} ffn={<MLP />} />,
    );
    expect(slot(container, "residual-branch")).toHaveLength(2);
    expect(slot(container, "block-step")).toHaveLength(2);
    const step = slot(container, "block-step")[0] as HTMLElement;
    expect(step.querySelector('[data-kind="norm"]')).not.toBeNull();
  });

  it("unknown residual style: plain chain, no skip line", () => {
    const { container } = render(
      <DecoderBlock residual="unknown" mixer={<Attention variant="gqa" />} ffn={<MLP />} />,
    );
    expect(slot(container, "residual-add")).toHaveLength(0);
  });
});

describe("figure layout", () => {
  it("ModelFigure inserts arrows between nodes and labels tensor edges", () => {
    const { container } = render(
      <ModelFigure evaluate={evaluator({ T: 4, H: 8 })}>
        <Norm />
        <TensorEdge shape={["T", "H"]} name="hidden" />
        <MLP />
        <Norm />
      </ModelFigure>,
    );
    // explicit TensorEdge between 1-2 (no duplicate arrow), auto arrow between 2-3
    expect(slot(container, "flow-arrow")).toHaveLength(1);
    expect(slot(container, "tensor-edge")).toHaveLength(1);
    expect(screen.getByRole("button", { name: /hidden \[T, H\] = \[4, 8\]/ })).toBeInTheDocument();
  });

  it("FigureBranch keeps the spine node and attaches the side head", () => {
    const { container } = render(
      <ModelFigure>
        <FigureBranch node={<Norm />} branch={<Generic moduleClass="MTP" />} label="side" />
      </ModelFigure>,
    );
    expect(slot(container, "figure-branch")).toHaveLength(1);
    expect(slot(container, "figure-branch-content")[0]).toHaveTextContent("MTP");
    expect(screen.getByText("side")).toBeInTheDocument();
  });

  it("bands show N×, a disclosure header and pattern groups their period", async () => {
    const onSelect = vi.fn();
    const { container } = render(
      <BlockStack layers={48}>
        <PatternGroup title="layers 0–47" repeats={12}>
          <Band title="linear" count={3} onSelect={onSelect}>
            <Norm />
          </Band>
          <Band title="full" count={1} selected>
            <Norm />
          </Band>
        </PatternGroup>
      </BlockStack>,
    );
    expect(screen.getByText("repeated 12 times")).toBeInTheDocument();
    expect(screen.getByText("repeated 3 times")).toBeInTheDocument();
    expect(slot(container, "band-then")).toHaveLength(1);
    const header = screen.getByRole("button", { name: /linear/ });
    expect(header).toHaveAttribute("aria-expanded", "false");
    await userEvent.click(header);
    expect(onSelect).toHaveBeenCalled();
  });
});
