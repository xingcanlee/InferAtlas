/**
 * @inferatlas/arch-ui — shadcn-style primitives for paper-style LLM architecture figures.
 * Presentational only: no data fetching, no expression parsing, no per-model drawing code.
 * Import `@inferatlas/arch-ui/theme.css` once and let Tailwind scan `src/`.
 */
export { cn } from "./cn";
export {
  type ModuleStatus,
  NoteBadge,
  type NoteBadgeProps,
  type NoteSeverity,
  StatusBadge,
  type StatusBadgeProps,
  statusLabel,
  VariantBadge,
  type VariantBadgeProps,
} from "./components/badges";
export {
  Band,
  BandCard,
  type BandCardProps,
  BandList,
  type BandProps,
  BlockStack,
  type BlockStackProps,
  BlockStep,
  DecoderBlock,
  type DecoderBlockProps,
  HyperBranch,
  type HyperBranchProps,
  MhcBox,
  type MhcBoxProps,
  PatternGroup,
  type PatternGroupProps,
  RepeatCount,
  ResidualBranch,
  type ResidualBranchProps,
  ResidualBus,
  type ResidualStyle,
} from "./components/block";
export {
  FigureBranch,
  type FigureBranchProps,
  FigureTerminal,
  type FigureTerminalProps,
  ModelFigure,
  type ModelFigureProps,
} from "./components/figure";
export {
  Flow,
  FlowArrow,
  type FlowArrowProps,
  type FlowProps,
  TensorEdge,
  type TensorEdgeProps,
} from "./components/flow";
export {
  Attention,
  type AttentionProps,
  type AttentionVariant,
  attentionBadge,
  Embedding,
  type EmbeddingProps,
  Generic,
  type GenericProps,
  type LatentPath,
  LatentPathDiagram,
  LMHead,
  MLP,
  type MLPProps,
  type MlpVariant,
  MoE,
  type MoEProps,
  mlpBadge,
  Norm,
  type NormProps,
  type NormVariant,
  ResidualAdd,
  type ResidualAddProps,
} from "./components/layers";
export {
  Chip,
  type ComponentNodeProps,
  FigureNode,
  type FigureNodeKind,
  type FigureNodeProps,
  figureNode,
  type NodeMarkers,
} from "./components/node";
export {
  evaluateShape,
  formatNumber,
  formatNumeric,
  formatSymbolic,
  numel,
  type Shape,
  ShapeLabel,
  type ShapeLabelProps,
  TensorChip,
  type TensorChipProps,
} from "./components/shape";
export { type ArchFigureContextValue, ArchFigureProvider, useArchFigure } from "./context";
