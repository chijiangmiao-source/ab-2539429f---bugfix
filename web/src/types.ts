// Costs are exact integers on the wire; values beyond the safe Number
// range arrive as bigint, ordinary-sized ones as number (see api.ts).
export type IntLike = number | bigint;

export interface IndexInput {
  name: string;
  dimension: number | string;
}

export interface TensorInput {
  name: string;
  indices: IndexInput[];
}

export interface NetworkPayload {
  tensors: TensorInput[];
}

export interface ApiError {
  code: string;
  message: string;
  path: (string | number)[];
}

export interface TreeNode {
  type: "leaf" | "node";
  name?: string;
  left?: TreeNode;
  right?: TreeNode;
  contraction?: {
    multiplications: IntLike;
    result_size: IntLike;
    result_indices: string[];
    contracted_indices: string[];
  };
}

export interface Step {
  order: number;
  left: string[];
  right: string[];
  multiplications: IntLike;
  result_size: IntLike;
  result_indices: string[];
  contracted_indices: string[];
  union_boundary: string[];
}

export type CutClass = "mandatory" | "optional" | "absent";

export interface Cut {
  left: string[];
  right: string[];
  classification: CutClass;
  label: string;
  evidence: string;
  optimal_root_cuts: number;
}

export interface IndexInfo {
  name: string;
  dimension: number;
  occurrences: number;
  on_tensors: string[];
}

export interface InputInfo {
  name: string;
  size: IntLike;
  indices: string[];
}

export interface PlanResult {
  summary: {
    peak_memory: IntLike;
    total_multiplications: IntLike;
    canonical: string;
    num_cotrees: IntLike;
    num_tensors: number;
  };
  tree: TreeNode;
  steps: Step[];
  cuts: Cut[];
  indices: IndexInfo[];
  inputs: InputInfo[];
}
