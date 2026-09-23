import type { TreeNode } from "../types";
import { formatBigInt } from "../format";

interface LaidOut {
  node: TreeNode;
  depth: number;
  x: number; // 0-based leaf order position for leaves; centroid for internal
  leafIndex: number; // -1 for internal
  children?: [LaidOut, LaidOut];
}

const LEAF_W = 84;
const LEAF_H = 40;
const LEAF_GAP = 18;
const LEVEL_H = 96;
const MARGIN = 28;

function layout(node: TreeNode, depth = 0, next = { v: 0 }): LaidOut {
  if (node.type === "leaf") {
    const x = next.v;
    next.v += 1;
    return { node, depth, x, leafIndex: x };
  }
  const left = layout(node.left!, depth + 1, next);
  const right = layout(node.right!, depth + 1, next);
  return {
    node,
    depth,
    x: (left.x + right.x) / 2,
    leafIndex: -1,
    children: [left, right],
  };
}

function collect(n: LaidOut, out: LaidOut[] = []): LaidOut[] {
  out.push(n);
  if (n.children) {
    collect(n.children[0], out);
    collect(n.children[1], out);
  }
  return out;
}

function depthOf(n: LaidOut): number {
  if (!n.children) return n.depth;
  return Math.max(n.depth, depthOf(n.children[0]), depthOf(n.children[1]));
}

const NODE_FILL: Record<string, string> = {
  leaf: "#0ea5e9",
  node: "#7c3aed",
};

export function TreeView({ tree }: { tree: TreeNode }) {
  const root = layout(tree);
  const nodes = collect(root);
  const leafCount = nodes.filter((n) => n.leafIndex >= 0).length;
  const width = MARGIN * 2 + leafCount * LEAF_W + (leafCount - 1) * LEAF_GAP;
  const height = MARGIN * 2 + (depthOf(root) + 1) * LEVEL_H;

  const xPos = (n: LaidOut) => MARGIN + n.x * (LEAF_W + LEAF_GAP) + LEAF_W / 2;
  const yPos = (n: LaidOut) => MARGIN + n.depth * LEVEL_H + (n.leafIndex >= 0 ? 0 : 4);

  const edges: JSX.Element[] = [];
  const shapes: JSX.Element[] = [];

  for (const n of nodes) {
    const cx = xPos(n);
    const cy = yPos(n);
    if (n.children) {
      for (const child of n.children) {
        edges.push(
          <line
            key={`${cx}-${cy}-${xPos(child)}-${yPos(child)}`}
            x1={cx}
            y1={cy + 18}
            x2={xPos(child)}
            y2={yPos(child) - 18}
            stroke="#94a3b8"
            strokeWidth={2}
          />
        );
      }
      const c = n.node.contraction;
      shapes.push(
        <g key={`n-${cx}-${cy}`}>
          <rect
            x={cx - 52}
            y={cy - 18}
            width={104}
            height={36}
            rx={8}
            fill="#ede9fe"
            stroke={NODE_FILL.node}
            strokeWidth={1.5}
          />
          <text x={cx} y={cy - 2} textAnchor="middle" fontSize={11} fill="#5b21b6">
            乘法 {formatBigInt(c!.multiplications)}
          </text>
          <text x={cx} y={cy + 12} textAnchor="middle" fontSize={11} fill="#5b21b6">
            结果 {formatBigInt(c!.result_size)}
          </text>
        </g>
      );
    } else {
      shapes.push(
        <g key={`l-${cx}-${cy}`}>
          <rect
            x={cx - LEAF_W / 2}
            y={cy - LEAF_H / 2}
            width={LEAF_W}
            height={LEAF_H}
            rx={8}
            fill="#e0f2fe"
            stroke={NODE_FILL.leaf}
            strokeWidth={1.5}
          />
          <text x={cx} y={cy + 5} textAnchor="middle" fontSize={14} fontWeight={600} fill="#075985">
            {n.node.name}
          </text>
        </g>
      );
    }
  }

  return (
    <div className="tree-scroll">
      <svg width={width} height={height} role="img" aria-label="收缩树图">
        {edges}
        {shapes}
      </svg>
    </div>
  );
}
