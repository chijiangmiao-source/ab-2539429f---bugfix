import type { PlanResult } from "../types";
import { formatBigInt, magnitudeHint } from "../format";
import { TreeView } from "./TreeView";

const CUT_STYLE: Record<string, string> = {
  mandatory: "badge mandatory",
  optional: "badge optional",
  absent: "badge absent",
};

function MetricCard({
  title,
  value,
  hint,
  tone,
}: {
  title: string;
  value: string;
  hint?: string;
  tone?: string;
}) {
  return (
    <div className={`metric ${tone ?? ""}`}>
      <div className="metric-title">{title}</div>
      <div className="metric-value">{value}</div>
      {hint && <div className="metric-hint">{hint}</div>}
    </div>
  );
}

export function ResultView({ result }: { result: PlanResult }) {
  const { summary } = result;
  return (
    <div className="results">
      <section className="panel">
        <h2>② 结论：先最小化峰值，再最小化总乘法数</h2>
        <div className="metric-grid">
          <MetricCard
            title="峰值内存（最大张量大小）"
            value={formatBigInt(summary.peak_memory)}
            hint={magnitudeHint(summary.peak_memory)}
            tone="primary"
          />
          <MetricCard
            title="总乘法数"
            value={formatBigInt(summary.total_multiplications)}
            hint={magnitudeHint(summary.total_multiplications)}
          />
          <MetricCard
            title="同优树数量"
            value={formatBigInt(summary.num_cotrees)}
            hint="峰值与总乘法数均最优的不同括号化数目"
          />
          <MetricCard title="张量数量" value={String(summary.num_tensors)} />
        </div>
        <div className="canonical">
          <span className="canon-label">规范括号串：</span>
          <code>{summary.canonical}</code>
        </div>
      </section>

      <section className="panel">
        <h2>③ 收缩树图</h2>
        <p className="panel-note">
          叶节点为输入张量；内部节点标注该次收缩的乘法数与结果张量大小。规范序按叶名递归排序子树。
        </p>
        <TreeView tree={result.tree} />
      </section>

      <section className="panel">
        <h2>④ 收缩步骤（指标证据）</h2>
        <table className="data-table">
          <thead>
            <tr>
              <th>#</th>
              <th>左组</th>
              <th>右组</th>
              <th>消去索引</th>
              <th>边界并集（乘法索引）</th>
              <th>结果索引</th>
              <th>乘法数</th>
              <th>结果大小</th>
            </tr>
          </thead>
          <tbody>
            {result.steps.map((s) => (
              <tr key={s.order}>
                <td>{s.order}</td>
                <td className="names">{s.left.join(", ")}</td>
                <td className="names">{s.right.join(", ")}</td>
                <td className="idx mono">{s.contracted_indices.join(", ") || "—"}</td>
                <td className="idx mono">{s.union_boundary.join(", ")}</td>
                <td className="idx mono">{s.result_indices.join(", ") || "∅（标量）"}</td>
                <td className="num">{formatBigInt(s.multiplications)}</td>
                <td className="num">{formatBigInt(s.result_size)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section className="panel">
        <h2>⑤ 非平凡根切分分类</h2>
        <p className="panel-note">
          枚举全部 {result.cuts.length} 个叶集二部切分。
          <span className="legend">
            <span className="badge mandatory">必现</span> 所有同优树都在此切分；
            <span className="badge optional">可选</span> 存在同优树采用、也存在不采用；
            <span className="badge absent">不出现</span> 任何同优树都不采用（附强制采用时的代价证据）。
          </span>
        </p>
        <table className="data-table cuts">
          <thead>
            <tr>
              <th>切分</th>
              <th>分类</th>
              <th>证据</th>
            </tr>
          </thead>
          <tbody>
            {result.cuts.map((c, i) => (
              <tr key={i} className={`cut-row ${c.classification}`}>
                <td className="cut-sides">
                  <div className="side">{c.left.join(", ")}</div>
                  <div className="cut-divider">│</div>
                  <div className="side">{c.right.join(", ")}</div>
                </td>
                <td>
                  <span className={CUT_STYLE[c.classification]}>{c.label}</span>
                </td>
                <td className="evidence">{c.evidence}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section className="panel">
        <h2>⑥ 指标与输入清单</h2>
        <h3>索引（出现次数与维数）</h3>
        <table className="data-table">
          <thead>
            <tr>
              <th>索引</th>
              <th>维数</th>
              <th>全网出现次数</th>
              <th>所在张量</th>
            </tr>
          </thead>
          <tbody>
            {result.indices.map((idx) => (
              <tr key={idx.name}>
                <td className="mono">{idx.name}</td>
                <td className="num">{idx.dimension}</td>
                <td className="num">{idx.occurrences}（{idx.occurrences === 2 ? "收缩边" : "悬空"}）</td>
                <td className="names">{idx.on_tensors.join(", ")}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <h3>输入张量大小（峰值计入）</h3>
        <table className="data-table">
          <thead>
            <tr>
              <th>张量</th>
              <th>索引</th>
              <th>大小</th>
            </tr>
          </thead>
          <tbody>
            {result.inputs.map((t) => (
              <tr key={t.name}>
                <td className="names">{t.name}</td>
                <td className="idx mono">{t.indices.join(", ")}</td>
                <td className="num">{formatBigInt(t.size)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  );
}
