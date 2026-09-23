import { useMemo, useState } from "react";
import type { ApiError, NetworkPayload } from "../types";

interface IndexDraft {
  name: string;
  dimension: string;
}
interface TensorDraft {
  name: string;
  indices: IndexDraft[];
}

const RING4: TensorDraft[] = [
  { name: "A", indices: [{ name: "i", dimension: "2" }, { name: "j", dimension: "2" }] },
  { name: "B", indices: [{ name: "j", dimension: "2" }, { name: "k", dimension: "2" }] },
  { name: "C", indices: [{ name: "k", dimension: "2" }, { name: "l", dimension: "2" }] },
  { name: "D", indices: [{ name: "l", dimension: "2" }, { name: "i", dimension: "2" }] },
];

const CHAIN5: TensorDraft[] = [
  { name: "A", indices: [{ name: "a", dimension: "2" }, { name: "i", dimension: "10" }] },
  { name: "B", indices: [{ name: "i", dimension: "10" }, { name: "j", dimension: "3" }] },
  { name: "C", indices: [{ name: "j", dimension: "3" }, { name: "k", dimension: "4" }] },
  { name: "D", indices: [{ name: "k", dimension: "4" }, { name: "m", dimension: "5" }] },
  { name: "E", indices: [{ name: "m", dimension: "5" }, { name: "e", dimension: "2" }] },
];

const BAD3: TensorDraft[] = [
  { name: "A", indices: [{ name: "i", dimension: "2" }] },
  { name: "B", indices: [{ name: "i", dimension: "2" }, { name: "j", dimension: "2" }] },
  { name: "C", indices: [{ name: "i", dimension: "2" }, { name: "j", dimension: "2" }] },
];

function errorKey(path: (string | number)[]): string {
  return path.map((p) => String(p)).join("/");
}

function toPayload(drafts: TensorDraft[]): NetworkPayload {
  return {
    tensors: drafts.map((t) => ({
      name: t.name,
      indices: t.indices.map((ix) => {
        const trimmed = ix.dimension.trim();
        const asInt = /^[+-]?\d+$/.test(trimmed) ? Number(trimmed) : trimmed;
        return { name: ix.name, dimension: asInt };
      }),
    })),
  };
}

function fromPayload(payload: NetworkPayload): TensorDraft[] {
  return payload.tensors.map((t) => ({
    name: t.name,
    indices: t.indices.map((i) => ({
      name: i.name,
      dimension: String(i.dimension),
    })),
  }));
}

function pathLabel(path: (string | number)[]): string {
  if (path.length === 0) return "整体";
  const labels: string[] = [];
  for (let i = 0; i < path.length; i++) {
    const p = path[i];
    if (p === "tensors") labels.push("张量");
    else if (p === "indices") labels.push("索引");
    else if (p === "name") labels.push("名称");
    else if (p === "dimension") labels.push("维数");
    else if (typeof p === "number") labels.push(`#${p + 1}`);
    else labels.push(String(p));
  }
  return labels.join(" › ");
}

export function NetworkEditor({
  errors,
  onSubmit,
  busy,
}: {
  errors: ApiError[];
  onSubmit: (payload: NetworkPayload) => void;
  busy: boolean;
}) {
  const [mode, setMode] = useState<"form" | "json">("form");
  const [drafts, setDrafts] = useState<TensorDraft[]>(RING4);
  const [jsonText, setJsonText] = useState<string>(() =>
    JSON.stringify(toPayload(RING4), null, 2)
  );
  const [jsonParseError, setJsonParseError] = useState<string | null>(null);

  const errorMap = useMemo(() => {
    const m = new Map<string, ApiError[]>();
    for (const e of errors) {
      const k = errorKey(e.path);
      m.set(k, [...(m.get(k) ?? []), e]);
    }
    return m;
  }, [errors]);

  const globalErrors = errors.filter((e) => e.path.length === 0 || e.path[0] === "tensors" && e.path.length === 1);

  function syncFromJson(text: string) {
    setJsonText(text);
    try {
      const parsed = JSON.parse(text) as NetworkPayload;
      if (parsed && Array.isArray(parsed.tensors)) {
        setDrafts(fromPayload(parsed));
        setJsonParseError(null);
      } else {
        setJsonParseError(null); // structural validation happens server-side
      }
    } catch (err) {
      setJsonParseError((err as Error).message);
    }
  }

  function updateDraft(next: TensorDraft[]) {
    setDrafts(next);
    setJsonText(JSON.stringify(toPayload(next), null, 2));
    setJsonParseError(null);
  }

  function loadExample(ex: TensorDraft[]) {
    updateDraft(ex);
  }

  function addTensor() {
    updateDraft([
      ...drafts,
      { name: `T${drafts.length + 1}`, indices: [{ name: "x", dimension: "2" }] },
    ]);
  }

  function removeTensor(i: number) {
    updateDraft(drafts.filter((_, idx) => idx !== i));
  }

  function updateTensor(i: number, patch: Partial<TensorDraft>) {
    updateDraft(drafts.map((t, idx) => (idx === i ? { ...t, ...patch } : t)));
  }

  function addIndex(i: number) {
    updateDraft(
      drafts.map((t, idx) =>
        idx === i ? { ...t, indices: [...t.indices, { name: "", dimension: "2" }] } : t
      )
    );
  }

  function removeIndex(ti: number, ii: number) {
    updateDraft(
      drafts.map((t, idx) =>
        idx === ti ? { ...t, indices: t.indices.filter((_, j) => j !== ii) } : t
      )
    );
  }

  function updateIndex(ti: number, ii: number, patch: Partial<IndexDraft>) {
    updateDraft(
      drafts.map((t, idx) =>
        idx === ti
          ? {
              ...t,
              indices: t.indices.map((x, j) => (j === ii ? { ...x, ...patch } : x)),
            }
          : t
      )
    );
  }

  function submit() {
    if (mode === "form") {
      onSubmit(toPayload(drafts));
    } else {
      try {
        onSubmit(JSON.parse(jsonText));
      } catch (err) {
        setJsonParseError((err as Error).message);
      }
    }
  }

  function fieldError(path: (string | number)[]): ApiError[] {
    return errorMap.get(errorKey(path)) ?? [];
  }

  function fieldClass(path: (string | number)[]) {
    return fieldError(path).length ? "input bad" : "input";
  }

  return (
    <section className="panel editor">
      <div className="panel-head">
        <h2>① 定义张量网络</h2>
        <div className="examples">
          <button type="button" className="linkbtn" onClick={() => loadExample(RING4)}>
            四元环示例
          </button>
          <button type="button" className="linkbtn" onClick={() => loadExample(CHAIN5)}>
            五元链示例
          </button>
          <button type="button" className="linkbtn warn" onClick={() => loadExample(BAD3)}>
            错误示例（索引出现 3 次）
          </button>
        </div>
      </div>

      <div className="tabs">
        <button
          type="button"
          className={mode === "form" ? "tab active" : "tab"}
          onClick={() => setMode("form")}
        >
          表单编辑
        </button>
        <button
          type="button"
          className={mode === "json" ? "tab active" : "tab"}
          onClick={() => {
            setJsonText(JSON.stringify(toPayload(drafts), null, 2));
            setMode("json");
          }}
        >
          JSON 编辑
        </button>
      </div>

      {globalErrors.length > 0 && (
        <div className="error-box">
          {globalErrors.map((e, i) => (
            <div key={i} className="error-line">
              <span className="error-path">{pathLabel(e.path)}</span>
              {e.message}
            </div>
          ))}
        </div>
      )}

      {mode === "form" ? (
        <div className="tensor-grid">
          {drafts.map((t, ti) => (
            <div className="tensor-card" key={ti}>
              <div className="tensor-card-head">
                <input
                  className={fieldClass(["tensors", ti, "name"])}
                  value={t.name}
                  placeholder={`张量 ${ti + 1} 名称`}
                  onChange={(e) => updateTensor(ti, { name: e.target.value })}
                />
                <button type="button" className="mini danger" onClick={() => removeTensor(ti)}>
                  删除
                </button>
              </div>
              {fieldError(["tensors", ti, "name"]).map((e, k) => (
                <div className="field-error" key={k}>{e.message}</div>
              ))}
              <table className="idx-table">
                <thead>
                  <tr>
                    <th>索引 (ASCII)</th>
                    <th>维数 (2–1000)</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {t.indices.map((ix, ii) => {
                    const nameErrs = fieldError(["tensors", ti, "indices", ii, "name"]);
                    const dimErrs = fieldError(["tensors", ti, "indices", ii, "dimension"]);
                    return (
                      <tr key={ii}>
                        <td>
                          <input
                            className={nameErrs.length ? "input bad" : "input"}
                            value={ix.name}
                            onChange={(e) => updateIndex(ti, ii, { name: e.target.value })}
                          />
                          {nameErrs.map((e, k) => (
                            <div className="field-error" key={k}>{e.message}</div>
                          ))}
                        </td>
                        <td>
                          <input
                            className={dimErrs.length ? "input bad num" : "input num"}
                            value={ix.dimension}
                            onChange={(e) => updateIndex(ti, ii, { dimension: e.target.value })}
                          />
                          {dimErrs.map((e, k) => (
                            <div className="field-error" key={k}>{e.message}</div>
                          ))}
                        </td>
                        <td>
                          <button type="button" className="mini" onClick={() => removeIndex(ti, ii)}>
                            ×
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              <button type="button" className="mini" onClick={() => addIndex(ti)}>
                + 索引
              </button>
            </div>
          ))}
          <button type="button" className="add-tensor" onClick={addTensor}>
            + 添加张量（2–11 个）
          </button>
        </div>
      ) : (
        <div className="json-editor">
          <textarea
            className={jsonParseError ? "json-area bad" : "json-area"}
            value={jsonText}
            spellCheck={false}
            rows={18}
            onChange={(e) => syncFromJson(e.target.value)}
          />
          {jsonParseError && (
            <div className="field-error">JSON 语法错误（提交前请修正）：{jsonParseError}</div>
          )}
          {errors.length > 0 && (
            <div className="error-box">
              <div className="error-title">后端返回的错误（输入已保留）：</div>
              {errors.map((e, i) => (
                <div key={i} className="error-line">
                  <span className="error-path">{pathLabel(e.path)}</span>
                  {e.message}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      <div className="submit-row">
        <button type="button" className="primary" onClick={submit} disabled={busy || !!jsonParseError}>
          {busy ? "求解中…" : "提交收缩计划"}
        </button>
        <span className="hint">
          规则：2–11 个张量；每张量 1–6 个互异 ASCII 索引；维数 2–1000；索引全网出现 1 或 2 次；网络连通。
        </span>
      </div>
    </section>
  );
}
