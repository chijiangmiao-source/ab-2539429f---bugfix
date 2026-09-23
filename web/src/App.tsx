import { useEffect, useState } from "react";
import { ApiClient, apiBaseUrl, PlanRequestError } from "./api";
import type { ApiError, NetworkPayload, PlanResult } from "./types";
import { NetworkEditor } from "./components/NetworkEditor";
import { ResultView } from "./components/ResultView";

const client = new ApiClient(apiBaseUrl);

export function App() {
  const [result, setResult] = useState<PlanResult | null>(null);
  const [errors, setErrors] = useState<ApiError[]>([]);
  const [busy, setBusy] = useState(false);
  const [apiUp, setApiUp] = useState<boolean | null>(null);

  useEffect(() => {
    let alive = true;
    const ping = async () => {
      const ok = await client.health();
      if (alive) setApiUp(ok);
    };
    ping();
    const id = setInterval(ping, 10000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  async function handleSubmit(payload: NetworkPayload) {
    setBusy(true);
    setErrors([]);
    try {
      const r = await client.plan(payload);
      setResult(r);
    } catch (e) {
      if (e instanceof PlanRequestError) {
        setErrors(e.errors);
        setResult(null);
      } else {
        setErrors([
          {
            code: "network",
            message: `无法连接 API：${(e as Error).message}。请检查 API 服务是否启动。`,
            path: [],
          },
        ]);
      }
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="page">
      <header className="topbar">
        <div>
          <h1>张量网络收缩规划器</h1>
          <p className="subtitle">
            精确整数 DP：先最小化峰值内存，再最小化总乘法数 · 规范括号串打破同优
          </p>
        </div>
        <div className={`health ${apiUp === null ? "unknown" : apiUp ? "up" : "down"}`}>
          <span className="dot" />
          API {apiUp === null ? "检测中" : apiUp ? "在线" : "不可达"}
        </div>
      </header>

      <NetworkEditor errors={errors} onSubmit={handleSubmit} busy={busy} />

      {errors.length > 0 && (
        <section className="panel error-panel">
          <h2>非法输入 — 已保留你的全部输入并定位原因</h2>
          <ul className="error-list">
            {errors.map((e, i) => (
              <li key={i}>
                <code className="error-code">{e.code}</code>
                <span>{e.message}</span>
                {e.path.length > 0 && (
                  <span className="error-path-raw">路径：{JSON.stringify(e.path)}</span>
                )}
              </li>
            ))}
          </ul>
        </section>
      )}

      {result && <ResultView result={result} />}

      <footer className="footer">
        React + Vite / FastAPI · 成本按边界索引并集的维数乘积精确计算
      </footer>
    </div>
  );
}
