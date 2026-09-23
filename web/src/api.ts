import JSONbig from "json-bigint";
import type { ApiError, PlanResult } from "./types";

// Costs may reach 1000**22, far beyond Number.MAX_SAFE_INTEGER; parse all
// integers as bigint so the reported evidence stays exact.
const JSONBig = JSONbig({ storeAsString: false, useNativeBigInt: true });

// json-bigint turns *every* integer into a bigint, including small scalars
// (dimensions, occurrence counts, tensor counts) used in ordinary numeric
// comparisons. Demote integers that safely fit into a Number back to
// number; only genuinely huge cost values remain bigint.
function normalize(value: unknown): unknown {
  if (typeof value === "bigint") {
    return value <= BigInt(Number.MAX_SAFE_INTEGER) &&
      value >= -BigInt(Number.MAX_SAFE_INTEGER)
      ? Number(value)
      : value;
  }
  if (Array.isArray(value)) {
    return value.map(normalize);
  }
  if (value && typeof value === "object") {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
      out[k] = normalize(v);
    }
    return out;
  }
  return value;
}

export class ApiClient {
  constructor(private baseUrl: string) {}

  async health(): Promise<boolean> {
    try {
      // /api/healthz is proxied to the API by both the vite dev server and
      // the production nginx config (the web tier's own /healthz is a
      // separate container-level probe).
      const r = await fetch(`${this.baseUrl}/api/healthz`);
      return r.ok;
    } catch {
      return false;
    }
  }

  async plan(payload: unknown): Promise<PlanResult> {
    const r = await fetch(`${this.baseUrl}/api/plan`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
    });
    const text = await r.text();
    let parsed: unknown = null;
    if (text) {
      try {
        parsed = normalize(JSONBig.parse(text));
      } catch {
        parsed = null;
      }
    }
    if (!r.ok) {
      const errors: ApiError[] =
        parsed && typeof parsed === "object" && Array.isArray((parsed as any).errors)
          ? (parsed as any).errors
          : [{ code: "http_error", message: `请求失败：HTTP ${r.status}`, path: [] }];
      throw new PlanRequestError(errors);
    }
    return parsed as PlanResult;
  }
}

export class PlanRequestError extends Error {
  constructor(public errors: ApiError[]) {
    super(errors.map((e) => e.message).join("; "));
  }
}

// When served by nginx the API shares the page origin; in `vite dev` the
// dev server proxies /api to the backend. An explicit override is honored.
export const apiBaseUrl = (import.meta as any).env?.VITE_API_BASE_URL ?? "";
