import type { IntLike } from "./types";

export function formatBigInt(v: IntLike | string): string {
  const n = typeof v === "bigint" ? v : BigInt(v);
  return n.toLocaleString("en-US");
}

/** Approximate scientific form for orientation, exact value shown alongside. */
export function magnitudeHint(v: IntLike): string {
  const n = typeof v === "bigint" ? v : BigInt(v);
  const neg = n < 0n;
  const s = (neg ? -n : n).toString();
  if (s.length <= 6) return "";
  const head = s.slice(0, 3);
  const mantissa = `${head[0]}.${head.slice(1)}`;
  return `≈ ${neg ? "-" : ""}${mantissa}×10^${s.length - 1}`;
}
