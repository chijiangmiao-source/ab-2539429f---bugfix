"""FastAPI application: tensor-network contraction planner."""

from __future__ import annotations

import json
import os

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .planner import ValidationError, plan

app = FastAPI(
    title="张量网络收缩规划器",
    description="提交连通张量网络，返回峰值内存/总乘法数最优的收缩树与根切分分类。",
    version="1.0.0",
)

# The web app is served by its own container; allow cross-origin calls in
# deployments where the two ports differ.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/healthz")
@app.get("/api/healthz")
def healthz() -> dict:
    # /healthz is used by the container/compose health check; /api/healthz
    # is the same probe reachable through the web tier's /api reverse proxy.
    return {"status": "ok", "service": "api"}


@app.get("/api/info")
def api_info() -> dict:
    return {
        "constraints": {
            "tensors_min": 2,
            "tensors_max": 11,
            "indices_per_tensor_min": 1,
            "indices_per_tensor_max": 6,
            "dimension_min": 2,
            "dimension_max": 1000,
            "index_occurrences": [1, 2],
            "network": "connected",
        }
    }


@app.post("/api/plan")
async def create_plan(request: Request) -> JSONResponse:
    raw = await request.body()
    try:
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else None
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            detail = exc.msg if isinstance(exc, json.JSONDecodeError) else str(exc)
            return JSONResponse(
                {
                    "errors": [
                        {
                            "code": "invalid_json",
                            "message": f"请求体不是合法 JSON：{detail}",
                            "path": [],
                        }
                    ]
                },
                status_code=400,
            )
        # the exact DP is CPU-bound (up to ~1s for n=11); run it in the
        # starlette worker threadpool so the event loop stays responsive
        result = await run_in_threadpool(plan, payload)
        return JSONResponse(result)
    except ValidationError as exc:
        return JSONResponse({"errors": exc.errors}, status_code=422)


def main() -> None:
    import uvicorn

    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", "8000"))
    uvicorn.run("app.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
