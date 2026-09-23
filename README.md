# 张量网络收缩规划器（Tensor-Network Contraction Planner）

输入一个连通张量网络（2–11 个张量），系统用**精确整数动态规划**求出收缩树：

1. **首要目标：最小化峰值内存**——整个收缩过程中必须驻留的最大张量大小
   （所有输入张量与所有中间结果都计入）；
2. **次要目标：最小化总乘法数**；
3. 同优树按**叶标识（张量名）与递归排序子树得到的括号串**选出唯一规范解；
4. 对每个**非平凡根切分**（叶集二部划分）给出 **必现 / 可选 / 不出现** 分类与证据。

- 后端：FastAPI（`backend/`），Python 任意精度整数，维数 1000 的最坏情形结果
  可达 1000²²，不溢出。
- 前端：React + Vite + TypeScript（`web/`），树图（SVG）、指标表、分类证据、
  错误定位。
- 部署：Web 与 API 各自独立 Dockerfile，Docker Compose 联动，两服务均带健康检查；
  一次性 `verify` 服务运行测试、构建检查与真实 API 冒烟，自行退出并用退出码报告结果。

## 成本模型

对收缩的两组张量 `A`、`B`：

- `boundary(A)` = 在 `A` 内部恰好出现一次的索引（悬空或连向组外）；
- **乘法数** = `∏ dim(i)`，其中 `i ∈ boundary(A) ∪ boundary(B)`；
- **结果大小** = 新边界 `boundary(A) △ boundary(B)`（对称差）上维数之积；
- **峰值** = 所有输入大小与所有中间结果大小的最大值。

算法对每个叶子子集维护 `(峰值, 总乘法)` 的 Pareto 前沿（区间 DP，
Bell 数级别的切分枚举，n ≤ 11 时亚秒级完成），并在全局最优点上精确判定哪些根切分
可以出现在某棵同优树中：

- **必现**：可达全局最优的根切分只有一个——所有同优树都必现它；
- **可选**：该切分可达最优，但存在别的根切分同样可达最优；
- **不出现**：强制该切分时，最小峰值或给定峰值下的总乘法数严格大于最优值
  （证据中给出具体数值）。

## 输入约束（非法输入会被保留并精确定位）

| 项目 | 约束 |
| --- | --- |
| 张量数量 | 2–11，名称唯一 |
| 每张量索引数 | 1–6，同张量内互异、ASCII 名称 |
| 维数 | 整数 2–1000，同一索引各处维数一致 |
| 索引出现次数 | 全网恰好 1 或 2 次 |
| 网络 | 必须连通 |

校验错误返回 `HTTP 422`，形如：

```json
{
  "errors": [
    {
      "code": "index_occurrence",
      "message": "索引 'i' 全网出现 3 次，必须恰好出现 1 或 2 次",
      "path": ["tensors", 0, "indices", 0, "name"]
    }
  ]
}
```

前端按 `path` 高亮对应字段；提交失败不会清空或修改用户输入。

## 请求示例

```bash
curl -X POST http://localhost:8000/api/plan \
  -H 'content-type: application/json' \
  -d '{
    "tensors": [
      {"name": "A", "indices": [{"name": "i", "dimension": 2}, {"name": "j", "dimension": 2}]},
      {"name": "B", "indices": [{"name": "j", "dimension": 2}, {"name": "k", "dimension": 2}]},
      {"name": "C", "indices": [{"name": "k", "dimension": 2}, {"name": "l", "dimension": 2}]},
      {"name": "D", "indices": [{"name": "l", "dimension": 2}, {"name": "i", "dimension": 2}]}
    ]
  }'
```

四元环（d=2）返回：峰值 4、总乘法 20；7 个根切分中 6 个可选、1 个不出现
——对角切分 `(A,C)|(B,D)` 会先制造 16 元素的巨型中间量，证据中给出该数值。

## 快速开始（Docker Compose）

```bash
cp .env.example .env        # 可选：修改宿主机端口
docker compose up --build
```

- Web UI：http://localhost:8080 （`WEB_HOST_PORT` 可改）
- API 与文档：http://localhost:8000 ，健康检查 `GET /healthz`（`API_HOST_PORT` 可改）

容器内监听地址同样可配置：API 服务读取 `API_HOST` / `API_PORT`；
Web 容器经 `API_UPSTREAM`（默认 `api:8000`）反向代理 `/api/*`。

### 一次性验证服务

```bash
docker compose run --rm verify
```

该服务顺序执行并以退出码报告结果：

1. 后端完整测试（`pytest`，含 40 组随机网络对暴力枚举参考实现的交叉验证）；
2. Web 生产构建（`tsc -b && vite build`）；
3. 对运行中的 API 做真实 HTTP 冒烟：
   - 四张量环的同优分类（0 必现 / 6 可选 / 1 不出现，对角切分证据为 16）；
   - 索引出现三次的错误边界（422 + 精确路径）。

全部通过退出码为 0，任一失败非零。

## 本地开发（不用 Docker）

```bash
# 后端
cd backend
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-test.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
pytest                      # 运行测试

# 前端
cd web
npm install
VITE_API_PROXY_TARGET=http://localhost:8000 npm run dev
```

## 目录结构

```
backend/
  app/
    planner.py        # 校验 + Pareto DP + 规范解 + 根切分分类
    main.py           # FastAPI：/api/plan、/api/info、/healthz
  tests/
    brute.py          # Catalan 全枚举参考实现
    test_planner.py   # 校验/成本/规范序/分类（对拍暴力）
    test_api.py       # 真实 ASGI HTTP 冒烟
web/
  src/
    api.ts            # BigInt 安全的 API 客户端
    components/
      NetworkEditor.tsx  # 表单/JSON 双模式编辑与错误定位
      TreeView.tsx       # SVG 收缩树图
      ResultView.tsx     # 结论/步骤/分类证据/指标
deploy/
  Dockerfile.verify       # verify 一次性服务镜像
  smoke.py                # 冒烟断言
  verify-entrypoint.sh
docker-compose.yml        # api + web + verify
```
