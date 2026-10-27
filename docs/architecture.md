# langchain_lab 运行时架构契约

> 本文是**开发边界契约**，不是实现快照。任何改动先对齐本文；只有当"新增的东西放不进任何 plane/域"时才讨论改本文。
> 配套：目录结构见 `gen_tree.py` → `docs/project_tree.md`。

---

## 1. 定位

面向 **多会话 Agent 运行时** 的分层架构：事件驱动的会话执行 + 三平面分离的接入面 + 配置覆盖链。
目标：边界按**领域职责**切（不按当前实现切），保证后续**只换组件、不动骨架**。

## 2. 四层架构（依赖单向）

```
接入面（对前端）         数据面 / 控制面 / 配置面
   ↓
编排层（AgentRuntime）    生命周期 + 操作面 + 配置解析注入 + 策略门控
   ↓
领域层（ChatSession + 协作者）  状态机 + 事件源 + 一轮编排；指令/压缩/分支/配置委托协作者
   ↓
基础设施层                EventBus · ContextStore(SQLite) · LLM Provider · 全局配置中心
```

**不变量：依赖只向下。** 接入层不读配置；会话不读配置、不碰持久化实现；编排层是唯一"读配置 + 组装依赖"的地方。

## 3. 三平面分离（接入面）

| 平面 | 语义 | 协议 | 生命周期 |
|---|---|---|---|
| **数据面** | 展示/实时状态 | 流式事件 / SSE（现有 EventBus） | 单向、高频 |
| **控制面** | 一次性运行时动作（发消息/压缩/切分支/重跑/重命名/删） | 指令注册表 + 事件命令（`POST /control/{action}`） | 命令，不持久化意图 |
| **配置面** | 用户/租户设置（模型、上下文策略、工具启停） | RESTful CRUD（`GET/PUT /config/{user|tenant}`） | 状态、持久化、幂等 |

- 控制面与配置面**必须区分**：动作 ≠ 设置。指令走 `core/commands/instruction_registry.py`（通信格式约定，全局单例）。
- 数据面由 EventBus 承载；控制面用指令注册表；配置面用 RESTful。

## 4. 配置覆盖链（ConfigResolver）

```
全局配置 (yaml，部署级默认)  →  租户配置 (覆盖)  →  用户配置 (覆盖)  →  EffectiveConfig → 注入会话
```

- 全局：`config/*.yaml`（llm/runtime/context/rag/mcp/project），`ConfigLoader` 启动加载。
- 租户/用户：配置面持久化（按 `session_ctx` 的 user/tenant 维度），运行时可改。
- `ConfigResolver` 把三层合并出 `EffectiveConfig`，**AgentRuntime 在 create/resume 会话时注入**；会话不自行读全局。
- 配置域划分（模型能力 / 上下文策略 / 运行时行为 互解耦）：
  - `llm.yaml`：模型能力（context_window、reasoning_effort…）——跟模型走
  - `context.yaml`：上下文策略（keep_recent_turns、headroom、compression）——跟产品走
  - `runtime.yaml`：Agent 行为（max_loop/max_tool_calls/db_path/system_prompt）——跟部署走

## 5. 会话聚合根 + 协作者

`ChatSession` 收敛为：**状态机 + 事件源 + 一轮编排**（输入→LLM→工具→输出）。协作逻辑委托协作者：

- `CommandRouter`：指令分发 + 管理类指令实现（branch/rename/update-compressor）
- `Compressor`：上下文压缩（保留策略/预算/摘要）
- `BranchService`：分支视图构建（当前走 store 回调）
- `Store`：持久化（单一真相源，事件订阅落库）
- `ConfigResolver`：有效配置解析

会话只持有 `_messages`（单一视图）与 `_summary`，对外发领域事件；依赖全部构造注入，无全局单例。

## 6. 事件总线分工（HITL 演进关键）

现有 `EventBus`（`core/bus/`）只承载**单向领域事件**：展示、状态、审计、持久化订阅、命令广播。
**不被污染成 RPC 总线。**

- 需要 **请求/响应 + 阻塞等待** 的能力（未来 human-in-loop / 权限确认）：用 **asyncio Future + correlation_id 的确认原语**（编排层本地），不经总线。
- 独立消息队列：只有当演进到**多实例/跨进程**时才升级（带 reply 语义的 MQ），那是升级不是重构。

## 7. 预留点位（先定边界，不实现）

以下能力**边界已定、暂不实现**，后续在各自域内补齐（不重构骨架）：

- **policy 域**（`core/policy/`）：工具白名单/黑名单、敏感操作需确认、沙箱边界、用户权限等级。收编点。
- **工具管理**：tools 层注册表 + 启停 + 可见性（`tool_disclosure`/`tool_middleware` 是雏形）。
- **human-in-loop / 权限确认**：编排层加**策略门控**（工具执行前过 policy 引擎），命中则 `await` 确认原语（approval_id + Future）；控制面补 `approve/reject/modify` 指令；确认状态/结果走 EventBus 通知+审计。

## 8. LLM 思考模型兼容

- `llm.yaml` 实例级 `reasoning_effort`（思考深度，如 `low/medium/high`，DeepSeek/OpenAI o 系支持）。
- `build_llm` 仅在显式配置时透传给 provider（不支持的 provider 不传，避免报错）。
- 会话/运行时透传不丢：配置走覆盖链，会话从注入的 effective config 取模型参数。

## 9. 演进自检法（防"又重构"）

每次新增先问：**落在哪个 plane / 哪个域？**
- 落在已有边界内 → 就是扩展（加字段/加组件/加 handler），不动骨架。
- 放不进任何 plane/域 → 才说明边界有问题，此时才讨论改本文。
- 触发真正重构的（罕见）：领域跃迁（单 Agent→多 Agent 编排、单机→分布式、跨会话时间旅行）。

## 10. 本次落地范围（dev-ing）

1. `docs/architecture.md`（本文）
2. 配置面：租户/用户配置存储 + `ConfigResolver` 覆盖链 + 控制面 config 端点
3. 会话解耦：抽 `CommandRouter`，会话收敛聚合根
4. `AgentRuntime` 升级：effective config 注入 + 思考模型透传
5. LLM：`reasoning_effort` 思考深度配置
6. policy 域 + 确认原语**接口骨架**（仅占位，不实现规则）
7. exp9 临时前端适配，测试 + ruff + pytest 全绿
