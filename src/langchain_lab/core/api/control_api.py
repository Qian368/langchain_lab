"""控制面 HTTP API：统一命令入口 + 指令发现（FastAPI）

- POST /commands：{session_id, action, payload} —— 统一命令入口（JSON-RPC/CQRS 风格）
  - payload 按注册表 schema 用 Pydantic 校验（校验是 API 层职责，注册表只提供 schema）
  - 身份锚点：session_id 路由到运行时会话实例，user/tenant 由会话创建时绑定，请求不重复传
  - 统一结果信封 {ok, data|error}，前端一套处理逻辑
- GET /instructions：注册表公开契约（前端帮助 / 载荷表单），发现端点解决可预测性

命令发布是异步的（runtime.handle_command → SESSION_COMMAND 事件 → ChatSession 分发），
响应只表示"已受理"，结果经事件通道（SESSION_NOTICE / SESSION_STATE_UPDATED）回传。
"""
from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, ValidationError

from langchain_lab.core.api.payload_validation import validate_payload
from langchain_lab.core.commands import build_public_specs, get_spec

app = FastAPI(title="LangChain Lab Control Plane", version="0.1.0")


# 运行时单例：由启动方注入（AgentRuntime 已在创建会话时绑定身份上下文）
_runtime = None


def set_runtime(runtime) -> None:
    """注入 AgentRuntime 实例（生产在应用启动时调用，绑定身份/实例池）"""
    global _runtime
    _runtime = runtime


def _get_runtime():
    if _runtime is None:
        raise RuntimeError("控制面运行时未初始化：请先 set_runtime()")
    return _runtime


# ---------- 请求/响应模型 ----------
class CommandRequest(BaseModel):
    session_id: str = Field(..., description="会话 id（身份锚点，路由到运行时实例）")
    action: str = Field(..., description="指令名，必须存在于指令注册表")
    payload: dict | None = Field(default=None, description="指令载荷，按注册表 schema 校验")


class CommandResponse(BaseModel):
    ok: bool
    data: dict | None = None
    error: str | None = None


class ConfigRequest(BaseModel):
    """配置面覆盖载荷（按域路径，如 {"context": {"keep_recent_turns": 8}} 或 {"llm": {"default_instance": "zhipuai"}}）"""
    overrides: dict = Field(default_factory=dict, description="用户/租户配置覆盖，经 ConfigResolver 合并出 EffectiveConfig")


# ---------- 端点 ----------
@app.get("/config")
def get_config_endpoint() -> dict:
    """配置面：读取当前运行时（身份维度）的用户/租户配置覆盖"""
    return {"config": _get_runtime().get_user_config()}


@app.put("/config", response_model=CommandResponse)
async def put_config_endpoint(req: ConfigRequest) -> CommandResponse:
    """配置面：写入用户/租户配置覆盖（先校验有效性，新会话生效）"""
    try:
        _get_runtime().set_user_config(req.overrides)
    except (ValueError, ValidationError) as exc:
        return CommandResponse(ok=False, error=f"配置校验失败: {exc}")
    except RuntimeError as exc:
        return CommandResponse(ok=False, error=str(exc))
    return CommandResponse(ok=True, data={"applied": True, "overrides": req.overrides})


@app.get("/instructions")
def instructions() -> dict:
    """指令发现：返回注册表公开契约（含别名/说明/载荷 schema，前端帮助与表单渲染）"""
    return {"instructions": build_public_specs()}


# ---------- 控制面控制台（前端骨架：从 GET /instructions 渲染帮助 + 载荷表单，POST /commands 发指令）----------
_CONSOLE_HTML = """
<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>LangChain Lab 控制面控制台</title>
<style>
  body{font-family:-apple-system,'Segoe UI','Microsoft YaHei',sans-serif;background:#f7f8fa;color:#1f2328;margin:0;padding:24px;line-height:1.5}
  h1{font-size:22px;margin:0 0 6px}
  p.sub{color:#57606a;font-size:13px;margin:0 0 16px}
  .bar{display:flex;gap:8px;align-items:center;margin-bottom:16px;flex-wrap:wrap}
  label{font-size:13px}
  input{padding:6px 8px;border:1px solid #d0d7de;border-radius:6px;font-size:13px}
  button{padding:6px 12px;border:1px solid #1f6feb;background:#1f6feb;color:#fff;border-radius:6px;cursor:pointer;font-size:13px}
  #cmds{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px}
  .card{background:#fff;border:1px solid #d0d7de;border-radius:10px;padding:12px}
  .alias code{background:#eff1f3;border:1px solid #d0d7de;border-radius:4px;padding:1px 5px;font-size:12px}
  .desc{font-size:13px;color:#57606a;margin:8px 0}
  .fields{margin:8px 0}
  .field{display:flex;flex-direction:column;gap:2px;margin-bottom:6px}
  .field small{color:#8b949e;font-size:11px}
  .none{color:#8b949e;font-size:12px;margin:8px 0}
  pre{background:#161b22;color:#e6edf3;border-radius:8px;padding:12px;overflow:auto;font-size:12px}
</style>
</head>
<body>
<h1>控制面控制台</h1>
<p class="sub">指令集由后端注册表（GET /instructions）自动渲染，载荷表单按 payload schema 生成；
执行走统一命令入口 POST /commands。身份由 session_id 锚定，不重复传用户/租户。</p>
<div class="bar">
  <label>session_id <input id="sid" placeholder="会话 id"></label>
  <button onclick="load()">刷新指令</button>
</div>
<div id="cmds"></div>
<h2 style="font-size:16px;margin:16px 0 4px">响应（统一信封 {ok, data|error}）</h2>
<pre id="result">尚未执行指令</pre>
<script>
async function api(path, opts){
  const r = await fetch(path, opts);
  return r.json();
}
async function load(){
  const d = await api('/instructions');
  document.getElementById('cmds').innerHTML = d.instructions.map(s => {
    const fields = Object.entries(s.payload || {}).map(([f, m]) => `
      <div class="field"><label>${f}${m.required ? ' *' : ''}</label>
      <input data-f="${f}" type="${m.type === 'integer' ? 'number' : 'text'}"
        ${m.required ? 'required' : ''}${m.minimum != null ? ` min="${m.minimum}"` : ''}>
      <small>${m.description || ''}</small></div>`).join('');
    return `<div class="card">
      <div class="alias">${s.aliases.map(a => `<code>${a}</code>`).join(' ')}</div>
      <div class="desc">${s.description}</div>
      ${fields ? `<div class="fields">${fields}</div>` : '<div class="none">无载荷</div>'}
      <button onclick="exec('${s.name}', this)">执行 ${s.name}</button>
    </div>`;
  }).join('');
}
async function exec(name, btn){
  const card = btn.closest('.card');
  const payload = {};
  card.querySelectorAll('input[data-f]').forEach(i => {
    const v = i.value.trim();
    if (v !== '') payload[i.dataset.f] = i.type === 'number' ? Number(v) : v;
  });
  const sid = document.getElementById('sid').value.trim();
  const d = await api('/commands', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({session_id: sid, action: name, payload})
  });
  document.getElementById('result').textContent = JSON.stringify(d, null, 2);
}
load();
</script>
</body>
</html>
"""


@app.get("/console", response_class=HTMLResponse)
def console() -> str:
    """控制面控制台前端骨架：从注册表自动渲染指令帮助与载荷表单"""
    return _CONSOLE_HTML


@app.post("/commands", response_model=CommandResponse)
async def commands(req: CommandRequest) -> CommandResponse:
    """统一命令入口：按注册表校验 action + payload，分发给运行时对应会话"""
    spec = get_spec(req.action)
    if spec is None:
        return CommandResponse(ok=False, error=f"未知指令 action={req.action}（可用指令见 /instructions）")
    try:
        clean_payload = validate_payload(req.action, req.payload or {})
    except ValidationError as exc:
        return CommandResponse(ok=False, error=f"载荷校验失败: {exc}")
    try:
        _get_runtime().handle_command(req.session_id, req.action, clean_payload)
    except KeyError as exc:
        return CommandResponse(ok=False, error=str(exc))
    except RuntimeError as exc:
        return CommandResponse(ok=False, error=str(exc))
    return CommandResponse(ok=True, data={"accepted": req.action, "session_id": req.session_id})


@app.get("/health")
def health() -> dict:
    """健康检查：控制面可用 + 运行时是否已注入"""
    try:
        _get_runtime()
        runtime_ok = True
    except RuntimeError:
        runtime_ok = False
    return {"status": "ok", "runtime_initialized": runtime_ok}


# ---------- 启动 ----------
def serve():
    uvicorn.run("langchain_lab.core.api.control_api:app", reload=True)
