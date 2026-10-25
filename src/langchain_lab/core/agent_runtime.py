"""AgentRuntime：后端运行时宿主（多会话实例池 + 生命周期 + 统一操作面 + 全局配置注入）

三层职责边界（接入层 / AgentRuntime / ChatSession）：
- AgentRuntime 是【唯一的真正运行时】：持有实例池 dict[session_id->ChatSession]、channel、store、llm/tools、全局配置
- 职责：生命周期(create/resume/close/delete)；统一操作面 send_message/handle_command/request_compress；
        依赖组装 + 全局配置注入（ChatSession 与接入层都不读配置，只从 runtime 拿注入值）；多会话隔离并发
- 不持有：渲染逻辑、终端/HTTP 适配（exp9 终端 与 未来 FastAPI 都只调本类）
- 持久化：ContextStore 为外部订阅者（发归档/作废/压缩事件落库，ChatSession 不持有 store），runtime 只组装不内联
- 指令面与消息面分开：消息走 send_message(USER_MESSAGE_SUBMITTED)；指令走 handle_command(SESSION_COMMAND)；
  压缩触发走 request_compress(COMPRESS_REQUESTED)。接入层解析/校验，runtime 只响应语义化事件。
- 系统提示词/运行上限/压缩配置：全部由 AgentRuntime 从全局配置加载并注入，前端不支持自定义
- 事件通道：持有 EventChannel（默认 EventBus 内存实现）；close 走 channel.close()（flush+清空订阅），停机语义统一
"""
import asyncio
import uuid

from langchain_lab.core.agent_session import ChatSession
from langchain_lab.core.config.config_core import get_config
from langchain_lab.core.context.compressor import SessionCompressor
from langchain_lab.core.context.context_store import ContextStore
from langchain_lab.core.event_bus import EventBus, EventChannel
from langchain_lab.core.events import Event, EventType
from langchain_lab.core.llm.llm import build_llm, get_llm
from langchain_lab.core.tools.tool_disclosure import get_init_tools

_DEFAULT_SESSION_NAME = "新会话"


class AgentRuntime:
    """后端运行时宿主：实例池 + 生命周期 + 统一操作面 + 全局配置注入"""

    def __init__(self, *, session_ctx: dict | None = None, db_path: str | None = None) -> None:
        self._cfg = get_config()
        self._session_ctx = session_ctx or dict(self._cfg.project.session_ctx)  # 身份字典（含 user/tenant）
        self._agent_cfg = self._cfg.runtime.agent
        self._comp_cfg = self._cfg.runtime.compression
        self._keep_recent_turns = self._comp_cfg.keep_recent_turns
        self._db_path = str(db_path or self._agent_cfg.session_db_path)
        self._bus: EventChannel = EventBus()
        self._store = ContextStore(self._db_path)
        self._store.subscribe_to_bus(self._bus)  # 持久化订阅者（归档/压缩/作废落库）
        self._llm = None
        self._tools: list = []
        self._sessions: dict[str, ChatSession] = {}  # 实例池（会话类是类定义，实例由运行时统一持有）
        self._compressor_cache: dict[str, SessionCompressor | None] = {}

    # —— 依赖初始化（llm/tools 需异步构建）——
    async def ainit(self) -> None:
        self._llm = await asyncio.to_thread(get_llm)
        self._tools = await get_init_tools(self._session_ctx)

    # —— 总线/存储暴露（供接入层订阅事件流 / 选会话读存储）——
    @property
    def bus(self) -> EventChannel:
        return self._bus

    @property
    def store(self) -> ContextStore:
        return self._store

    # —— 会话查询 ——
    async def list_sessions(self) -> list[dict]:
        """按身份(user/tenant)列出非删除会话（供接入层选择 UI）"""
        return await self._store.list_sessions(
            self._session_ctx.get("user_id", ""), self._session_ctx.get("tenant_id", "")
        )

    def get(self, session_id: str) -> ChatSession | None:
        return self._sessions.get(session_id)

    # —— 生命周期：create / resume ——
    async def create_session(self, name: str | None = None) -> ChatSession:
        """新建会话：system_prompt 由全局配置注入（不支持前端自定义），name 默认值"""
        sid = str(uuid.uuid4())
        system_prompt = self._agent_cfg.default_system_prompt
        await self._store.ensure_session(sid, self._session_ctx, system_prompt, name or _DEFAULT_SESSION_NAME)
        chat = self._build(sid, branch_id="root", history=None, summary=None)
        self._sessions[sid] = chat
        return chat

    async def resume_session(self, session_id: str) -> ChatSession | None:
        """恢复会话：store 回溯构建 llm 视图注入（KV 一致），无则返回 None"""
        history = await self._store.load_view(session_id, "root", self._keep_recent_turns)
        if history is None:
            return None
        summary = await self._store.load_summary(session_id)
        chat = self._build(session_id, branch_id="root", history=history, summary=summary)
        self._sessions[session_id] = chat
        return chat

    # —— 统一操作面（把接入层的"输入/指令"翻译成事件发布，runtime 不做业务解析）——
    def send_message(self, session_id: str, content: str) -> None:
        """消息面：普通对话内容 → USER_MESSAGE_SUBMITTED"""
        self._require(session_id)
        self._bus.publish(Event(EventType.USER_MESSAGE_SUBMITTED, session_id, {"content": content}))

    def handle_command(self, session_id: str, action: str, payload: dict | None = None) -> None:
        """指令面：action 由接入层解析/校验（branch_*/delete/rename/regenerate），runtime 只按 action 响应"""
        self._require(session_id)
        self._bus.publish(Event(EventType.SESSION_COMMAND, session_id, {"action": action, **(payload or {})}))

    def request_compress(self, session_id: str, *, force: bool = False, keep: int | None = None) -> None:
        """压缩触发：force=False 自动(每轮后置)，force=True 前端手动提前"""
        self._require(session_id)
        self._bus.publish(Event(EventType.COMPRESS_REQUESTED, session_id, {"force": force, "keep": keep}))

    # —— 关闭（停机语义：关全部会话 + channel.close + 关 MCP 长连接）——
    async def close(self) -> None:
        for chat in self._sessions.values():
            await chat.close()
        await self._bus.close()  # flush 后台任务 + 清空订阅（停机语义，统一走 channel）
        from langchain_lab.mcp.client.mcp_session_manager import _session_mgr
        await _session_mgr.clear_all()

    # —— 内部 ——
    def _require(self, session_id: str) -> None:
        if session_id not in self._sessions:
            raise KeyError(f"会话 {session_id} 不存在（未 create/resume）")

    def _build(self, session_id: str, branch_id: str, history, summary) -> ChatSession:
        """组装 ChatSession：全部依赖 + 配置值注入；branch 回调闭包绑定本会话/存储"""
        compressor = self._compressor_cache.get(session_id)
        if session_id not in self._compressor_cache:
            compressor = self._build_compressor(session_id)
            self._compressor_cache[session_id] = compressor
        return ChatSession(
            session_id=session_id, session_ctx=self._session_ctx, bus=self._bus, llm=self._llm, tools=self._tools,
            compressor=compressor, branch_id=branch_id, system_prompt=self._agent_cfg.default_system_prompt,
            max_llm_loop=self._agent_cfg.max_llm_loop, max_tool_calls=self._agent_cfg.max_tool_calls,
            history=history, summary=summary,
            branch_loader=lambda bid: self._store.load_view(session_id, bid, self._keep_recent_turns),
            branch_lister=lambda: self._store.list_branches(session_id),
            branch_creator=lambda parent, point: self._store.create_branch(session_id, parent, point),
            branch_deleter=lambda bid: self._store.delete_branch(session_id, bid),
            session_renamer=lambda n: self._store.rename_session(session_id, n),
            branch_renamer=lambda bid, n: self._store.rename_branch(session_id, bid, n),
        )

    def _build_compressor(self, session_id: str) -> SessionCompressor | None:
        if not self._comp_cfg.enabled:
            return None
        inst_name = self._comp_cfg.summarize_model or self._cfg.llm.default_instance
        if not inst_name or inst_name not in self._cfg.llm.instances:
            raise ValueError(f"压缩摘要模型实例 '{inst_name}' 未在 config/llm.yaml 中配置")
        summarize_llm = build_llm(self._cfg.llm.instances[inst_name])
        return SessionCompressor(
            summarize_llm=summarize_llm, keep_recent_turns=self._keep_recent_turns,
            summarize_prompt=self._comp_cfg.summarize_prompt, bus=self._bus, session_id=session_id,
        )
