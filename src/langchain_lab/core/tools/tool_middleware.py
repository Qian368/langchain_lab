import json
import traceback

import httpx
from langchain.tools import BaseTool
from langchain_core.messages import ToolCall
from mcp import MCPError
from pydantic import ValidationError

# ========== 项目的 logger 替换这里 ==========
from langchain_lab.util.logger import LogName, setup_logger

logger = setup_logger(LogName.CORE)

# ========== 1. 工具运行时配置（只存@tool承载不了的业务/运维规则） ==========
TOOL_RUNTIME_CONFIG = {
    "query_order": {
        # 业务格式规则（可选，按需添加）
        # "param_rules": {"order_no": {"regex": r"^ORD_\d+$", "max_len": 32}},
        # "require_auth": True,  # 是否需要权限校验（预留）
        # "timeout": 3,  # 超时时间（可选）
        "audit": True,  # 是否记录审计日志
    }
}


# ========== 2. 自定义业务异常 ==========
class ToolExecutionError(Exception):
    """工具业务异常，可直接返回给LLM修正"""

    pass


# ========== 3. 核心：生产级安全工具执行器（所有中间件封装） ==========
async def execute_safe_tool_call(tool_call: ToolCall, tools: list[BaseTool], session_ctx: dict, tool_index: int = 0) -> dict:
    """
    统一工具执行入口，内置完整中间件链路
    :param tool_call: 单个LLM返回的tool_call字典（name/id/args）
    :param tools: 全局工具列表
    :param session_ctx: 会话上下文（user_id/tenant_id/trace_id），不来自LLM
    :param tool_index: 工具序号，用于控制台打印
    :return: 标准化返回，直接用于拼装ToolMessage
    """
    tool_name = tool_call["name"]
    tool_call_id = tool_call["id"]
    raw_args = tool_call.get("args", {})
    trace_id = session_ctx.get("trace_id", "-")

    # ===== 中间件1：工具存在性校验 + 控制台打印 =====
    target_tool = next((t for t in tools if t.name == tool_name), None)

    if target_tool is None:
        err_msg = f"未发现工具：{tool_name}"
        logger.warning(f"[trace:{trace_id}] 📢 工具 {[tool_index]} 不存在: {tool_name}")
        return {"tool_call_id": tool_call_id, "name": tool_name, "content": f"{err_msg}，可用工具为：{[t.name for t in tools]}"}

    logger.info(
        f"[trace:{trace_id}] 工具 {[tool_index]} 开始执行: {tool_name}, "
        f"\n       ├─ 动态按需注入参数：{json.dumps(session_ctx, ensure_ascii=False)}"
        f"\n       └─ llm调用原参数：{json.dumps(raw_args, ensure_ascii=False)}"
    )

    try:
        # ===== 中间件2：业务规则校验（可选，按需开启） =====
        runtime_cfg = TOOL_RUNTIME_CONFIG.get(tool_name, {})
        param_rules = runtime_cfg.get("param_rules", {})
        if param_rules and isinstance(param_rules, dict):
            import re

            for param, rules in param_rules.items():
                if param not in raw_args:
                    continue
                val = raw_args[param]
                if "max_len" in rules and isinstance(val, str) and len(val) > rules["max_len"]:
                    raise ToolExecutionError(f"参数{param}长度超过限制：最大{rules['max_len']}")
                if "regex" in rules and isinstance(val, str) and not re.match(rules["regex"], val):
                    raise ToolExecutionError(f"参数{param}格式错误，需符合规则：{rules['regex']}")

        # ===== 中间件3：上下文注入 + 执行工具 =====
        # 推荐：对应 InjectedToolArg：隐式工具参数拆分进工具入参，工具内部无需直接使用
        if getattr(target_tool, "_tool_type", None) == "mcp":
            # MCP 工具：session_ctx 内部注入，不混入业务参数
            tool_result = await target_tool.ainvoke(
                raw_args,
                session_ctx=session_ctx,
            )
        else:
            # 本地工具：参数合并注入
            merged_args = {**raw_args, **session_ctx}
            tool_result = await target_tool.ainvoke(merged_args)

        # 对应 RunnableConfig 隐式工具参数：通过 configurable 注入，工具内部需手动拆分
        # tool_result = await target_tool.ainvoke(raw_args, config={"configurable": session_ctx})

        # ===== 中间件4：后置处理 + 日志 =====
        if runtime_cfg.get("audit", True):
            logger.info(f"[trace:{trace_id}] 工具 {[tool_index]} 成功执行: {tool_name}")
            print(f"       └─ 执行结果：{tool_result}")
        return {"tool_call_id": tool_call_id, "name": tool_name, "content": tool_result}

    # ===== 中间件5：异常分层捕获与包装 =====
    except ValidationError as e:
        # @tool 原生参数校验错误：包装成自然语言返回给LLM修正
        errors = [err["msg"] for err in e.errors()]
        err_msg = f"参数错误：{'; '.join(errors)}，请修正后重试"
        logger.warning(f"[trace:{trace_id}] 工具 {[tool_index]} 参数校验失败: {tool_name}, {err_msg}")
        return {"tool_call_id": tool_call_id, "name": tool_name, "content": err_msg}
    except ToolExecutionError as e:
        # 业务规则错误：直接返回提示
        err_msg = str(e)
        logger.warning(f"[trace:{trace_id}] 工具 {[tool_index]} 业务校验失败: {tool_name}, {err_msg}")
        return {"tool_call_id": tool_call_id, "name": tool_name, "content": f"工具调用失败：{err_msg}"}
    except MCPError as e:
        # MCP 协议层错误：服务端返回错误、SSE流异常、响应超时等
        err_msg = f"MCP 工具调用失败：{e.message}"
        logger.warning(f"[trace:{trace_id}] 工具 {[tool_index]} MCP协议错误: {tool_name}, {err_msg}")
        return {"tool_call_id": tool_call_id, "name": tool_name, "content": err_msg}
    except httpx.ConnectError as e:
        # 传输层错误：MCP服务不可达、连接断开
        err_msg = "MCP 服务连接失败，无法调用工具"
        logger.warning(f"[trace:{trace_id}] 工具 {[tool_index]} 连接失败: {tool_name}, {e!s}")
        return {"tool_call_id": tool_call_id, "name": tool_name, "content": err_msg}
    except Exception as e:
        # 未知异常：不暴露内部细节，只返回通用错误
        logger.error(f"[trace:{trace_id}] 工具 {[tool_index]} 执行异常: {tool_name}, 错误:{e!s}\n{traceback.format_exc()}")
        return {"tool_call_id": tool_call_id, "name": tool_name, "content": "工具执行发生内部错误，请稍后重试"}
