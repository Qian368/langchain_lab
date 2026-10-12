"""实验2：多轮对话（single——chat）：工具调用 + 流式输出 + 最小循环 + 美化终端交互"""

import asyncio
import json
import logging
from typing import cast

from colorama import Fore, init
from langchain.chat_models import BaseChatModel
from langchain.tools import BaseTool
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables.base import Runnable
from prompt_toolkit import PromptSession

from langchain_lab.core.config.config_core import get_config
from langchain_lab.core.llm.llm import ModelNotFoundError, get_llm
from langchain_lab.core.tools.tool_middleware import execute_safe_tool_call
from langchain_lab.core.tools.tools import get_tools
from langchain_lab.util.prompt_style import Printer, bindings

# 屏蔽 httpx 的INFO 级别日志，调用llm时会打印
logging.getLogger("httpx").setLevel(logging.WARNING)

init(autoreset=True)

printer = Printer()

MAX_LLM_LOOP = 3
MAX_TOOL_CALLS = 4

session_ctx = get_config().project.session_ctx


async def main():
    session = PromptSession()

    try:
        llm = await asyncio.to_thread(get_llm)
        tools = get_tools()
        llm_with_tools = llm.bind_tools(tools=tools)
        printer.reset_all_state()

        printer.begin("X" * 50 + " 开始你的多轮对话 " + "X" * 50)
        # 第一轮：模型决定调用工具
        system_prompt = (
            await session.prompt_async(
                printer.message(
                    "（可回车跳过——内置默认系统提示词为：你是一个配合测试的助手，按照用户要求执行Agent循环任务）\n"
                    "请输入你的系统提示词(shift + enter 换行)：\n> ",
                    style_name=printer.style_name.MESSAGE,
                ),
                style=printer.style,
                key_bindings=bindings,
                multiline=True,
                prompt_continuation=printer.message("> ", style_name=printer.style_name.MESSAGE),
            )
            or "你是一个配合测试的助手，按照用户要求执行最小Agent循环任务"
        )

        messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]

        print("=" * 38 + '\033[1;34m 开始聊天 or 输入"0" 退出 \033[0m' + "=" * 40)

        # 多轮对话循环
        while True:
            human_prompt = (
                await session.prompt_async(
                    printer.message("请向你的助手发送消息(shift + enter 换行)：\n> ", style_name=printer.style_name.MESSAGE),
                    style=printer.style,
                    key_bindings=bindings,
                    multiline=True,
                    prompt_continuation=printer.message("> ", style_name=printer.style_name.MESSAGE),
                )
                or "我还没想好我的问题，你给我一点建议"
            )
            if human_prompt == "0":
                printer.warning("X" * 50 + " 用户选择终止 " + "X" * 50 + "\n\n")
                break
            messages.append(HumanMessage(content=human_prompt))
            messages = await _single_chat_loop(tools=tools, llm_with_tools=llm_with_tools, messages=messages, session_ctx=session_ctx)
            print("\n\n" + "=" * 38 + '\033[1;32m 继续聊天 or 输入"0" 退出 \033[0m' + "=" * 40)

    except ModelNotFoundError as e:
        logging.warning(f"模型选择错误：{e}")
        raise


async def _single_chat_loop(
    tools: list[BaseTool],
    llm_with_tools: Runnable,
    messages: list[BaseMessage],
    session_ctx: dict,
    llm: BaseChatModel | None = None,
) -> list[BaseMessage]:
    """单次对话循环完成任务"""
    tool_call_count = 0
    llm_loop_count = 0
    print(Fore.BLUE + "=" * 30 + "用户发起调用 llm " + "=" * 30)
    while True:
        injected_llm_limit_msg: SystemMessage | None = None
        injected_tool_limit_msg: SystemMessage | None = None

        # 到达llm轮次上限，准备注入临时提示
        if llm_loop_count >= MAX_LLM_LOOP:  # 先判断，再自增，用 >=
            injected_llm_limit_msg = SystemMessage(
                content="本轮对话已到达最大Agent轮次上限，请基于现有全部信息直接给出最终答案，禁止调用任何工具。",
                additional_kwargs={"meta": "llm_loop_break"},
            )

        if tool_call_count >= MAX_TOOL_CALLS:  # 先判断，再自增，用 >=
            injected_tool_limit_msg = SystemMessage(
                content="本轮对话工具调用次数已耗尽，不再执行后续工具，请基于已获得的工具结果总结回答，禁止继续调用工具",
                additional_kwargs={"meta": "tool_call_break"},
            )

        # 临时消息临时加入消息列表
        current_messages = messages.copy()
        if injected_llm_limit_msg is not None:
            current_messages.append(injected_llm_limit_msg)
        if injected_tool_limit_msg is not None:
            current_messages.append(injected_tool_limit_msg)

        llm_loop_count += 1  # 比工具多了一轮自增
        # 使用拷贝后的current_messages调用stream，原始messages不受临时标记影响！
        result = llm_with_tools.stream(current_messages)

        printer.label_blue("模型答复> ")

        full_chunk: AIMessage | AIMessageChunk | None = None
        for chunk in result:
            # region：调试
            # # 调试用1：观察 chunk 的所有属性和方法
            # print(dir(chunk))

            # # 调试用2：实时打印工具调用增量片段）——结果：集中一个 chunk 里面抛
            # if isinstance(chunk, AIMessageChunk) and chunk.tool_call_chunks:
            #     print(f"\n 工具调用增量分片：{chunk.tool_call_chunks}")

            # # 调试用3：打印单个 chunk —— 结果详：docs
            # # 打印LangChain封装后的完整对象结构
            # print(json.dumps(chunk.model_dump(), ensure_ascii=False, indent=2))
            # # 单独打印当前分片的tool_call_chunks
            # if isinstance(chunk, AIMessageChunk) and chunk.tool_call_chunks:
            #     print(json.dumps(chunk.tool_call_chunks, ensure_ascii=False, indent=2))
            # endregion

            # 累加所有分片，合并成完整消息
            full_chunk = chunk if full_chunk is None else cast(AIMessage | AIMessageChunk, full_chunk + chunk)

            # 打印文本片段
            if chunk.content:
                # print(chunk.content, end="", flush=True)
                printer.stream_chunk(chunk.content, style_name=printer.style_name.MAGENTA)

        print()
        printer.label_blue("<答复完成")

        # 没有流式内容输出，直接终止循环
        if not full_chunk:
            break

        # 流式结果存入上下文
        messages.append(full_chunk)

        # 最后答复后退出 while
        # LLM 的收尾轮会走多一次自增流程，计数溢出上限；工具的收尾直接无额外自增，计数顶格停在上限
        if llm_loop_count > MAX_LLM_LOOP or tool_call_count >= MAX_TOOL_CALLS:  # 因为LLM注入系统消息到这里多了一次计数
            printer.error("=" * 35 + " 本次对话轮数或工具调用次数已耗尽~ " + "=" * 35)
            break

        # region 笔记：
        # 1、❌ 不要用循环内单 chunk 的`chunk.tool_call_chunks`直接执行业务逻辑：
        # 它是分片字符串，可能不完整。只能用来调试、实时 UI 展示。
        # 2、✅ 最稳定方案：等整轮 stream 迭代全部结束，使用合并完成后的full_chunk.tool_calls去调用工具。
        # 此时 args 已经是解析好的字典，稳定可靠。具体如下：
        # endregion
        # 判断是否需要调用工具：流式打印结束后，拿到完整解析好的 tool_calls
        if full_chunk.tool_calls:
            printer.label_yellow("有工具执行> ")
            print("-" * 25 + " 工具执行详情 " + "-" * 25)
            print("llm 想调用的全部工具：")
            for i, tc in enumerate(full_chunk.tool_calls, 1):
                print(f"工具 {[i]} 名称：{tc['name']}； 调用ID：{tc['id']}")
                print(f"      └─ 参数：{tc['args']}")

            # 上面报告完再执行
            print("-" * 50)
            print("下面开始执行工具：")
            for i, tc in enumerate(full_chunk.tool_calls, 1):
                # 超限退出for循环，否则继续调用工具，统一外面负责注入 SystemMessage
                if tool_call_count >= MAX_TOOL_CALLS:  # 先判断，再自增，用 >=
                    break

                tool_call_count += 1  # 业务计数 + 1

                # ✅ 一行调用，所有中间件全部内置
                result = await execute_safe_tool_call(tool_call=tc, tools=tools, session_ctx=session_ctx, tool_index=i)

                # 统一拼装 ToolMessage
                messages.append(
                    ToolMessage(content=json.dumps(result["content"], ensure_ascii=False), tool_call_id=result["tool_call_id"], name=result["name"])
                )

                # region 已迁移至 工具调用中间件
                # # 查找是否有匹配的工具
                # target_tool = next((tool for tool in tools if tool.name == tc["name"]), None)
                # if target_tool is None:
                #     messages.append(
                #         ToolMessage(
                #             content=f"未发现工具：{tc['name']}，能用有效工具为:{[t.name for t in tools]}",
                #             tool_call_id=tc["id"],
                #             name=tc["name"],
                #         )
                #     )
                #     print(f"📢向大哥报告：llm 调用了不存在的工具{[i]}：{tc['name']}")
                # else:
                #     print(f"执行工具{[i]}：{tc['name']}")
                #     tool_result = await target_tool.ainvoke(tc["args"])

                #     messages.append(ToolMessage(content=json.dumps(tool_result), tool_call_id=tc["id"], name=tc["name"]))
                #     print(f"       └─ 执行结果：{tool_result}")
                # endregion

            # 有工具调用，回到循环，再次带工具调用 LLM
            print("-" * 25 + " 工具执行结束 " + "-" * 25)
            print()
            print(Fore.GREEN + "=" * 30 + "自动再次调用 llm " + "=" * 30)
            continue
        else:
            # 没有工具调用，模型输出最终答案，退出Agent循环
            break

    return messages


if __name__ == "__main__":
    asyncio.run(main())
