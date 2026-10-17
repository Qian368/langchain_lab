"""实验2：单轮对话（single——chat）：工具调用 + 流式输出 + 最小循环"""

import asyncio
import json
from typing import cast

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from prompt_toolkit import prompt

from langchain_lab.core.llm.llm import ModelNotFoundError, get_llm
from langchain_lab.core.tools.tools import get_base_tools
from langchain_lab.util.logger import LogName, setup_logger

logger = setup_logger(LogName.EXPER)


async def main():
    print("\n---实验2：工具调用---")

    MAX_LLM_LOOP = 5
    MAX_TOOL_CALLS = 10
    tool_call_count = 0
    llm_loop_count = 0

    try:
        llm = get_llm()
        tools = get_base_tools()
        llm_with_tools = llm.bind_tools(tools=tools)

        # 第一轮：模型决定调用工具
        system_prompt = await asyncio.to_thread(prompt, "请输入系统提示词> ") or "你是一个配合测试的助手，按照用户要求执行最小Agent循环任务"
        human_prompt = (
            await asyncio.to_thread(prompt, "请输入你的问题> ")
            or "请输入你的问题：计算55 乘以32，先调用同时两个工具，一个是我给你的有存在的工具，一个是不存在的工具。拿到结果后再回复用户。"
        )
        system_message = SystemMessage(content=system_prompt)
        human_message = HumanMessage(content=human_prompt)

        messages: list[BaseMessage] = [system_message, human_message]

        print("===调用模型并开始回答===")
        while True:
            injected_llm_limit_msg: SystemMessage | None = None
            injected_tool_limit_msg: SystemMessage | None = None

            # 到达llm轮次上限，准备注入临时提示
            if llm_loop_count > MAX_LLM_LOOP:
                injected_llm_limit_msg = SystemMessage(
                    content="本轮对话已到达最大Agent轮次上限，请基于现有全部信息直接给出最终答案，禁止调用任何工具。",
                    additional_kwargs={"meta": "llm_loop_break"},
                )

            if tool_call_count > MAX_TOOL_CALLS:
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

            llm_loop_count += 1  # 业务计数 + 1
            # 使用拷贝后的current_messages调用stream，原始messages不受临时标记影响！
            result = llm_with_tools.stream(current_messages)

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
                    print(chunk.content, end="", flush=True)

            # 没有流式内容输出，直接终止循环
            if not full_chunk:
                break

            # 流式结果存入上下文
            messages.append(full_chunk)

            # 最后答复后退出 while
            if llm_loop_count > MAX_LLM_LOOP + 1 or tool_call_count > MAX_TOOL_CALLS:  # 因为LLM注入系统消息到这里多了一次计数
                print("本次对话轮数或工具调用次数已耗尽~请继续")
                break

            # region 笔记：
            # 1、❌ 不要用循环内单 chunk 的`chunk.tool_call_chunks`直接执行业务逻辑：
            # 它是分片字符串，可能不完整。只能用来调试、实时 UI 展示。
            # 2、✅ 最稳定方案：等整轮 stream 迭代全部结束，使用合并完成后的full_chunk.tool_calls去调用工具。
            # 此时 args 已经是解析好的字典，稳定可靠。具体如下：
            # endregion
            # 判断是否需要调用工具：流式打印结束后，拿到完整解析好的 tool_calls
            if full_chunk.tool_calls:
                print("\n\n报告： llm 想调用的全部工具：")
                for i, tc in enumerate(full_chunk.tool_calls, 1):
                    print(f"工具 {[i]} 名称：{tc['name']}； 调用ID：{tc['id']}")
                    print(f"      └─ 参数：{tc['args']}")

                # 上面报告完再执行
                print("-" * 50)
                print("下面开始执行工具：")
                for i, tc in enumerate(full_chunk.tool_calls, 1):
                    # 超限退出for循环，否则继续调用工具，统一外面负责注入 SystemMessage
                    if tool_call_count > MAX_TOOL_CALLS:
                        break
                    tool_call_count += 1  # 业务计数 + 1

                    # 查找是否有匹配的工具
                    target_tool = next((tool for tool in tools if tool.name == tc["name"]), None)

                    if target_tool is None:
                        messages.append(ToolMessage(content=f"未知工具：{tc['name']}", tool_call_id=tc["id"], name=tc["name"]))
                        print(f"📢向大哥报告：llm 调用了不存在的工具{[i]}：{tc['name']}")
                    else:
                        print(f"执行工具{[i]}：{tc['name']}")
                        tool_result = await target_tool.ainvoke(tc["args"])
                        messages.append(ToolMessage(content=json.dumps(tool_result), tool_call_id=tc["id"], name=tc["name"]))
                        print(f"       └─ 执行结果：{tool_result}")
                # 有工具调用，回到循环，再次带工具调用 LLM
                print()
                print("=" * 30 + "再次调用 llm " + "=" * 30)
                continue
            else:
                # 没有工具调用，模型输出最终答案，退出Agent循环
                break

    except ModelNotFoundError as e:
        logger.warning(f"模型选择错误：{e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
