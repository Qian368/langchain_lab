"""管道组件：组合模型、模版、解析器"""

from langchain_core.output_parsers import StrOutputParser

from langchain_lab.core.llm import ModelNotFoundError, get_llm, logging
from langchain_lab.core.messages import get_basic_prompt


def get_basic_chain():
    """
    基础管道：模板 | 模型 | 解析器
    捕获模型不存在异常，输出业务告警日志，异常继续向上抛出
    """
    try:
        prompt = get_basic_prompt()
        llm = get_llm()
        parser = StrOutputParser()

        return prompt | llm | parser

    except ModelNotFoundError as e:
        logging.warning(f"模型选择错误：{e}")
        # 直接raise 不要再 raise e ，如果打印堆栈会指向这里，不会指向初始异常抛出的地方
        raise
