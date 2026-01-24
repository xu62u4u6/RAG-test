"""
生成器 - 基於檢索到的上下文生成回答
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseChatModel


SYSTEM_PROMPT = """你是臺北榮民總醫院神經內科的衛教助理，專門回答病人關於神經內科疾病的問題。

請根據提供的參考資料回答問題。回答時請注意：
1. 使用病人能理解的語言，避免過多專業術語
2. 如果需要使用專業術語，請附上簡單解釋
3. 回答要準確、簡潔
4. 如果參考資料中沒有相關資訊，請誠實說明並建議病人諮詢醫師
5. 不要編造資訊

參考資料：
{context}
"""

USER_PROMPT = """問題：{question}

請根據上述參考資料回答這個問題。"""


class AnswerGenerator:
    """回答生成器"""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", USER_PROMPT),
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()

    def generate(self, question: str, context: str) -> str:
        """根據問題和上下文生成回答"""
        response = self.chain.invoke({
            "question": question,
            "context": context,
        })
        return response

    async def agenerate(self, question: str, context: str) -> str:
        """非同步生成回答"""
        response = await self.chain.ainvoke({
            "question": question,
            "context": context,
        })
        return response
