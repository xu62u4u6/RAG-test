"""LLM-as-Judge 評分器"""

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from .models import EvalResult


JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一位醫療問答系統的評估專家。請根據以下三個維度對 RAG 系統的回答進行評分，每項分數為 0.0 到 1.0。

評分維度：
1. **faithfulness（忠實性）**：RAG 答案的內容是否有根據、不包含捏造資訊？
   - 1.0：完全根據事實，無捏造
   - 0.5：大致正確但有輕微不準確
   - 0.0：包含明顯錯誤或捏造

2. **answer_relevancy（相關性）**：RAG 答案是否有回答到問題？
   - 1.0：完整回答問題
   - 0.5：部分回答
   - 0.0：答非所問或拒絕回答

3. **correctness（正確性）**：RAG 答案與標準答案相比，內容是否正確一致？
   - 1.0：與標準答案高度一致
   - 0.5：部分一致，有遺漏但無錯誤
   - 0.0：與標準答案明顯矛盾

請以 JSON 格式回傳，包含 faithfulness、answer_relevancy、correctness 三個分數，以及一行 reasoning 說明理由。"""),
    ("human", """問題：{question}

標準答案：
{ground_truth}

RAG 系統的回答：
{rag_answer}

請評分："""),
])


class LLMJudge:
    """使用 LLM 評估 RAG 回答品質"""

    def __init__(self, llm: BaseChatModel):
        self.chain = JUDGE_PROMPT | llm | JsonOutputParser()

    def evaluate(self, result: EvalResult) -> EvalResult:
        """評估單題，回傳填入分數的 EvalResult"""
        scores = self.chain.invoke({
            "question": result.question,
            "ground_truth": result.ground_truth,
            "rag_answer": result.rag_answer,
        })
        result.faithfulness = float(scores.get("faithfulness", 0.0))
        result.answer_relevancy = float(scores.get("answer_relevancy", 0.0))
        result.correctness = float(scores.get("correctness", 0.0))
        result.reasoning = scores.get("reasoning", "")
        return result
