"""評估流程執行器"""

from langchain_core.language_models import BaseChatModel

from ..question_generator.models import QuestionSet
from ..rag_engine.pipeline import RAGPipeline
from .judge import LLMJudge
from .models import EvalResult, EvalReport


class EvalRunner:
    """執行完整評估流程：RAG 回答 → LLM 評分 → 產出報告"""

    def __init__(self, rag_pipeline: RAGPipeline, judge_llm: BaseChatModel):
        self.pipeline = rag_pipeline
        self.judge = LLMJudge(judge_llm)

    def run(self, question_set: QuestionSet, verbose: bool = True) -> EvalReport:
        report = EvalReport()
        total = len(question_set)

        for i, qa in enumerate(question_set.pairs, 1):
            if verbose:
                print(f"[{i:>3}/{total}] {qa.question[:50]}...")

            # 1. RAG 回答
            rag_response = self.pipeline.query(qa.question)

            # 2. 建立評估結果
            result = EvalResult(
                question=qa.question,
                ground_truth=qa.ground_truth,
                rag_answer=rag_response.answer,
                source_title=qa.source_title,
            )

            # 3. LLM 評分
            result = self.judge.evaluate(result)
            report.results.append(result)

            if verbose:
                print(
                    f"       faithfulness={result.faithfulness:.1f}  "
                    f"relevancy={result.answer_relevancy:.1f}  "
                    f"correctness={result.correctness:.1f}"
                )

        return report
