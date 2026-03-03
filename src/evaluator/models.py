"""評估結果資料模型"""

from dataclasses import dataclass, field, asdict
import json


@dataclass
class EvalResult:
    """單題評估結果"""
    question: str
    ground_truth: str
    rag_answer: str
    source_title: str

    # LLM-as-judge 評分（0.0 ~ 1.0）
    faithfulness: float = 0.0      # 答案是否忠於檢索到的文件
    answer_relevancy: float = 0.0  # 答案是否回答到問題
    correctness: float = 0.0       # 答案是否與 ground truth 一致

    reasoning: str = ""            # 評分理由


@dataclass
class EvalReport:
    """整體評估報告"""
    results: list[EvalResult] = field(default_factory=list)

    @property
    def avg_faithfulness(self) -> float:
        return sum(r.faithfulness for r in self.results) / len(self.results) if self.results else 0.0

    @property
    def avg_answer_relevancy(self) -> float:
        return sum(r.answer_relevancy for r in self.results) / len(self.results) if self.results else 0.0

    @property
    def avg_correctness(self) -> float:
        return sum(r.correctness for r in self.results) / len(self.results) if self.results else 0.0

    def save(self, path: str) -> None:
        data = {
            "summary": {
                "total": len(self.results),
                "avg_faithfulness": round(self.avg_faithfulness, 3),
                "avg_answer_relevancy": round(self.avg_answer_relevancy, 3),
                "avg_correctness": round(self.avg_correctness, 3),
            },
            "results": [asdict(r) for r in self.results],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "EvalReport":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        results = [EvalResult(**r) for r in data["results"]]
        return cls(results=results)
