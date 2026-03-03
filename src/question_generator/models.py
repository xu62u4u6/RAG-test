"""問答題資料模型"""

from dataclasses import dataclass, field, asdict
import json


@dataclass
class QAPair:
    """一組問答題"""
    question: str          # 問題
    ground_truth: str      # 標準答案（來自原文）
    source_title: str      # 來源文章標題
    source_file: str       # 來源檔案路徑


@dataclass
class QuestionSet:
    """問答題集合"""
    pairs: list[QAPair] = field(default_factory=list)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump([asdict(p) for p in self.pairs], f, ensure_ascii=False, indent=2)
        print(f"已儲存 {len(self.pairs)} 題至 {path}")

    @classmethod
    def load(cls, path: str) -> "QuestionSet":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        pairs = [QAPair(**item) for item in data]
        return cls(pairs=pairs)

    def __len__(self) -> int:
        return len(self.pairs)
