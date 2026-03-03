"""從衛教文章自動生成問答題"""

from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from .models import QAPair, QuestionSet


GENERATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一位醫療教育專家，負責根據衛教文章設計問答題，用來評估 RAG 問答系統的品質。

請根據提供的文章內容，生成 {n} 道問答題。每道題需符合以下要求：
1. 問題必須能從文章中找到明確答案
2. 問題要多樣化，涵蓋文章不同段落與面向
3. 標準答案要完整、準確，直接摘錄或整理自原文
4. 問題要用病人或家屬可能提問的自然語言

請以 JSON 陣列格式回傳，每個元素包含：
- "question": 問題文字
- "ground_truth": 標準答案

只回傳 JSON，不要有其他說明文字。"""),
    ("human", """文章標題：{title}

文章內容：
{content}

請生成 {n} 道問答題："""),
])


class QuestionGenerator:
    """從衛教文章自動生成問答題"""

    def __init__(self, llm: BaseChatModel, questions_per_doc: int = 5):
        self.llm = llm
        self.questions_per_doc = questions_per_doc
        self.chain = GENERATE_PROMPT | llm | JsonOutputParser()

    def generate_from_file(self, txt_path: Path) -> list[QAPair]:
        """從單一文字檔生成問答題"""
        content = txt_path.read_text(encoding="utf-8")

        # 解析標題
        lines = content.split("\n")
        title = lines[0].replace("# ", "").strip() if lines else txt_path.stem

        print(f"  生成題目：{title}...")
        result = self.chain.invoke({
            "title": title,
            "content": content,
            "n": self.questions_per_doc,
        })

        pairs = []
        for item in result:
            pairs.append(QAPair(
                question=item["question"],
                ground_truth=item["ground_truth"],
                source_title=title,
                source_file=str(txt_path),
            ))
        print(f"    ✓ 生成 {len(pairs)} 題")
        return pairs

    def generate_from_directory(self, docs_dir: Path) -> QuestionSet:
        """從目錄中所有 txt 檔生成問答題集"""
        docs_dir = Path(docs_dir)
        all_pairs = []

        txt_files = sorted(docs_dir.glob("*.txt"))
        print(f"找到 {len(txt_files)} 篇文章，每篇生成 {self.questions_per_doc} 題\n")

        for txt_file in txt_files:
            try:
                pairs = self.generate_from_file(txt_file)
                all_pairs.extend(pairs)
            except Exception as e:
                print(f"    ✗ 失敗：{e}")

        print(f"\n共生成 {len(all_pairs)} 題")
        return QuestionSet(pairs=all_pairs)
