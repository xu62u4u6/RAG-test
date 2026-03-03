"""英文內容翻譯成繁體中文"""

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .models import Document


TRANSLATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是醫療翻譯專家，請將以下英文醫療衛教內容翻譯成繁體中文。

翻譯要求：
1. 保持醫療術語準確性，必要時在括號內保留英文原文（如：乙醯膽鹼酶抑制劑（Cholinesterase Inhibitor））
2. 使用台灣習慣用語（如：藥物、醫師、健保）
3. 保留原文段落結構
4. 只回傳翻譯結果，不加說明"""),
    ("human", "{text}"),
])


class Translator:
    """使用 LLM 翻譯英文文件"""

    # 每次翻譯的最大字元數（避免超過 LLM context）
    CHUNK_SIZE = 3000

    def __init__(self, llm: BaseChatModel):
        self.chain = TRANSLATE_PROMPT | llm | StrOutputParser()

    def translate(self, text: str) -> str:
        """翻譯文字，自動分段處理長文"""
        if len(text) <= self.CHUNK_SIZE:
            return self.chain.invoke({"text": text})

        # 按段落切分後逐段翻譯
        paragraphs = text.split("\n\n")
        chunks, current = [], ""

        for para in paragraphs:
            if len(current) + len(para) > self.CHUNK_SIZE:
                if current:
                    chunks.append(current.strip())
                current = para
            else:
                current += "\n\n" + para

        if current.strip():
            chunks.append(current.strip())

        translated_parts = [self.chain.invoke({"text": chunk}) for chunk in chunks]
        return "\n\n".join(translated_parts)

    def translate_document(self, doc: Document) -> Document:
        """翻譯整份英文 Document，回傳繁中版本"""
        if doc.language == "zh":
            return doc

        print(f"  翻譯：{doc.title}...")
        translated_content = self.translate(doc.content)
        translated_title = self.chain.invoke({"text": doc.title})

        return Document(
            id=doc.id,
            title=translated_title,
            content=translated_content,
            source_url=doc.source_url,
            source_name=doc.source_name,
            language="zh",
        )
