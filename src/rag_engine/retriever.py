"""
檢索器 - 從向量資料庫檢索相關文檔
"""

from langchain_chroma import Chroma
from langchain_core.documents import Document


class DocumentRetriever:
    """文檔檢索器"""

    def __init__(self, vectorstore: Chroma, top_k: int = 5):
        self.vectorstore = vectorstore
        self.top_k = top_k

    def retrieve(self, query: str, top_k: int | None = None) -> list[Document]:
        """檢索與查詢相關的文檔"""
        k = top_k or self.top_k
        docs = self.vectorstore.similarity_search(query, k=k)
        return docs

    def retrieve_with_scores(
        self, query: str, top_k: int | None = None
    ) -> list[tuple[Document, float]]:
        """檢索文檔並回傳相似度分數"""
        k = top_k or self.top_k
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return results

    def format_context(self, docs: list[Document]) -> str:
        """將檢索到的文檔格式化為上下文字串"""
        context_parts = []
        for i, doc in enumerate(docs, 1):
            title = doc.metadata.get("title", "未知來源")
            content = doc.page_content.strip()
            context_parts.append(f"[來源 {i}: {title}]\n{content}")
        return "\n\n---\n\n".join(context_parts)
