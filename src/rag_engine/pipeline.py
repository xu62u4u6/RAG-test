"""
RAG Pipeline - 整合檢索與生成的完整流程
"""

from pathlib import Path
from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_core.language_models import BaseChatModel

from .indexer import DocumentIndexer
from .retriever import DocumentRetriever
from .generator import AnswerGenerator


@dataclass
class RAGResponse:
    """RAG 回應結果"""
    question: str
    answer: str
    sources: list[dict]
    context: str


class RAGPipeline:
    """RAG 完整流程管線"""

    def __init__(
        self,
        llm: BaseChatModel,
        vectorstore: Chroma | None = None,
        persist_directory: str = "./data/vectordb",
        embedding_model: str = "BAAI/bge-large-zh-v1.5",
        top_k: int = 5,
    ):
        self.persist_directory = Path(persist_directory)
        self.embedding_model = embedding_model
        self.top_k = top_k

        # 初始化元件
        self.indexer = DocumentIndexer(
            embedding_model=embedding_model,
            persist_directory=persist_directory,
        )

        if vectorstore:
            self.vectorstore = vectorstore
        elif self.persist_directory.exists():
            print("載入現有向量索引...")
            self.vectorstore = self.indexer.load_vectorstore()
        else:
            self.vectorstore = None

        self.retriever = None
        if self.vectorstore:
            self.retriever = DocumentRetriever(self.vectorstore, top_k=top_k)

        self.generator = AnswerGenerator(llm)

    def index_documents(self, documents_dir: Path) -> None:
        """建立文件索引"""
        self.vectorstore = self.indexer.index_from_directory(documents_dir)
        self.retriever = DocumentRetriever(self.vectorstore, top_k=self.top_k)

    def query(self, question: str, top_k: int | None = None) -> RAGResponse:
        """執行 RAG 查詢"""
        if not self.retriever:
            raise ValueError("尚未建立索引，請先執行 index_documents()")

        # 1. 檢索相關文檔
        k = top_k or self.top_k
        docs_with_scores = self.retriever.retrieve_with_scores(question, top_k=k)

        # 2. 格式化上下文
        docs = [doc for doc, _ in docs_with_scores]
        context = self.retriever.format_context(docs)

        # 3. 生成回答
        answer = self.generator.generate(question, context)

        # 4. 整理來源資訊
        sources = []
        for doc, score in docs_with_scores:
            sources.append({
                "title": doc.metadata.get("title", "未知"),
                "source": doc.metadata.get("source", ""),
                "score": float(score),
                "content_preview": doc.page_content[:200] + "...",
            })

        return RAGResponse(
            question=question,
            answer=answer,
            sources=sources,
            context=context,
        )

    async def aquery(self, question: str, top_k: int | None = None) -> RAGResponse:
        """非同步執行 RAG 查詢"""
        if not self.retriever:
            raise ValueError("尚未建立索引，請先執行 index_documents()")

        k = top_k or self.top_k
        docs_with_scores = self.retriever.retrieve_with_scores(question, top_k=k)
        docs = [doc for doc, _ in docs_with_scores]
        context = self.retriever.format_context(docs)

        answer = await self.generator.agenerate(question, context)

        sources = []
        for doc, score in docs_with_scores:
            sources.append({
                "title": doc.metadata.get("title", "未知"),
                "source": doc.metadata.get("source", ""),
                "score": float(score),
                "content_preview": doc.page_content[:200] + "...",
            })

        return RAGResponse(
            question=question,
            answer=answer,
            sources=sources,
            context=context,
        )


def create_pipeline(
    llm: BaseChatModel,
    persist_directory: str = "./data/vectordb",
    embedding_model: str = "BAAI/bge-large-zh-v1.5",
    top_k: int = 5,
) -> RAGPipeline:
    """建立 RAG Pipeline 的便捷函式"""
    return RAGPipeline(
        llm=llm,
        persist_directory=persist_directory,
        embedding_model=embedding_model,
        top_k=top_k,
    )
