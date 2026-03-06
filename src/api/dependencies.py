"""
API 依賴注入 (Dependencies) 與共用資源管理
"""

import asyncio
import os
from typing import Optional

from fastapi import HTTPException

from src.rag_engine.pipeline import RAGPipeline
from src.cli import get_llm

# 全局 RAG 管線實例與載入狀態
_rag_pipeline_instance: Optional[RAGPipeline] = None
_loading: bool = False
_loaded: bool = False


async def init_pipeline_background() -> None:
    """在背景線程中初始化 RAG Pipeline"""
    global _rag_pipeline_instance, _loading, _loaded

    if _loaded or _loading:
        return

    _loading = True
    print("⏳ [RAG Pipeline] 正在背景載入模型...")

    def _init():
        vectordb_path = os.getenv("VECTORDB_PATH", "./data/vectordb")
        embedding_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
        llm = get_llm()
        return RAGPipeline(
            llm=llm,
            persist_directory=vectordb_path,
            embedding_model=embedding_model,
        )

    _rag_pipeline_instance = await asyncio.to_thread(_init)
    _loaded = True
    _loading = False

    if _rag_pipeline_instance.retriever is None:
        print("⚠️ [RAG Pipeline] 模型已載入，但向量索引為空。請先執行 index 建立索引。")
    else:
        print("✅ [RAG Pipeline] 已成功載入並初始化。")


def get_rag_pipeline() -> RAGPipeline:
    """
    取得 RAG Pipeline 實例 (依賴注入使用)
    若模型尚未載入完成，回傳 503 提示前端稍後再試
    """
    if not _loaded or _rag_pipeline_instance is None:
        raise HTTPException(
            status_code=503,
            detail="系統正在載入模型，請稍後再試...",
        )
    return _rag_pipeline_instance
