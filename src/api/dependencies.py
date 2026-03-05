"""
API 依賴注入 (Dependencies) 與共用資源管理
"""

import os
from typing import Optional

from src.rag_engine.pipeline import RAGPipeline
from src.cli import get_llm

# 全局 RAG 管線實例
_rag_pipeline_instance: Optional[RAGPipeline] = None


def get_rag_pipeline() -> RAGPipeline:
    """
    取得 RAG Pipeline 實例 (依賴注入使用)
    確保在應用程式生命週期內，只會建立一次向量資料庫連線和模型實例
    """
    global _rag_pipeline_instance
    
    if _rag_pipeline_instance is None:
        vectordb_path = os.getenv("VECTORDB_PATH", "./data/vectordb")
        llm = get_llm()
        
        # 建立共用的 RAG Pipeline
        _rag_pipeline_instance = RAGPipeline(
            llm=llm,
            persist_directory=vectordb_path,
        )
        print("✅ [RAG Pipeline] 已成功載入並初始化。")
        
    return _rag_pipeline_instance
