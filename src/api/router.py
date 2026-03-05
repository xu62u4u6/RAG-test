"""
API 路由設定 - 處理問答請求
"""

from fastapi import APIRouter, Depends, HTTPException

from src.api.models import ChatRequest, ChatResponse, SourceInfo
from src.api.dependencies import get_rag_pipeline
from src.rag_engine.pipeline import RAGPipeline

router = APIRouter(prefix="/api", tags=["Chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """
    接收使用者的問題，利用 RAG 系統檢索並產生回答
    """
    try:
        # 使用非同步方法處理請求
        result = await pipeline.aquery(request.question, top_k=request.top_k)
        
        # 轉換來源格式
        sources = [
            SourceInfo(
                title=src["title"],
                source=src["source"],
                score=src["score"],
                content_preview=src["content_preview"]
            )
            for src in result.sources
        ]
        
        # 回傳 API 格式
        return ChatResponse(
            question=result.question,
            answer=result.answer,
            sources=sources
        )
        
    except ValueError as e:
        # 處理尚未建立索引的情況
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # 處理其他例外
        raise HTTPException(status_code=500, detail=f"系統內部錯誤: {str(e)}")
