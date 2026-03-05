"""
API 請求與回應的 Pydantic 模型定義
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """使用者提問的請求格式"""
    question: str = Field(..., description="使用者的問題")
    top_k: Optional[int] = Field(5, description="要檢索的相關文件數量")


class SourceInfo(BaseModel):
    """參考來源資訊"""
    title: str = Field(..., description="文件標題")
    source: str = Field(..., description="來源名稱")
    score: float = Field(..., description="關聯度分數")
    content_preview: str = Field(..., description="內容預覽")


class ChatResponse(BaseModel):
    """API 回答格式"""
    question: str = Field(..., description="原始問題")
    answer: str = Field(..., description="AI 生成的回答")
    sources: List[SourceInfo] = Field(default_factory=list, description="參考的來源資料")
