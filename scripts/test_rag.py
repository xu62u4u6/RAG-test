#!/usr/bin/env python3
"""
快速測試 RAG 系統
用法: python scripts/test_rag.py
"""

import os
import sys
from pathlib import Path

# 加入專案根目錄到 path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()


def main():
    print("=" * 60)
    print("神經內科 RAG 系統測試")
    print("=" * 60)

    # 檢查環境變數
    provider = os.getenv("LLM_PROVIDER", "openai")
    print(f"\nLLM 提供商: {provider}")

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("錯誤: 請在 .env 檔案中設定 OPENAI_API_KEY")
            return
        print(f"API Key: {api_key[:10]}...")
    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("錯誤: 請在 .env 檔案中設定 ANTHROPIC_API_KEY")
            return
        print(f"API Key: {api_key[:10]}...")

    # 建立 LLM
    print("\n初始化 LLM...")
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        llm = ChatOpenAI(model=model, temperature=0.7)
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
        llm = ChatAnthropic(model=model, temperature=0.7)

    # 建立 RAG Pipeline
    vectordb_path = Path(os.getenv("VECTORDB_PATH", "./data/vectordb"))
    data_dir = project_root / "data" / "raw" / "vghtpe"

    print("\n載入 RAG Pipeline...")
    from src.rag_engine import RAGPipeline

    pipeline = RAGPipeline(
        llm=llm,
        persist_directory=str(vectordb_path),
    )

    # 如果沒有 retriever（索引為空或不存在），建立索引
    if pipeline.retriever is None:
        print("\n向量索引為空或不存在，正在建立...")
        print("-" * 60)

        if not data_dir.exists():
            print(f"錯誤: 資料目錄不存在: {data_dir}")
            print("請先執行爬蟲: python scripts/scrape_vghtpe.py")
            return

        pipeline.index_documents(data_dir)
        print("-" * 60)

    # 測試問題
    test_questions = [
        "失智症可以治療嗎？",
        "治療阿茲海默症的藥物有哪些？",
        "失智症藥物有什麼副作用？",
    ]

    print("\n" + "=" * 60)
    print("開始測試查詢")
    print("=" * 60)

    for i, question in enumerate(test_questions, 1):
        print(f"\n[問題 {i}] {question}")
        print("-" * 40)

        try:
            response = pipeline.query(question)
            print(f"[回答]\n{response.answer}")
            print(f"\n[來源] 共參考 {len(response.sources)} 份文件")
        except Exception as e:
            print(f"錯誤: {e}")

        print("-" * 40)

    print("\n測試完成！")


if __name__ == "__main__":
    main()
