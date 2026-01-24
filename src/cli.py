"""
CLI 入口 - 神經內科 RAG 系統命令列介面
"""

import os
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

app = typer.Typer(
    name="neuro-rag",
    help="神經內科醫療 RAG 系統",
    add_completion=False,
)
console = Console()


def get_llm():
    """根據環境變數取得 LLM 實例"""
    provider = os.getenv("LLM_PROVIDER", "openai")

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=model, temperature=0.7)
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
        return ChatAnthropic(model=model, temperature=0.7)
    else:
        raise ValueError(f"不支援的 LLM 提供商: {provider}")


@app.command()
def index(
    data_dir: Path = typer.Option(
        Path("./data/raw/vghtpe"),
        "--data-dir", "-d",
        help="文件目錄路徑",
    ),
    vectordb_path: Path = typer.Option(
        Path("./data/vectordb"),
        "--vectordb", "-v",
        help="向量資料庫儲存路徑",
    ),
    embedding_model: str = typer.Option(
        "BAAI/bge-large-zh-v1.5",
        "--embedding", "-e",
        help="Embedding 模型名稱",
    ),
):
    """建立文件向量索引"""
    from src.rag_engine import DocumentIndexer

    console.print(f"\n[bold blue]建立向量索引[/bold blue]")
    console.print(f"文件目錄: {data_dir}")
    console.print(f"向量庫路徑: {vectordb_path}")
    console.print(f"Embedding 模型: {embedding_model}\n")

    indexer = DocumentIndexer(
        embedding_model=embedding_model,
        persist_directory=str(vectordb_path),
    )
    indexer.index_from_directory(data_dir)

    console.print("\n[bold green]索引建立完成！[/bold green]")


@app.command()
def query(
    question: str = typer.Argument(..., help="要詢問的問題"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="檢索文件數量"),
    show_sources: bool = typer.Option(False, "--sources", "-s", help="顯示來源資訊"),
):
    """查詢 RAG 系統"""
    from src.rag_engine import RAGPipeline

    vectordb_path = os.getenv("VECTORDB_PATH", "./data/vectordb")

    if not Path(vectordb_path).exists():
        console.print("[bold red]錯誤：向量索引不存在，請先執行 index 命令[/bold red]")
        raise typer.Exit(1)

    console.print(f"\n[bold blue]問題：[/bold blue]{question}\n")

    with console.status("[bold green]正在檢索並生成回答..."):
        llm = get_llm()
        pipeline = RAGPipeline(
            llm=llm,
            persist_directory=vectordb_path,
            top_k=top_k,
        )
        response = pipeline.query(question)

    # 顯示回答
    console.print(Panel(
        Markdown(response.answer),
        title="[bold green]回答[/bold green]",
        border_style="green",
    ))

    # 顯示來源
    if show_sources:
        console.print("\n[bold blue]參考來源：[/bold blue]")
        for i, source in enumerate(response.sources, 1):
            console.print(f"  {i}. {source['title']} (相似度: {source['score']:.3f})")


@app.command()
def chat(
    top_k: int = typer.Option(5, "--top-k", "-k", help="檢索文件數量"),
):
    """互動式對話模式"""
    from src.rag_engine import RAGPipeline

    vectordb_path = os.getenv("VECTORDB_PATH", "./data/vectordb")

    if not Path(vectordb_path).exists():
        console.print("[bold red]錯誤：向量索引不存在，請先執行 index 命令[/bold red]")
        raise typer.Exit(1)

    console.print(Panel(
        "歡迎使用神經內科 RAG 問答系統\n輸入問題進行查詢，輸入 'exit' 或 'quit' 離開",
        title="[bold blue]互動模式[/bold blue]",
        border_style="blue",
    ))

    llm = get_llm()
    pipeline = RAGPipeline(
        llm=llm,
        persist_directory=vectordb_path,
        top_k=top_k,
    )

    while True:
        try:
            question = console.input("\n[bold cyan]您的問題：[/bold cyan] ")
            if question.lower() in ["exit", "quit", "q"]:
                console.print("\n[bold yellow]再見！[/bold yellow]")
                break
            if not question.strip():
                continue

            with console.status("[bold green]思考中..."):
                response = pipeline.query(question)

            console.print(Panel(
                Markdown(response.answer),
                title="[bold green]回答[/bold green]",
                border_style="green",
            ))

        except KeyboardInterrupt:
            console.print("\n\n[bold yellow]已中斷[/bold yellow]")
            break


if __name__ == "__main__":
    app()
