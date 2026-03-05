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


@app.command()
def collect(
    output_dir: Path = typer.Option(
        Path("./data/collected"),
        "--output", "-o",
        help="收集後文件的輸出目錄",
    ),
    pdf_cache: Path = typer.Option(
        Path("./data/raw/pdf_cache"),
        "--pdf-cache",
        help="PDF 快取目錄",
    ),
):
    """從所有來源收集並處理衛教文件（HTML + PDF + 翻譯）"""
    from src.data import DataCollectionPipeline

    console.print("\n[bold blue]資料收集流程[/bold blue]\n")
    llm = get_llm()
    pipeline = DataCollectionPipeline(llm=llm, output_dir=output_dir, pdf_cache_dir=pdf_cache)
    docs = pipeline.run()
    console.print(f"\n[bold green]完成！共 {len(docs)} 份文件[/bold green]")


@app.command()
def preprocess(
    input_dir: Path = typer.Option(
        Path("./data/collected"),
        "--input", "-i",
        help="collect 指令的輸出目錄",
    ),
    output_dir: Path = typer.Option(
        Path("./data/processed"),
        "--output", "-o",
        help="清理後文件的輸出目錄",
    ),
):
    """清理並標準化收集到的文件"""
    from src.data import Preprocessor

    console.print(f"\n[bold blue]前處理文件[/bold blue]")
    console.print(f"{input_dir} → {output_dir}\n")

    preprocessor = Preprocessor()
    results = preprocessor.process_directory(input_dir, output_dir)
    console.print(f"\n[bold green]完成！{len(results)} 份文件已儲存至 {output_dir}[/bold green]")


@app.command()
def generate_questions(
    data_dir: Path = typer.Option(
        Path("./data/raw/vghtpe"),
        "--data-dir", "-d",
        help="衛教文章目錄",
    ),
    output: Path = typer.Option(
        Path("./data/questions/questions.json"),
        "--output", "-o",
        help="輸出的問答題 JSON 路徑",
    ),
    n: int = typer.Option(5, "--num", "-n", help="每篇文章生成幾題"),
):
    """從衛教文章自動生成問答題"""
    from src.question_generator import QuestionGenerator

    console.print(f"\n[bold blue]自動出題[/bold blue]")
    console.print(f"文章目錄：{data_dir}，每篇 {n} 題\n")

    llm = get_llm()
    generator = QuestionGenerator(llm=llm, questions_per_doc=n)
    question_set = generator.generate_from_directory(data_dir)

    output.parent.mkdir(parents=True, exist_ok=True)
    question_set.save(str(output))
    console.print(f"\n[bold green]完成！共 {len(question_set)} 題，儲存至 {output}[/bold green]")


@app.command()
def evaluate(
    questions_file: Path = typer.Option(
        Path("./data/questions/questions.json"),
        "--questions", "-q",
        help="問答題 JSON 路徑",
    ),
    output: Path = typer.Option(
        Path("./data/eval_report.json"),
        "--output", "-o",
        help="評估報告輸出路徑",
    ),
    top_k: int = typer.Option(5, "--top-k", "-k", help="RAG 檢索文件數量"),
    limit: int = typer.Option(0, "--limit", "-l", help="只評估前 N 題（0 = 全部）"),
):
    """執行 RAG 系統評估，產出品質報告"""
    from src.rag_engine import RAGPipeline
    from src.question_generator import QuestionSet
    from src.evaluator import EvalRunner

    vectordb_path = os.getenv("VECTORDB_PATH", "./data/vectordb")

    console.print(f"\n[bold blue]RAG 評估[/bold blue]")

    llm = get_llm()
    pipeline = RAGPipeline(llm=llm, persist_directory=vectordb_path, top_k=top_k)
    question_set = QuestionSet.load(str(questions_file))

    if limit > 0:
        question_set.pairs = question_set.pairs[:limit]
        console.print(f"評估前 {limit} 題（共 {len(QuestionSet.load(str(questions_file)))} 題）\n")
    else:
        console.print(f"評估全部 {len(question_set)} 題\n")

    runner = EvalRunner(rag_pipeline=pipeline, judge_llm=llm)
    report = runner.run(question_set)

    output.parent.mkdir(parents=True, exist_ok=True)
    report.save(str(output))

    # 顯示摘要
    console.print(f"\n[bold green]評估完成[/bold green]")
    console.print(f"  Faithfulness    : {report.avg_faithfulness:.3f}")
    console.print(f"  Answer Relevancy: {report.avg_answer_relevancy:.3f}")
    console.print(f"  Correctness     : {report.avg_correctness:.3f}")
    console.print(f"\n報告已儲存至 {output}")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="API 伺服器主機位址"),
    port: int = typer.Option(8000, "--port", "-p", help="API 伺服器通訊埠"),
    reload: bool = typer.Option(True, "--reload", "-r", help="是否啟用熱重載 (開發模式)"),
):
    """啟動 FastAPI 後端伺服器與 Web UI"""
    import uvicorn
    
    console.print("\n[bold blue]啟動 RAG 問答系統伺服器[/bold blue]")
    console.print(f"[green]伺服器位址:[/green] http://{host}:{port}")
    console.print(f"[green]API 文件:[/green] http://{host}:{port}/docs")
    console.print(f"[green]網頁對話:[/green] http://{host}:{port}/")
    console.print("[yellow]按下 Ctrl+C 停止伺服器[/yellow]\n")

    uvicorn.run("src.api.main:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
