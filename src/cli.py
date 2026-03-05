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
def init():
    """首次使用設定精靈 — 互動式建立 .env 設定檔"""
    import shutil

    env_path = Path(".env")

    skip_env_setup = False
    if env_path.exists():
        overwrite = typer.confirm("⚠️  .env 已存在，是否覆寫？", default=False)
        if not overwrite:
            console.print("[dim]保留現有 .env 設定[/dim]")
            skip_env_setup = True

    console.print(Panel(
        "歡迎使用神經內科 RAG 系統！\n以下將引導您完成首次設定。",
        title="[bold blue]初始化設定[/bold blue]",
        border_style="blue",
    ))

    if not skip_env_setup:
        # 1. 選擇 LLM 提供商
        provider = typer.prompt(
            "選擇 LLM 提供商 (openai / anthropic)",
            default="openai",
        ).strip().lower()
        if provider not in ("openai", "anthropic"):
            console.print(f"[red]不支援的提供商: {provider}[/red]")
            raise typer.Exit(1)

        # 2. 輸入 API Key
        if provider == "openai":
            api_key = typer.prompt("請輸入 OpenAI API Key", hide_input=True)
            model = typer.prompt("OpenAI 模型名稱", default="gpt-4o-mini")
            env_lines = [
                f"OPENAI_API_KEY={api_key}",
                f"OPENAI_MODEL={model}",
                "",
                "# ANTHROPIC_API_KEY=",
                "# ANTHROPIC_MODEL=claude-sonnet-4-20250514",
            ]
        else:
            api_key = typer.prompt("請輸入 Anthropic API Key", hide_input=True)
            model = typer.prompt("Anthropic 模型名稱", default="claude-sonnet-4-20250514")
            env_lines = [
                "# OPENAI_API_KEY=",
                "# OPENAI_MODEL=gpt-4o-mini",
                "",
                f"ANTHROPIC_API_KEY={api_key}",
                f"ANTHROPIC_MODEL={model}",
            ]

        # 3. 其他設定
        embedding = typer.prompt("Embedding 模型", default="BAAI/bge-large-zh-v1.5")
        vectordb = typer.prompt("向量資料庫路徑", default="./data/vectordb")

        # 4. 寫入 .env
        env_content = "\n".join([
            f"LLM_PROVIDER={provider}",
            "",
            *env_lines,
            "",
            f"EMBEDDING_MODEL={embedding}",
            f"VECTORDB_PATH={vectordb}",
            "",
        ])
        env_path.write_text(env_content, encoding="utf-8")
        console.print(f"\n[green]✅ .env 已建立[/green]")

    # 5. 檢查 uv 是否可用
    if shutil.which("uv"):
        console.print("[green]✅ uv 已安裝[/green]")
    else:
        console.print("[yellow]⚠️  未偵測到 uv，請先安裝: curl -LsSf https://astral.sh/uv/install.sh | sh[/yellow]")
        raise typer.Exit(1)

    # 6. 自動執行資料準備流程（檢查哪些步驟已完成）
    load_dotenv(override=True)
    embedding = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
    vectordb = os.getenv("VECTORDB_PATH", "./data/vectordb")

    collected_dir = Path("./data/collected")
    processed_dir = Path("./data/processed")
    vectordb_dir = Path(vectordb)

    has_collected = collected_dir.exists() and any(collected_dir.iterdir())
    has_processed = processed_dir.exists() and any(processed_dir.iterdir())
    has_index = vectordb_dir.exists() and any(vectordb_dir.iterdir())

    # 判斷需要執行哪些步驟
    pending_steps = []
    if not has_collected:
        pending_steps.append("收集衛教文件")
    if not has_processed:
        pending_steps.append("前處理文件")
    if not has_index:
        pending_steps.append("建立向量索引")

    if not pending_steps:
        console.print("\n[green]✅ 資料收集、前處理、向量索引皆已完成[/green]")
        console.print(Panel(
            "  uv run src/cli.py chat             # 互動對話\n"
            "  uv run src/cli.py query \"問題\"      # 單次查詢\n"
            "  uv run src/cli.py serve             # 啟動 Web UI",
            title="[bold green]可以開始使用[/bold green]",
            border_style="green",
        ))
        raise typer.Exit()

    console.print(f"\n[yellow]尚未完成的步驟：{', '.join(pending_steps)}[/yellow]")
    run_pipeline = typer.confirm("是否立即執行？", default=True)

    if not run_pipeline:
        console.print(Panel(
            "1. uv run src/cli.py collect         # 收集衛教文件\n"
            "2. uv run src/cli.py preprocess      # 前處理文件\n"
            "3. uv run src/cli.py index           # 建立向量索引\n"
            "4. uv run src/cli.py chat            # 開始對話！",
            title="[bold green]後續步驟[/bold green]",
            border_style="green",
        ))
        raise typer.Exit()

    step = 1
    total = len(pending_steps)

    if not has_collected:
        console.print(f"\n[bold blue]Step {step}/{total} — 收集衛教文件[/bold blue]")
        from src.data import DataCollectionPipeline
        llm = get_llm()
        pipeline = DataCollectionPipeline(
            llm=llm,
            output_dir=collected_dir,
            pdf_cache_dir=Path("./data/raw/pdf_cache"),
        )
        docs = pipeline.run()
        console.print(f"[green]✅ 收集完成，共 {len(docs)} 份文件[/green]\n")
        step += 1

    if not has_processed:
        console.print(f"[bold blue]Step {step}/{total} — 前處理文件[/bold blue]")
        from src.data import Preprocessor
        preprocessor = Preprocessor()
        results = preprocessor.process_directory(collected_dir, processed_dir)
        console.print(f"[green]✅ 前處理完成，{len(results)} 份文件[/green]\n")
        step += 1

    if not has_index:
        console.print(f"[bold blue]Step {step}/{total} — 建立向量索引[/bold blue]")
        from src.rag_engine import DocumentIndexer
        indexer = DocumentIndexer(
            embedding_model=embedding,
            persist_directory=vectordb,
        )
        indexer.index_from_directory(processed_dir)
        console.print("[green]✅ 索引建立完成[/green]\n")

    console.print(Panel(
        "初始化完成！您可以開始使用：\n\n"
        "  uv run src/cli.py chat             # 互動對話\n"
        "  uv run src/cli.py query \"問題\"      # 單次查詢\n"
        "  uv run src/cli.py serve             # 啟動 Web UI",
        title="[bold green]🎉 設定完成[/bold green]",
        border_style="green",
    ))


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
