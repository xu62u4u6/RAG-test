"""
文本索引器 - 負責文本分塊、Embedding 和向量儲存
"""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class DocumentIndexer:
    """文本索引器"""

    def __init__(
        self,
        embedding_model: str = "BAAI/bge-large-zh-v1.5",
        persist_directory: str = "./data/vectordb",
        collection_name: str = "neuro_docs",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 初始化 Embedding 模型
        print(f"載入 Embedding 模型: {embedding_model}...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},  # 或 "cuda"
            encode_kwargs={"normalize_embeddings": True},
        )

        # 初始化文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
        )

        self.vectorstore = None

    def load_documents_from_directory(self, directory: Path) -> list[Document]:
        """從目錄載入所有 txt 檔案"""
        documents = []
        directory = Path(directory)

        for txt_file in directory.glob("*.txt"):
            with open(txt_file, "r", encoding="utf-8") as f:
                content = f.read()

            # 解析標題（第一行）
            lines = content.split("\n")
            title = lines[0].replace("# ", "").strip() if lines else txt_file.stem

            doc = Document(
                page_content=content,
                metadata={
                    "source": str(txt_file),
                    "title": title,
                    "filename": txt_file.name,
                },
            )
            documents.append(doc)
            print(f"  載入: {title}")

        return documents

    def index_documents(self, documents: list[Document]) -> Chroma:
        """將文件分塊並建立向量索引"""
        print(f"\n分割文本 (chunk_size={self.chunk_size})...")
        chunks = self.text_splitter.split_documents(documents)
        print(f"  共 {len(chunks)} 個文本區塊")

        print(f"\n建立向量索引...")
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=str(self.persist_directory),
            collection_name=self.collection_name,
        )

        print(f"  索引已儲存至: {self.persist_directory}")
        return self.vectorstore

    def load_vectorstore(self) -> Chroma | None:
        """載入已存在的向量索引，如果索引為空則返回 None"""
        self.vectorstore = Chroma(
            persist_directory=str(self.persist_directory),
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
        )
        # 檢查是否有資料
        count = self.vectorstore._collection.count()
        if count == 0:
            print("  警告：向量索引為空")
            return None
        print(f"  已載入 {count} 個向量")
        return self.vectorstore

    def get_document_count(self) -> int:
        """取得索引中的文件數量"""
        if self.vectorstore is None:
            return 0
        return self.vectorstore._collection.count()

    def index_from_directory(self, directory: Path) -> Chroma:
        """從目錄載入文件並建立索引"""
        print(f"從 {directory} 載入文件...")
        documents = self.load_documents_from_directory(directory)
        print(f"共載入 {len(documents)} 份文件")
        return self.index_documents(documents)
