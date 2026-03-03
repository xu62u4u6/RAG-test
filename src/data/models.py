"""資料模型"""

from dataclasses import dataclass


@dataclass
class Document:
    """統一的文件格式"""
    id: str            # 唯一識別碼（用於檔名）
    title: str         # 文章標題
    content: str       # 純文字內容
    source_url: str    # 來源網址
    source_name: str   # 來源機構名稱
    language: str = "zh"  # 語言：zh / en

    def to_txt(self) -> str:
        """轉成與 vghtpe 相同的 txt 格式"""
        return (
            f"# {self.title}\n\n"
            f"來源: {self.source_name}\n"
            f"網址: {self.source_url}\n\n"
            f"---\n\n"
            f"{self.content}"
        )
