"""PDF 下載與文字提取"""

import re
import ssl
import urllib.request
from pathlib import Path

import pymupdf

from .models import Document


class PDFLoader:
    """下載 PDF 並提取純文字內容"""

    DEFAULT_HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    def download(self, url: str, save_path: Path, timeout: int = 30) -> None:
        req = urllib.request.Request(url, headers=self.DEFAULT_HEADERS)
        # 部分政府網站 SSL 憑證不完整，允許略過驗證
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            save_path.write_bytes(resp.read())

    def extract_text(self, pdf_path: Path, skip_pages: int = 0) -> str:
        """
        從 PDF 提取文字。
        skip_pages: 跳過開頭幾頁（封面、目錄等）
        """
        doc = pymupdf.open(str(pdf_path))
        pages_text = []

        for i, page in enumerate(doc):
            if i < skip_pages:
                continue
            text = page.get_text()
            cleaned = self._clean_text(text)
            if cleaned:
                pages_text.append(cleaned)

        doc.close()
        return "\n\n".join(pages_text)

    def _clean_text(self, text: str) -> str:
        """清理 PDF 提取的文字"""
        # 移除單獨的頁碼行（如 "- 3 -" 或純數字行）
        text = re.sub(r"^\s*[-–]\s*\d+\s*[-–]\s*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)

        # 合併因 PDF 換行導致的斷句（中文不需要空格連接）
        text = re.sub(r"(?<=[^\n。！？])\n(?=[^\n])", "", text)

        # 清理多餘空白
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
        return text.strip()

    def load(
        self,
        doc_id: str,
        title: str,
        url: str,
        source_name: str,
        cache_dir: Path,
        skip_pages: int = 0,
        language: str = "zh",
    ) -> Document | None:
        """下載 PDF 並回傳 Document（會快取到 cache_dir）"""
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = cache_dir / f"{doc_id}.pdf"

        try:
            if not pdf_path.exists():
                print(f"  下載 PDF：{title}...")
                self.download(url, pdf_path)
            else:
                print(f"  使用快取：{title}")

            content = self.extract_text(pdf_path, skip_pages=skip_pages)
            if not content:
                print(f"  ✗ 無法提取內容：{title}")
                return None

            print(f"  ✓ {title} ({len(content)} 字)")
            return Document(
                id=doc_id,
                title=title,
                content=content,
                source_url=url,
                source_name=source_name,
                language=language,
            )
        except Exception as e:
            print(f"  ✗ 錯誤 [{title}]: {e}")
            return None
