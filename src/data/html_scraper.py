"""通用 HTML 爬蟲 - 抽象化自 scrape_vghtpe.py"""

import re
import urllib.request

from .models import Document


class HTMLScraper:
    """通用網頁爬蟲，支援自訂內容區塊選擇器"""

    DEFAULT_HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    def fetch_html(self, url: str, timeout: int = 15) -> str:
        req = urllib.request.Request(url, headers=self.DEFAULT_HEADERS)
        response = urllib.request.urlopen(req, timeout=timeout)
        raw = response.read()
        # 嘗試偵測編碼
        for encoding in ("utf-8", "big5", "gbk"):
            try:
                return raw.decode(encoding)
            except UnicodeDecodeError:
                continue
        return raw.decode("utf-8", errors="replace")

    def extract_text(self, html: str, content_marker: str) -> str:
        """
        從 HTML 中提取主文內容。
        content_marker: 定位主文區塊的字串（如 class="ckeditor"）
        """
        idx = html.find(content_marker)
        if idx == -1:
            return ""

        start = html.find(">", idx) + 1

        # 找結束位置：嘗試常見的結束標記
        end = -1
        for marker in [
            "</div>\n\t\t            \t</div>",
            "</div>\r\n\t\t            \t</div>",
            "</article>",
            "</section>",
            "</main>",
        ]:
            end = html.find(marker, start)
            if end != -1:
                break

        if end == -1:
            end = start + 15000

        return self._clean_html(html[start:end])

    def extract_text_between(self, html: str, start_marker: str, end_marker: str) -> str:
        """提取兩個標記之間的內容（適合結構較簡單的頁面）"""
        start_idx = html.find(start_marker)
        if start_idx == -1:
            return ""
        start_idx += len(start_marker)

        end_idx = html.find(end_marker, start_idx)
        if end_idx == -1:
            end_idx = start_idx + 15000

        return self._clean_html(html[start_idx:end_idx])

    def _clean_html(self, html: str) -> str:
        """將 HTML 轉為純文字"""
        # 移除 script / style
        html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)

        # 轉換區塊標籤為換行
        html = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
        html = re.sub(r"</p>|</div>|</h[1-6]>|</li>|</tr>", "\n\n", html, flags=re.IGNORECASE)

        # 移除所有剩餘 HTML 標籤
        html = re.sub(r"<[^>]+>", "", html)

        # 還原 HTML entities
        for entity, char in [
            ("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"),
            ("&gt;", ">"), ("&quot;", '"'), ("&#39;", "'"),
        ]:
            html = html.replace(entity, char)

        # 清理多餘空白與換行
        html = re.sub(r"[ \t]+", " ", html)
        html = re.sub(r"\n\s*\n\s*\n+", "\n\n", html)
        return html.strip()

    def scrape(
        self,
        doc_id: str,
        title: str,
        url: str,
        source_name: str,
        content_marker: str,
        language: str = "zh",
    ) -> Document | None:
        """爬取單一網頁，回傳 Document"""
        try:
            html = self.fetch_html(url)
            content = self.extract_text(html, content_marker)
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
