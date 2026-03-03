"""前處理器 - 清理 collected 文件並輸出一致格式"""

import re
from pathlib import Path

from .models import Document


# ── 各文件的清理規則 ──────────────────────────────────────────

def _remove_lines_matching(text: str, pattern: str) -> str:
    """移除符合 pattern 的整行"""
    return re.sub(rf"^.*{pattern}.*$\n?", "", text, flags=re.MULTILINE)


def _remove_toc(text: str) -> str:
    """移除目錄區塊（Q01-Q04 這類索引行）"""
    # 移除「Qxx-Qxx 標題」格式的目錄行
    text = re.sub(r"^Q\d+[-–]Q\d+.*$\n?", "", text, flags=re.MULTILINE)
    # 移除「Qxx 標題」格式的目錄行（只有一個問題的項目）
    text = re.sub(r"^Q\d+\s{2,}.*$\n?", "", text, flags=re.MULTILINE)
    # 移除「第X章 標題」目錄行
    text = re.sub(r"^第[一二三四五六七八九十]+章\s+\S+.*$\n?", "", text, flags=re.MULTILINE)
    return text


def _remove_page_headers(text: str, title: str) -> str:
    """移除 PDF 每頁重複的頁首標題"""
    escaped = re.escape(title)
    # 移除重複出現的標題（含空格變體）
    text = re.sub(rf"{escaped}[\s　]*{escaped}", "", text)
    text = re.sub(rf"^{escaped}\s*$\n?", "", text, flags=re.MULTILINE)
    return text


def _remove_date_strings(text: str) -> str:
    """移除更新日期、版權年份等"""
    text = re.sub(r"\d{4}/\d{1,2}/\d{1,2}\s*\d*:\d*:\d*", "", text)
    text = re.sub(r"最後更新[：:]\s*\n?\s*\d{4}/\d{1,2}/\d{1,2}.*", "", text)
    return text


def _remove_ui_artifacts(text: str) -> str:
    """移除網頁 UI 殘留字串"""
    # 語言切換連結（Español、English 等單行）
    text = re.sub(r"^\s*(Español|English|中文|繁體中文|简体中文)\s*$\n?", "", text, flags=re.MULTILINE)
    # 「了解更多...」類的連結文字
    text = re.sub(r"^了解更多.*$\n?", "", text, flags=re.MULTILINE)
    # Content 標記（PDF 目錄殘留）
    text = re.sub(r"^Content\s*$\n?", "", text, flags=re.MULTILINE)
    return text


def _normalize_whitespace(text: str) -> str:
    """統一空白與換行"""
    # 全形空格轉半形
    text = text.replace("\u3000", " ")
    # 多餘空格
    text = re.sub(r"[ \t]+", " ", text)
    # 超過兩個連續換行縮減為兩個
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 每行開頭結尾空白
    text = "\n".join(line.strip() for line in text.splitlines())
    return text.strip()


# ── 文件專屬規則 ─────────────────────────────────────────────

DOCUMENT_RULES: dict[str, list] = {
    "mohw_dementia_handbook": [
        lambda t: _remove_page_headers(t, "失智症衛教 及資源手冊"),
        lambda t: _remove_page_headers(t, "失智症衛教及資源手冊"),
        # 移除目錄行（含伍、附表的那行）
        lambda t: re.sub(r"^壹、.*目錄\s*$\n?", "", t, flags=re.MULTILINE | re.DOTALL),
        # 移除附表區塊（機構清單，不適合 RAG）—— 只切實際內容中的附表段落
        lambda t: re.split(r"\n【附表1】衛生福利部", t)[0],
    ],
    "ntpc_100qa": [
        _remove_toc,
        # 移除目錄的頁碼行（如「06」「綜論 認識失智症」）
        lambda t: re.sub(r"^\d{2,3}$\n?", "", t, flags=re.MULTILINE),
    ],
    "nia_faq_en": [
        _remove_ui_artifacts,
    ],
    "cmuh_dementia": [
        _remove_date_strings,
    ],
}

# 所有文件都套用的通用規則（依序執行）
COMMON_RULES = [
    _remove_date_strings,
    _remove_ui_artifacts,
    _normalize_whitespace,
]


# ── 主要前處理器 ──────────────────────────────────────────────

class Preprocessor:
    """清理 collected 文件，輸出格式一致的 processed 文件"""

    def process_file(self, txt_path: Path) -> tuple[str, str]:
        """
        處理單一 txt 檔案。
        回傳 (doc_id, cleaned_content)
        """
        raw = txt_path.read_text(encoding="utf-8")
        doc_id = txt_path.stem

        # 分離 header（前四行）與 body
        lines = raw.split("\n")
        header_lines = lines[:5]   # # title \n\n 來源 \n 網址 \n\n ---
        body = "\n".join(lines[6:]) if len(lines) > 6 else ""

        # 套用專屬規則
        for rule in DOCUMENT_RULES.get(doc_id, []):
            body = rule(body)

        # 套用通用規則
        for rule in COMMON_RULES:
            body = rule(body)

        cleaned = "\n".join(header_lines) + "\n\n" + body
        return doc_id, cleaned

    def process_directory(self, input_dir: Path, output_dir: Path) -> list[Path]:
        """處理整個目錄，輸出到 output_dir"""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        txt_files = sorted(input_dir.glob("*.txt"))
        print(f"前處理 {len(txt_files)} 份文件：{input_dir} → {output_dir}\n")

        for txt_file in txt_files:
            doc_id, cleaned = self.process_file(txt_file)
            out_path = output_dir / txt_file.name

            original_len = len(txt_file.read_text(encoding="utf-8"))
            cleaned_len = len(cleaned)
            removed = original_len - cleaned_len

            out_path.write_text(cleaned, encoding="utf-8")
            results.append(out_path)
            print(f"  ✓ {doc_id:<35} {original_len:>6} → {cleaned_len:>6} 字  (-{removed})")

        print(f"\n完成！{len(results)} 份文件 → {output_dir}")
        return results
