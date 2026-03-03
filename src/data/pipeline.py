"""資料收集統一入口 - 整合所有來源"""

from pathlib import Path

from langchain_core.language_models import BaseChatModel

from .models import Document
from .html_scraper import HTMLScraper
from .pdf_loader import PDFLoader
from .translator import Translator


# ── 資料來源設定 ──────────────────────────────────────────────

HTML_SOURCES = [
    # 臺北榮總失智治療及研究中心
    {
        "id": "vghtpe_drug",
        "title": "失智症藥物",
        "url": "https://wd.vghtpe.gov.tw/dtrc/Fpage.action?muid=10561&fid=10194",
        "source_name": "臺北榮民總醫院失智治療及研究中心",
        "content_marker": 'class="table-responsive ckeditor"',
        "language": "zh",
    },
    {
        "id": "vghtpe_bpsd",
        "title": "失智症行為與情緒問題BPSD",
        "url": "https://wd.vghtpe.gov.tw/dtrc/Fpage.action?muid=10562&fid=10195",
        "source_name": "臺北榮民總醫院失智治療及研究中心",
        "content_marker": 'class="table-responsive ckeditor"',
        "language": "zh",
    },
    {
        "id": "vghtpe_sleep",
        "title": "失智症患者睡眠問題",
        "url": "https://wd.vghtpe.gov.tw/dtrc/Fpage.action?muid=18143&fid=16286",
        "source_name": "臺北榮民總醫院失智治療及研究中心",
        "content_marker": 'class="table-responsive ckeditor"',
        "language": "zh",
    },
    {
        "id": "vghtpe_eating",
        "title": "失智者的進食問題",
        "url": "https://wd.vghtpe.gov.tw/dtrc/Fpage.action?muid=18144&fid=16287",
        "source_name": "臺北榮民總醫院失智治療及研究中心",
        "content_marker": 'class="table-responsive ckeditor"',
        "language": "zh",
    },
    {
        "id": "vghtpe_feeding_knowledge",
        "title": "手握手進食技巧-知識篇",
        "url": "https://wd.vghtpe.gov.tw/dtrc/Fpage.action?muid=18994&fid=17013",
        "source_name": "臺北榮民總醫院失智治療及研究中心",
        "content_marker": 'class="table-responsive ckeditor"',
        "language": "zh",
    },
    {
        "id": "vghtpe_feeding_skill",
        "title": "手握手進食技巧-技巧篇",
        "url": "https://wd.vghtpe.gov.tw/dtrc/Fpage.action?muid=18995&fid=17014",
        "source_name": "臺北榮民總醫院失智治療及研究中心",
        "content_marker": 'class="table-responsive ckeditor"',
        "language": "zh",
    },
    {
        "id": "vghtpe_wandering",
        "title": "失智症患者遊走問題",
        "url": "https://wd.vghtpe.gov.tw/dtrc/Fpage.action?muid=18142&fid=16284",
        "source_name": "臺北榮民總醫院失智治療及研究中心",
        "content_marker": 'class="table-responsive ckeditor"',
        "language": "zh",
    },
    {
        "id": "vghtpe_caregiver",
        "title": "照顧者壓力與支持",
        "url": "https://wd.vghtpe.gov.tw/dtrc/Fpage.action?muid=10564&fid=10196",
        "source_name": "臺北榮民總醫院失智治療及研究中心",
        "content_marker": 'class="table-responsive ckeditor"',
        "language": "zh",
    },
    # NIA 繁體中文
    {
        "id": "nia_factsheet_zh",
        "title": "阿茲海默病簡明資料",
        "url": "https://www.nia.nih.gov/health/alzheimers-and-dementia/alzheimers-disease-fact-sheet-traditional-chinese",
        "source_name": "美國國家老齡研究所（NIA/NIH）",
        "content_marker": 'class="content_main content_full col_left col-9-left"',
        "language": "zh",
    },
    # NIA 英文 FAQ（需要翻譯）
    {
        "id": "nia_faq_en",
        "title": "Frequently Asked Questions About Alzheimer's Disease",
        "url": "https://www.nia.nih.gov/health/alzheimers-and-dementia/frequently-asked-questions-about-alzheimers-disease",
        "source_name": "美國國家老齡研究所（NIA/NIH）",
        "content_marker": 'class="content_main content_full col_left col-9-left"',
        "language": "en",
    },
    # 中國醫大
    {
        "id": "cmuh_dementia",
        "title": "認識失智症",
        "url": "https://www.cmuh.cmu.edu.tw/HealthEdus/Detail?no=8002",
        "source_name": "中國醫藥大學附設醫院",
        "content_marker": 'class="main_content healthedu_detail_content news_detail_content size"',
        "language": "zh",
    },
]

PDF_SOURCES = [
    {
        "id": "mohw_dementia_handbook",
        "title": "失智症衛教及資源手冊",
        "url": "https://www-ws.pthg.gov.tw/Upload/2015pthg/35/relfile/7616/38967/13ce8c14-d379-43ff-b167-afb62aa1d393.pdf",
        "source_name": "衛生福利部",
        "skip_pages": 2,   # 跳過封面與目錄
        "language": "zh",
    },
    {
        "id": "ntpc_100qa",
        "title": "失智100問",
        "url": "https://www.careyou.ntpc.gov.tw/WebUPD/missing/resources/%E5%A4%B1%E6%99%BA100%E5%95%8F.pdf",
        "source_name": "新北市政府",
        "skip_pages": 1,
        "language": "zh",
    },
]


# ── Pipeline ──────────────────────────────────────────────────

class DataCollectionPipeline:
    """統一資料收集流程"""

    def __init__(self, llm: BaseChatModel, output_dir: Path, pdf_cache_dir: Path):
        self.output_dir = Path(output_dir)
        self.pdf_cache_dir = Path(pdf_cache_dir)
        self.scraper = HTMLScraper()
        self.pdf_loader = PDFLoader()
        self.translator = Translator(llm)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_cache_dir.mkdir(parents=True, exist_ok=True)

    def _save(self, doc: Document) -> None:
        path = self.output_dir / f"{doc.id}.txt"
        path.write_text(doc.to_txt(), encoding="utf-8")

    def run(self) -> list[Document]:
        documents = []

        # HTML 來源
        print("=" * 50)
        print("抓取 HTML 來源")
        print("=" * 50)
        for src in HTML_SOURCES:
            doc = self.scraper.scrape(
                doc_id=src["id"],
                title=src["title"],
                url=src["url"],
                source_name=src["source_name"],
                content_marker=src["content_marker"],
                language=src["language"],
            )
            if doc:
                if doc.language == "en":
                    doc = self.translator.translate_document(doc)
                self._save(doc)
                documents.append(doc)

        # PDF 來源
        print("\n" + "=" * 50)
        print("載入 PDF 來源")
        print("=" * 50)
        for src in PDF_SOURCES:
            doc = self.pdf_loader.load(
                doc_id=src["id"],
                title=src["title"],
                url=src["url"],
                source_name=src["source_name"],
                cache_dir=self.pdf_cache_dir,
                skip_pages=src.get("skip_pages", 0),
                language=src["language"],
            )
            if doc:
                self._save(doc)
                documents.append(doc)

        print(f"\n完成！共收集 {len(documents)} 份文件 → {self.output_dir}")
        print("下一步：執行 preprocess 指令清理文件")
        return documents
