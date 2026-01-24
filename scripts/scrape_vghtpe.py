"""
北榮失智治療及研究中心 衛教專欄爬蟲
來源: https://wd.vghtpe.gov.tw/dtrc/Fpage.action?muid=7
"""

import re
import json
import urllib.request
from pathlib import Path
from dataclasses import dataclass, asdict


BASE_URL = "https://wd.vghtpe.gov.tw/dtrc/"

# 衛教專欄文章列表
ARTICLES = [
    {"id": "drug", "name": "失智症藥物", "path": "Fpage.action?muid=10561&fid=10194"},
    {"id": "bpsd", "name": "失智症行為與情緒問題BPSD", "path": "Fpage.action?muid=10562&fid=10195"},
    {"id": "sleep", "name": "失智症患者睡眠問題", "path": "Fpage.action?muid=18143&fid=16286"},
    {"id": "eating", "name": "失智者的進食問題", "path": "Fpage.action?muid=18144&fid=16287"},
    {"id": "feeding_knowledge", "name": "手握手進食技巧-知識篇", "path": "Fpage.action?muid=18994&fid=17013"},
    {"id": "feeding_skill", "name": "手握手進食技巧-技巧篇", "path": "Fpage.action?muid=18995&fid=17014"},
    {"id": "wandering", "name": "失智症患者遊走問題", "path": "Fpage.action?muid=18142&fid=16284"},
    {"id": "caregiver", "name": "照顧者壓力與支持", "path": "Fpage.action?muid=10564&fid=10196"},
]


@dataclass
class Article:
    id: str
    title: str
    url: str
    content: str
    source: str = "臺北榮民總醫院失智治療及研究中心"


def fetch_page(url: str) -> str:
    """抓取網頁內容"""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    )
    response = urllib.request.urlopen(req, timeout=15)
    return response.read().decode("utf-8")


def extract_content(html: str) -> str:
    """從 HTML 中提取主要內容"""
    # 找 ckeditor 區塊
    idx = html.find('class="table-responsive ckeditor"')
    if idx == -1:
        return ""

    # 找這個 div 的結束位置 (往後找 </div> 直到 </section>)
    start = html.find(">", idx) + 1

    # 找結束標記
    end_markers = ["</div>\n\t\t            \t</div>", "</div>\r\n\t\t            \t</div>"]
    end = -1
    for marker in end_markers:
        end = html.find(marker, start)
        if end != -1:
            break

    if end == -1:
        # fallback: 找下一個大區塊
        end = html.find("</section>", start)
        if end == -1:
            end = start + 10000

    content = html[start:end]

    # 清理 HTML
    # 移除 script 和 style
    content = re.sub(r"<script[^>]*>.*?</script>", "", content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r"<style[^>]*>.*?</style>", "", content, flags=re.DOTALL | re.IGNORECASE)

    # 轉換換行
    content = re.sub(r"<br\s*/?>", "\n", content, flags=re.IGNORECASE)
    content = re.sub(r"</p>", "\n\n", content, flags=re.IGNORECASE)
    content = re.sub(r"</h[1-6]>", "\n\n", content, flags=re.IGNORECASE)
    content = re.sub(r"</li>", "\n", content, flags=re.IGNORECASE)
    content = re.sub(r"</tr>", "\n", content, flags=re.IGNORECASE)

    # 移除所有其他 HTML 標籤
    content = re.sub(r"<[^>]+>", "", content)

    # 清理 HTML entities
    content = content.replace("&nbsp;", " ")
    content = content.replace("&amp;", "&")
    content = content.replace("&lt;", "<")
    content = content.replace("&gt;", ">")
    content = content.replace("&quot;", '"')

    # 清理多餘空白
    content = re.sub(r"[ \t]+", " ", content)
    content = re.sub(r"\n\s*\n\s*\n+", "\n\n", content)
    content = content.strip()

    return content


def scrape_all(output_dir: Path) -> list[Article]:
    """爬取所有文章"""
    articles = []

    for article_info in ARTICLES:
        url = BASE_URL + article_info["path"]
        print(f"正在抓取: {article_info['name']}...")

        try:
            html = fetch_page(url)
            content = extract_content(html)

            if content:
                article = Article(
                    id=article_info["id"],
                    title=article_info["name"],
                    url=url,
                    content=content
                )
                articles.append(article)

                # 儲存為單獨的 txt 檔案
                txt_path = output_dir / f"{article_info['id']}.txt"
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(f"# {article.title}\n\n")
                    f.write(f"來源: {article.source}\n")
                    f.write(f"網址: {article.url}\n\n")
                    f.write("---\n\n")
                    f.write(article.content)

                print(f"  ✓ 成功 ({len(content)} 字)")
            else:
                print(f"  ✗ 無法提取內容")

        except Exception as e:
            print(f"  ✗ 錯誤: {e}")

    return articles


def main():
    output_dir = Path(__file__).parent.parent / "data" / "raw" / "vghtpe"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("北榮失智治療及研究中心 衛教專欄爬蟲")
    print("=" * 50)

    articles = scrape_all(output_dir)

    # 儲存 JSON 索引
    index_path = output_dir / "index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump([asdict(a) for a in articles], f, ensure_ascii=False, indent=2)

    print("=" * 50)
    print(f"完成！共抓取 {len(articles)} 篇文章")
    print(f"儲存位置: {output_dir}")


if __name__ == "__main__":
    main()
