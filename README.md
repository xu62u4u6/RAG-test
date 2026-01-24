# 神經內科醫療 RAG 系統

> 榮民醫院神經內科門診病人問答系統

## 專案概述

本系統旨在建立一個能夠回答神經內科門診病人常見問題的 RAG（Retrieval-Augmented Generation）系統。系統包含兩大核心模組：**出題者**（模擬病人提問）與 **解題者**（RAG 回答系統），用於生成評估問題集並測試系統效能。

## 系統架構

```
┌─────────────────────────────────────────────────────────────────┐
│                        神經內科 RAG 系統                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   資料層      │    │   出題者      │    │   解題者      │      │
│  │  Data Layer  │───▶│  Generator   │    │  RAG Engine  │      │
│  └──────────────┘    └──────┬───────┘    └──────▲───────┘      │
│         │                   │                   │              │
│         │            ┌──────▼───────┐           │              │
│         │            │  病人變體轉換  │───────────┘              │
│         │            │  Persona     │                          │
│         │            │  Transformer │                          │
│         │            └──────────────┘                          │
│         │                   │                                  │
│         │            ┌──────▼───────┐                          │
│         └───────────▶│   評估模組    │                          │
│                      │  Evaluator   │                          │
│                      └──────────────┘                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 目錄結構

```
RAG-test/
├── README.md
├── requirements.txt
├── config/
│   ├── settings.yaml          # 全域設定
│   └── personas/              # 病人變體設定
│       ├── elderly_woman.yaml # 老奶奶變體
│       ├── veteran.yaml       # 老榮民變體
│       └── ...
│
├── src/
│   ├── data/                  # 資料層
│   │   ├── scrapers/          # 爬蟲模組
│   │   │   ├── base_scraper.py
│   │   │   ├── health_gov.py  # 衛福部
│   │   │   └── neuro_society.py # 神經學學會
│   │   ├── processors/        # 文本處理
│   │   │   ├── cleaner.py     # 清洗
│   │   │   ├── chunker.py     # 分塊
│   │   │   └── hospital_adapter.py # 醫院文本適配器
│   │   └── vectorstore/       # 向量儲存
│   │       ├── embedder.py    # Embedding
│   │       └── store.py       # 向量資料庫操作
│   │
│   ├── question_generator/    # 出題者模組
│   │   ├── core_generator.py  # 核心問題生成器
│   │   ├── persona_transformer.py  # 變體轉換器
│   │   ├── knowledge_mask.py  # 知識遮罩（限制能力）
│   │   └── personas/          # 各變體實作
│   │       ├── base_persona.py
│   │       ├── elderly_woman.py
│   │       └── veteran.py
│   │
│   ├── rag_engine/            # 解題者模組
│   │   ├── retriever.py       # 檢索器
│   │   ├── generator.py       # 生成器
│   │   └── pipeline.py        # RAG 管線
│   │
│   └── evaluator/             # 評估模組
│       ├── ai_reviewer.py     # AI 評審
│       ├── metrics.py         # 評估指標
│       └── report_generator.py # 報告生成
│
├── data/
│   ├── raw/                   # 原始資料
│   ├── processed/             # 處理後資料
│   ├── vectordb/              # 向量資料庫
│   └── questions/             # 生成的問題集
│       ├── seed_questions.json     # 種子問題
│       └── synthetic_questions.json # 合成問題集
│
├── outputs/
│   ├── evaluations/           # 評估結果
│   └── reports/               # 給醫院的報告
│
├── scripts/
│   ├── scrape.py              # 執行爬蟲
│   ├── generate_questions.py  # 生成問題集
│   ├── run_evaluation.py      # 執行評估
│   └── export_report.py       # 匯出報告
│
└── tests/
    └── ...
```

## 核心模組設計

### 1. 資料層 (Data Layer)

#### 資料來源
| 來源 | 說明 | 優先級 |
|------|------|--------|
| 衛福部網站 | 官方衛教資料 | 高 |
| 台灣神經學學會 | 專業指引、衛教文章 | 高 |
| 醫院提供文本 | 內部衛教單張、常見問答 | 高 |
| PubMed 中文摘要 | 最新研究（需篩選） | 中 |

#### 文本處理流程
```
原始文本 → 清洗 → 分塊 (Chunking) → Embedding → 向量資料庫
                    ↓
            Chunk 大小: 512 tokens
            重疊: 50 tokens
```

### 2. 出題者模組 (Question Generator)

#### 設計理念：知識遮罩 (Knowledge Masking)

核心概念：**不是讓 AI 扮演病人，而是限制 AI 的能力來模擬真實病人的認知水平**

```
┌─────────────────────────────────────────────────────────┐
│                    知識遮罩機制                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  專業知識層 ████████████████████████████ (AI 完整知識)   │
│                        ↓ 遮罩                           │
│  老榮民層   ████████████░░░░░░░░░░░░░░░ (部分知識)       │
│                        ↓ 遮罩                           │
│  老奶奶層   ████░░░░░░░░░░░░░░░░░░░░░░░ (基礎知識)       │
│                                                         │
│  ░ = 被遮罩的知識（不可使用的詞彙/概念）                   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

#### 病人變體定義

| 變體 | 知識程度 | 詞彙特徵 | 資訊來源 | 常見誤解 |
|------|----------|----------|----------|----------|
| 老奶奶 | 低 | 口語化、台語混用 | 電視、鄰居 | 把延緩當治癒 |
| 老榮民 | 中低 | 較正式、偶有軍旅用語 | 報紙、廣播 | 對藥物副作用過度擔憂 |
| 中年子女 | 中 | 會查網路但一知半解 | Google、LINE群組 | 混淆不同疾病 |
| 高知識長者 | 中高 | 能使用部分術語 | 會看英文資料 | 過度解讀研究結果 |

#### 問題生成流程

```
1. 種子問題生成
   知識庫文本 → LLM 提取關鍵概念 → 生成專業版問題

2. 變體轉換
   專業問題 → 知識遮罩 → 變體轉換器 → 病人版問題

3. 品質篩選
   生成問題 → 多樣性檢查 → 合理性驗證 → 最終問題集
```

#### 範例

**種子問題（專業版）：**
> Lecanemab 作為 anti-amyloid 單株抗體，其清除 β-amyloid 斑塊的機制為何？

**老奶奶版：**
> 醫生啊，我聽隔壁阿桑說美國有一種新的藥可以治失智，那個藥是在做什麼的？真的會好嗎？

**老榮民版：**
> 報紙寫說有新藥可以治療老人癡呆症，我想請問這個藥跟以前吃的有什麼不一樣？

### 3. 解題者模組 (RAG Engine)

#### 技術選型

| 組件 | 開發階段 | 部署階段 |
|------|----------|----------|
| Embedding | `BAAI/bge-large-zh-v1.5` | 同左（本地） |
| 向量資料庫 | ChromaDB | ChromaDB / Milvus |
| LLM | OpenAI GPT-4 / Claude | Llama 3 / Qwen2 (本地) |
| Framework | LangChain | LangChain |

#### RAG 流程

```
病人問題
    ↓
Query 改寫（處理口語化表達）
    ↓
向量檢索 (Top-K = 5)
    ↓
Re-ranking（可選）
    ↓
Prompt 組裝（含病人理解程度指引）
    ↓
LLM 生成回答
    ↓
回答後處理（確保用詞適當）
```

### 4. 評估模組 (Evaluator)

#### 評估維度

| 維度 | 說明 | 評估方式 |
|------|------|----------|
| 正確性 | 醫學知識是否正確 | AI + 醫師審核 |
| 完整性 | 是否回答了問題 | AI 評估 |
| 易懂性 | 病人能否理解 | AI + 醫師審核 |
| 安全性 | 是否有誤導風險 | 醫師審核 |
| 適切性 | 語氣是否恰當 | AI 評估 |

#### 評估流程

```
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│ RAG回答  │───▶│ AI初審  │───▶│ 標記風險 │───▶│醫師複審  │
└─────────┘    └─────────┘    └─────────┘    └─────────┘
                   │                              │
                   ▼                              ▼
              自動評分報告                    最終評估報告
              (內部使用)                     (提供給醫院)
```

#### 輸出格式

```json
{
  "question_id": "Q001",
  "question": "醫生啊，那個新的失智藥...",
  "persona": "elderly_woman",
  "answer": "這個藥叫做...",
  "evaluation": {
    "ai_scores": {
      "correctness": 0.85,
      "completeness": 0.90,
      "readability": 0.75
    },
    "ai_flags": ["需確認劑量說明"],
    "human_review": {
      "reviewer": "Dr. Chen",
      "approved": true,
      "comments": "建議加入副作用說明"
    }
  }
}
```

## 問題集規劃

### 規模設計

| 階段 | 數量 | 用途 |
|------|------|------|
| 種子集 | 30-50 題 | 人工精選，驗證生成品質 |
| 擴展集 | 150-200 題 | 自動生成，正式評估 |
| 完整集 | 300+ 題 | 視需求擴展 |

### 主題矩陣

| 主題類別 | 子主題 | 老奶奶 | 老榮民 | 中年子女 | 小計 |
|----------|--------|--------|--------|----------|------|
| **失智症** | 阿茲海默新藥 (Lecanemab等) | 5 | 5 | 5 | 15 |
| | 症狀與早期徵兆 | 5 | 5 | 5 | 15 |
| | 日常照護 | 5 | 5 | 5 | 15 |
| **帕金森氏症** | 藥物治療 | 5 | 5 | 5 | 15 |
| | 症狀管理 | 5 | 5 | 5 | 15 |
| **腦中風** | 預防與風險因子 | 5 | 5 | 5 | 15 |
| | 復健與預後 | 5 | 5 | 5 | 15 |
| **頭痛** | 偏頭痛 | 5 | 5 | 5 | 15 |
| | 緊張型頭痛 | 5 | 5 | 5 | 15 |
| **其他** | 眩暈/耳鳴 | 5 | 5 | 5 | 15 |
| | 手腳麻木 | 5 | 5 | 5 | 15 |
| | 睡眠障礙 | 5 | 5 | 5 | 15 |
| **合計** | | 60 | 60 | 60 | **180** |

> 此為初期目標，可依醫院需求調整主題優先級

### 問題類型分布

每個主題內的問題應涵蓋：
- **知識型** (40%)：「這個藥是做什麼的？」
- **決策型** (30%)：「我應該開始吃藥嗎？」
- **情境型** (30%)：「我媽最近一直忘記吃藥怎麼辦？」

---

## 技術架構

### 技術選型總覽

```
┌─────────────────────────────────────────────────────────────────┐
│                        技術堆疊                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  爬蟲層        httpx / BeautifulSoup / Playwright              │
│                                                                 │
│  文本處理      langchain-text-splitters                         │
│                                                                 │
│  Embedding    sentence-transformers (BGE-zh)                   │
│                                                                 │
│  向量資料庫    ChromaDB (開發) → Milvus (部署)                   │
│                                                                 │
│  LLM 調用     litellm (統一介面)                                │
│               ├─ 開發: OpenAI / Anthropic API                  │
│               └─ 部署: vLLM + Qwen2 / Llama3                   │
│                                                                 │
│  RAG 框架     LangChain / LlamaIndex (待討論)                   │
│                                                                 │
│  設定管理     Pydantic + YAML                                   │
│                                                                 │
│  CLI 介面     Typer                                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 各模組技術細節

#### 1. 資料層

| 功能 | 套件 | 說明 |
|------|------|------|
| HTTP 請求 | `httpx` | 異步支援、比 requests 現代 |
| HTML 解析 | `beautifulsoup4` + `lxml` | 經典組合 |
| 動態網頁 | `playwright` | 需要時才用，處理 JS 渲染 |
| 文本分塊 | `langchain-text-splitters` | RecursiveCharacterTextSplitter |
| PDF 解析 | `pymupdf` | 處理醫院 PDF 文件 |

#### 2. 向量儲存

| 功能 | 套件 | 說明 |
|------|------|------|
| Embedding | `sentence-transformers` | 載入 BGE 模型 |
| 模型 | `BAAI/bge-large-zh-v1.5` | 中文效果好，1024 維 |
| 向量 DB | `chromadb` | 開發用，輕量 |
| 生產 DB | `pymilvus` | 部署用，可擴展 |

#### 3. LLM 調用

| 功能 | 套件 | 說明 |
|------|------|------|
| 統一介面 | `litellm` | 一套 API 切換 OpenAI/Claude/本地 |
| 本地推理 | `vllm` | 高效本地 LLM 服務 |
| 備選 | `ollama` | 更簡單的本地方案 |

#### 4. RAG 框架

採用 **LangChain** 作為主要框架：
- 生態豐富、文件完整
- 內建多種 LLM 提供商支援
- 方便日後切換本地模型（langchain-ollama）

#### 5. 其他工具

| 功能 | 套件 | 說明 |
|------|------|------|
| 設定管理 | `pydantic-settings` | 型別安全的設定 |
| YAML 解析 | `pyyaml` | 設定檔格式 |
| CLI | `typer` | 現代 CLI 框架 |
| 日誌 | `loguru` | 比 logging 好用 |
| 測試 | `pytest` | 標準測試框架 |
| 環境變數 | `python-dotenv` | .env 檔案載入 |

---

## 開發階段規劃

### Phase 1: 基礎建設
- [ ] 設置開發環境
- [ ] 實作爬蟲模組（衛福部、神經學學會）
- [ ] 建立向量資料庫
- [ ] 基礎 RAG pipeline

### Phase 2: 出題者開發
- [ ] 核心問題生成器
- [ ] 知識遮罩機制
- [ ] 老奶奶、老榮民變體
- [ ] 問題集品質驗證

### Phase 3: 評估系統
- [ ] AI 評審模組
- [ ] 評估報告生成
- [ ] 醫院人員審核介面

### Phase 4: 優化與部署
- [ ] 本地 LLM 整合
- [ ] 效能優化
- [ ] 部署文件

## 快速開始

```bash
# 安裝依賴
pip install -r requirements.txt

# 設定環境變數
cp .env.example .env
# 編輯 .env 填入 API keys

# 執行爬蟲
python scripts/scrape.py

# 生成問題集
python scripts/generate_questions.py --persona elderly_woman --count 50

# 執行評估
python scripts/run_evaluation.py --questions data/questions/synthetic_questions.json
```

## 設定說明

### config/settings.yaml

```yaml
# 向量資料庫設定
vectorstore:
  type: chroma
  path: ./data/vectordb
  collection: neuro_docs

# Embedding 設定
embedding:
  model: BAAI/bge-large-zh-v1.5
  device: cuda  # 或 cpu

# LLM 設定
llm:
  provider: openai  # 或 local
  model: gpt-4
  temperature: 0.7

# 本地 LLM 設定（部署用）
local_llm:
  model_path: ./models/qwen2-7b
  quantization: 4bit
```

### config/personas/elderly_woman.yaml

```yaml
name: 老奶奶
description: 70-85歲女性長者，教育程度國小或不識字

knowledge_mask:
  allowed_terms:
    - 失智症
    - 老人痴呆
    - 忘記
    - 藥
  forbidden_terms:
    - amyloid
    - 單株抗體
    - 神經退化
    - biomarker

  max_vocabulary_level: basic  # basic, intermediate, advanced

speech_patterns:
  - 會用「啊」「喔」等語助詞
  - 可能夾雜台語
  - 會說「聽人家說」
  - 常用問句：「是不是...」「會不會...」

common_misconceptions:
  - 以為失智可以完全治好
  - 分不清失智和正常老化
  - 擔心藥物會傷身

information_sources:
  - 電視健康節目
  - 鄰居親友
  - 廟裡的人
```

## 注意事項

1. **醫療資訊免責聲明**：本系統產生的回答僅供參考，不能取代專業醫療建議
2. **資料隱私**：如使用醫院提供的文本，需確保已去識別化
3. **持續更新**：神經內科領域發展快速，知識庫需定期更新

## 貢獻指南

（待補充）

## 授權

（待確認）

---

*本專案為榮民醫院神經內科合作開發*
