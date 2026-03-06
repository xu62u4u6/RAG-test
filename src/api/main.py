"""
FastAPI 應用程式主程式與簡易 Web 介面
"""

import os
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.router import router
from src.api.dependencies import init_pipeline_background


class TokenAuthMiddleware(BaseHTTPMiddleware):
    """簡易 token 驗證，從 URL query param ?token=xxx 驗證"""

    async def dispatch(self, request: Request, call_next):
        api_secret = os.getenv("API_SECRET")
        if not api_secret:
            return await call_next(request)

        token = request.query_params.get("token")
        if token != api_secret:
            return JSONResponse(
                status_code=403,
                content={"detail": "請在網址加上 ?token=YOUR_TOKEN"},
            )

        return await call_next(request)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 啟動時在背景載入模型（不阻塞 server 啟動）
    asyncio.create_task(init_pipeline_background())
    yield


app = FastAPI(
    title="神經內科醫療 RAG 系統 API",
    description="提供門診問答系統的 HTTP API 服務",
    version="1.0.0",
    lifespan=lifespan,
)

# Token 驗證
app.add_middleware(TokenAuthMiddleware)

# 加入 CORS Middleware 確保可以從各種前端呼叫
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 註冊 API 路由
app.include_router(router)


# --- 簡易 Web 測試介面 ---

HTML_CONTENT = """
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>神經內科 RAG 問答系統</title>
    <!-- 引入 Tailwind CSS CDN -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- 引入 marked.js 渲染 Markdown -->
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
</head>
<body class="bg-gray-50 h-screen flex flex-col font-sans">
    
    <!-- 標題列 -->
    <header class="bg-blue-600 text-white shadow-md w-full sticky top-0 z-50">
        <div class="max-w-4xl mx-auto px-4 py-4 flex items-center justify-between">
            <h1 class="text-xl font-bold flex items-center gap-2">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                </svg>
                神經內科 RAG 詢問系統
            </h1>
            <span class="text-blue-100 text-sm">北榮衛教小助手</span>
        </div>
    </header>

    <!-- 聊天室內容 -->
    <main id="chat-container" class="flex-1 w-full max-w-4xl mx-auto p-4 flex flex-col gap-4 overflow-y-auto overflow-x-hidden md:py-8">
        <!-- 歡迎訊息 -->
        <div class="flex flex-col gap-1.5 w-full md:w-4/5 self-start">
            <span class="text-xs text-gray-500 font-semibold uppercase tracking-wider pl-1">AI 助理</span>
            <div class="bg-white px-5 py-4 rounded-2xl rounded-tl-sm shadow-sm border border-gray-100 text-gray-800 leading-relaxed text-left">
                您好！我是神經內科 RAG 衛教助理。您可以試著問我關於失智症、用藥、照顧技巧等問題！
            </div>
        </div>
    </main>

    <!-- 輸入區 -->
    <footer class="bg-white border-t border-gray-200 p-4 sticky bottom-0 z-50 shadow-[0_-4px_6px_-1px_rgba(0,0,0,0.05)] w-full">
        <div class="max-w-4xl mx-auto flex gap-3">
            <input 
                type="text" 
                id="question-input" 
                class="flex-1 border border-gray-300 rounded-xl px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-shadow text-gray-700 placeholder-gray-400" 
                placeholder="輸入您的問題... (例如：失智症有什麼常見症狀？)"
                autocomplete="off"
            >
            <button 
                id="send-btn"
                onclick="sendQuestion()" 
                class="bg-blue-600 hover:bg-blue-700 text-white font-medium py-3 px-6 rounded-xl transition-colors shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 flex items-center gap-2 group"
            >
                發送
                <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 transform group-hover:translate-x-1 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7m0 0l-7 7m7-7H3" />
                </svg>
            </button>
        </div>
    </footer>

    <script>
        const chatContainer = document.getElementById('chat-container');
        const questionInput = document.getElementById('question-input');
        const sendBtn = document.getElementById('send-btn');

        // 從 URL 取得 token，API 請求時自動帶上
        const urlToken = new URLSearchParams(window.location.search).get('token') || '';

        // 按 Enter 鍵送出
        questionInput.addEventListener('keypress', function (e) {
            if (e.key === 'Enter') {
                sendQuestion();
            }
        });

        // 加入 Markdown 樣式（給 marked.js 的輸出使用）
        const markdownStyles = `
            <style>
                .markdown-body { font-size: 15px; }
                .markdown-body p { margin-bottom: 0.75em; }
                .markdown-body p:last-child { margin-bottom: 0; }
                .markdown-body strong { font-weight: 600; color: #1f2937; }
                .markdown-body ul { list-style-type: disc; padding-left: 1.5em; margin-bottom: 0.75em; }
                .markdown-body ol { list-style-type: decimal; padding-left: 1.5em; margin-bottom: 0.75em; }
                .markdown-body li { margin-bottom: 0.25em; }
            </style>
        `;
        document.head.insertAdjacentHTML('beforeend', markdownStyles);

        function scrollToBottom() {
            chatContainer.scrollTo({
                top: chatContainer.scrollHeight,
                behavior: 'smooth'
            });
        }

        function appendMessage(role, text, sources = null) {
            const wrapper = document.createElement('div');
            wrapper.className = `flex flex-col gap-1 w-full md:w-5/6 ${role === 'user' ? 'self-end items-end' : 'self-start items-start'} mb-4`;
            
            // 加入發話者標籤
            const label = document.createElement('span');
            label.className = `text-xs font-semibold uppercase tracking-wider px-1 ${role === 'user' ? 'text-blue-500' : 'text-gray-500'}`;
            label.textContent = role === 'user' ? '您' : 'AI 助理';
            wrapper.appendChild(label);

            // 加入訊息泡泡
            const bubbleDiv = document.createElement('div');
            
            if (role === 'user') {
                bubbleDiv.className = "bg-blue-600 text-white px-5 py-3.5 rounded-2xl rounded-tr-sm shadow-sm leading-relaxed text-left break-words";
                bubbleDiv.textContent = text;
            } else {
                bubbleDiv.className = "bg-white px-5 py-4 rounded-2xl rounded-tl-sm shadow-sm border border-gray-100 text-gray-800 leading-relaxed text-left break-words w-full overflow-hidden";
                const innerBody = document.createElement('div');
                innerBody.className = "markdown-body";
                // 如果是特殊狀態訊息(如正在思考)
                if (text === '...') {
                    innerBody.innerHTML = '<div class="flex space-x-1.5 h-6 items-center px-1"><div class="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce"></div><div class="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce" style="animation-delay: 0.15s"></div><div class="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce" style="animation-delay: 0.3s"></div></div>';
                } else {
                     // 解析 markdown
                     innerBody.innerHTML = marked.parse(text);
                }
                bubbleDiv.appendChild(innerBody);
            }
            
            wrapper.appendChild(bubbleDiv);

            // 如果有來源文件，加入小字註記清單
            if (sources && sources.length > 0) {
                const sourceDiv = document.createElement('div');
                sourceDiv.className = "mt-2 bg-blue-50/50 rounded-xl p-3 border border-blue-100 text-sm text-gray-600 w-full";
                
                let sourceHtml = '<p class="font-medium text-blue-800 mb-1 flex items-center gap-1.5"><svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" /></svg>參考資料：</p><ul class="space-y-1.5">';
                sources.forEach((src, idx) => {
                    const simScore = Math.round(src.score * 100);
                    // 提供懸浮提示(Tooltip)來顯示部分內容預覽
                    const titleHtml = src.url
                        ? `<a href="${src.url}" target="_blank" rel="noopener" class="font-medium text-blue-600 hover:text-blue-800 underline underline-offset-2 transition-colors">${src.title}</a>`
                        : `<span class="font-medium text-gray-700">${src.title}</span>`;
                    sourceHtml += `<li class="flex items-start gap-2 group cursor-help" title="${src.content_preview.replace(/"/g, '&quot;')}">
                        <span class="inline-flex items-center justify-center bg-blue-100 text-blue-700 rounded-full h-5 w-5 text-xs font-bold shrink-0">${idx + 1}</span>
                        <div class="leading-tight">
                            ${titleHtml}
                            <span class="text-xs text-gray-400 ml-1">(${simScore}% 相關)</span>
                        </div>
                    </li>`;
                });
                sourceHtml += '</ul>';
                sourceDiv.innerHTML = sourceHtml;
                wrapper.appendChild(sourceDiv);
            }

            chatContainer.appendChild(wrapper);
            scrollToBottom();
            return wrapper; // 回傳 wrapper 以便後續更新
        }

        async function sendQuestion() {
            const question = questionInput.value.trim();
            if (!question) return;

            // 清空輸入並禁用按鈕
            questionInput.value = '';
            questionInput.disabled = true;
            sendBtn.disabled = true;
            sendBtn.classList.add('opacity-70', 'cursor-not-allowed');

            // 顯示使用者問題
            appendMessage('user', question);

            // 顯示載入中的動畫泡泡
            const loadingBubble = appendMessage('assistant', '...');

            try {
                // 發送 API 請求
                const response = await fetch('/api/chat?token=' + encodeURIComponent(urlToken), {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question: question, top_k: 5 })
                });

                const data = await response.json();

                // 移除載入中的泡泡
                chatContainer.removeChild(loadingBubble);

                if (response.ok) {
                    appendMessage('assistant', data.answer, data.sources);
                } else {
                    appendMessage('assistant', `⚠️ 發生錯誤：${data.detail || '請稍後再試'}`);
                }
            } catch (error) {
                chatContainer.removeChild(loadingBubble);
                appendMessage('assistant', `⚠️ 網路連線錯誤：請檢查伺服器狀態。`);
                console.error('Error:', error);
            } finally {
                // 恢復輸入列狀態
                questionInput.disabled = false;
                sendBtn.disabled = false;
                sendBtn.classList.remove('opacity-70', 'cursor-not-allowed');
                questionInput.focus();
            }
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def get_index():
    """
    回傳簡易聊天室 UI，用於快速測試
    """
    return HTML_CONTENT
