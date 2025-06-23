import os
import traceback
import uuid
import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from rag_chain.retrieval_decision_agent import should_retrieve
from rag_chain.chain import rag_chain
from rag_chain.memory import build_messages_from_history
from rag_chain.faq_matcher import try_faq_match
from rag_chain.log_utils import (
    log_request_start, log_decision_agent,
    log_faq_hit, log_cache_usage,
    log_rag_docs, log_final_answer
)
from langchain_core.messages import HumanMessage
from langchain.callbacks import StdOutCallbackHandler

# ========== 缓存上一次的查询与结果 ==========
cached_last_query = None
cached_docs = []

# ========== 日志配置 ==========
os.makedirs("logs", exist_ok=True)
log_path = os.path.join("logs", "rag_logs2.txt")

logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8"
)

# ========== FastAPI 初始化 ==========
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_path = os.path.join(os.getcwd(), "frontend")
print("前端静态文件目录:", frontend_path)
app.mount("/frontend", StaticFiles(directory=frontend_path, html=True), name="frontend")

# ========== 数据模型 ==========
class ChatRequest(BaseModel):
    query: str
    history: Optional[List[str]] = []
    enterprise_info: Optional[Dict] = None

class Source(BaseModel):
    excerpt: str
    title: Optional[str] = "未命名政策"
    policy_basis: Optional[str] = "未知政策依据"

class ChatResponse(BaseModel):
    answer: str
    history: List[str]
    sources: Optional[List[Source]] = []


@app.post("/api/policy-qa", response_model=ChatResponse)
async def policy_qa_endpoint(request: ChatRequest):
    request_id = str(uuid.uuid4())[:8]
    query = request.query.strip()
    last_query = request.history[-2] if len(request.history) >= 2 else ""

    log_request_start(request_id, query, last_query)

    # Step 1: 判断是否需要检索
    # should_do_retrieve = should_retrieve(query, last_query)
    should_do_retrieve = should_retrieve(query, last_query, request.history)

    # log_decision_agent(request_id, query, last_query, should_do_retrieve)
    log_decision_agent(request_id, query, last_query, should_do_retrieve, request.history)

    if not should_do_retrieve:
        log_cache_usage(request_id)

        if not cached_docs:
            # ❌ 无缓存文档，无法继续问答
            answer_text = "您好，我是一个专注于解答湖北省惠企政策的自动问答机器人，您可以向我提问任何和惠企政策相关的问题。"
            updated_history = request.history + [query, answer_text]
            return ChatResponse(answer=answer_text, history=updated_history, sources=[])

        # ✅ 有缓存文档，继续问答
        clean_history = [msg for msg in request.history if isinstance(msg, str)]
        history_messages = [HumanMessage(content=msg) for msg in clean_history]
        messages = build_messages_from_history(history_messages, query)

        input_data = {
            "question": query,
            "history": messages,
            "retrieved_docs": cached_docs
        }

        result = rag_chain.invoke(input_data, config={"verbose": True, "callbacks": [StdOutCallbackHandler()]})
        answer_text = result.get("answer", "")

        sources = [
            Source(
                excerpt=doc.page_content[:200],
                title=doc.metadata.get("title", "未命名政策"),
                policy_basis=doc.metadata.get("policy_basis", "未知政策依据")
            ) for doc in cached_docs
        ]

        updated_history = request.history + [query, answer_text]
        return ChatResponse(answer=answer_text, history=updated_history, sources=sources)

    # Step 2: FAQ 匹配
    faq_answer = try_faq_match(query)
    if faq_answer:
        updated_history = request.history + [query, faq_answer]
        log_faq_hit(request_id, query, faq_answer)
        return ChatResponse(answer=faq_answer, history=updated_history, sources=[])

    # Step 3: 构建历史消息
    clean_history = [msg for msg in request.history if isinstance(msg, str)]
    history_messages = [HumanMessage(content=msg) for msg in clean_history]
    messages = build_messages_from_history(history_messages, query)

    # Step 4: 调用 RAG 链
    try:
        input_data = {"question": query, "history": messages}
        if request.enterprise_info:
            input_data["enterprise_info"] = request.enterprise_info

        result = rag_chain.invoke(input_data, config={"verbose": True, "callbacks": [StdOutCallbackHandler()]})
        retrieved_docs = result.get("retrieved_docs", [])
        answer_text = result.get("answer", "")

        cached_docs.clear()
        cached_docs.extend(retrieved_docs)

        log_rag_docs(request_id, retrieved_docs)

        if retrieved_docs:
            reference_lines = ["\n\n参考政策："]
            for doc in retrieved_docs:
                title = doc.metadata.get("title", "未命名政策")
                basis = doc.metadata.get("policy_basis", "未知政策依据")
                reference_lines.append(f"- 《{title}》（依据：《{basis}》）")
            answer_text += "\n" + "\n".join(reference_lines)

        # log_final_answer(request_id, answer_text)

        log_final_answer(request_id, answer_text)

    except Exception as e:
        error_msg = f"❌ 请求 {request_id} 异常: {str(e)}"
        logging.error(error_msg)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="系统内部错误，请稍后重试。")

    # Step 5: 构建响应
    sources = [
        Source(
            excerpt=doc.page_content[:200],
            title=doc.metadata.get("title", "未命名政策"),
            policy_basis=doc.metadata.get("policy_basis", "未知政策依据")
        ) for doc in retrieved_docs
    ]

    updated_history = request.history + [query, answer_text]
    return ChatResponse(answer=answer_text, history=updated_history, sources=sources)

# ==========首页重定向==========

@app.get("/")
async def root():
    return RedirectResponse(url="/frontend/")

# ========== 本地运行 ==========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8001, reload=True)


