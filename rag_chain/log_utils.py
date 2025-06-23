# rag_chain/log_utils.py
from typing import Optional, List

import logging

def log_request_start(request_id: str, query: str, last_query: str):
    logging.info(f"\n====== 用户请求 {request_id} ======")
    logging.info(f"用户问题: {query}")
    logging.info(f"历史问题: {last_query if last_query else '（无）'}")


# def log_decision_agent(request_id: str, query: str, last_query: str, should_retrieve: bool, history: Optional[List[str]] = None):
#     logging.info(f"\n====== 检索判断 Agent - 请求 {request_id} ======")
#     logging.info(f"当前问题: {query}")
#     logging.info(f"上个问题: {last_query if last_query else '（无）'}")
#     logging.info(f"是否需要进行检索: {'是 ✅' if should_retrieve else '否 ❌'}")
#
#     if history:
#         formatted_history = "\n".join([f"{i+1}. {line}" for i, line in enumerate(history[-10:])])
#         logging.info(f"参考对话历史（最多5轮）:\n{formatted_history}")

def log_decision_agent(
    request_id: str,
    query: str,
    last_query: str,
    should_retrieve: bool,
    history: Optional[List[str]] = None,
    max_history_items: int = 5,
    max_item_length: int = 60
):
    logging.info(f"\n====== 检索判断 Agent - 请求 {request_id} ======")
    logging.info(f"当前问题: {query}")
    logging.info(f"上个问题: {last_query if last_query else '（无）'}")
    logging.info(f"是否需要进行检索: {'是 ✅' if should_retrieve else '否 ❌'}")

    if history:
        logging.info("最近历史对话（最多显示 %d 条，每条最多 %d 字）：" % (max_history_items, max_item_length))
        recent = history[-max_history_items:]
        for i, h in enumerate(recent):
            short = h[:max_item_length] + ("..." if len(h) > max_item_length else "")
            role = "系统" if i % 2 == 0 else "用户"
            logging.info(f"{role}: {short}")


def log_faq_hit(request_id: str, query: str, faq_answer: str):
    logging.info(f"\n====== FAQ 匹配命中 - 请求 {request_id} ======")
    logging.info(f"用户问题: {query}")
    logging.info("命中 FAQ 问题，直接返回答案 ✅")
    logging.info(f"FAQ 回答: {faq_answer}")

def log_cache_usage(request_id: str):
    logging.info(f"\n====== 使用缓存文档回答（未进行检索）{request_id} ======")
    logging.info("基于上一次检索结果生成回答，无需重复检索。")

def log_rag_docs(request_id: str, retrieved_docs):
    logging.info(f"\n====== RAG 检索结果 - 请求 {request_id} ======")
    logging.info(f"检索文档数量: {len(retrieved_docs)}")
    if not retrieved_docs:
        logging.info("⚠️ 未检索到任何文档")
        return
    for i, doc in enumerate(retrieved_docs):
        title = doc.metadata.get("title", "未命名政策")
        basis = doc.metadata.get("policy_basis", "未知政策依据")
        snippet = doc.page_content[:150].replace("\n", "")
        logging.info(f"[{i+1}] 《{title}》（依据：{basis}）摘要：{snippet}...")

def log_final_answer(request_id: str, answer: str):
    logging.info(f"📤 最终回答 - 请求 {request_id}")
    logging.info(f"LLM 回答内容:\n{answer}")
