# rag_chain/faq_matcher.py

import json
import os
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

FAQ_FILE_PATH = "data/faq_knowledge.json"

# 初始化嵌入模型
embeddings_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-zh-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

def load_faq_data():
    if not os.path.exists(FAQ_FILE_PATH):
        return []

    with open(FAQ_FILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 始终重新生成所有 embedding
    questions = [item["question"] for item in data]
    embeddings = embeddings_model.embed_documents(questions)

    for item, emb in zip(data, embeddings):
        item["embedding"] = np.array(emb, dtype=np.float32)

    return data



def try_faq_match(query: str, threshold: float = 0.85):
    faq_data = load_faq_data()
    query_embedding = embeddings_model.embed_query(query)
    query_embedding = np.array(query_embedding, dtype=np.float32)

    def cosine_sim(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    best_match = None
    best_score = -1.0

    for item in faq_data:
        score = cosine_sim(query_embedding, item["embedding"])
        if score > best_score:
            best_match = item
            best_score = score

    if best_score >= threshold:
        return best_match["answer"]
    else:
        return None
