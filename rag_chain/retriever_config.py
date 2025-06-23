import os
import json
import uuid
import re
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_chroma import Chroma
from langchain.storage import InMemoryByteStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_huggingface import HuggingFaceEmbeddings

# 创建 embedding 模型
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-zh-v1.5",
    # model_name=r"C:\Users\Administrator\Desktop\rag_project\rag_project\models\bge-large-zh-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# 提取 markdown 文段中的“政策名称”
def extract_policy_name(text: str) -> str:
    match = re.search(r"(政策名称|名称)[：:]\s*(.+)", text)
    return match.group(2).strip() if match else "未命名政策"

# 提取“政策依据”字段
def extract_policy_basis(text: str) -> str:
    match = re.search(r"(?:政策依据|依据)[^\n]{0,20}?《([^》]+)》", text)
    return f"《{match.group(1)}》" if match else "未知政策依据"

# 拆分 Markdown 文件并提取原始文档列表
def load_markdown_sections(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    headers = [("#", "Header 1"), ("##", "Header 2")]
    splitter = MarkdownHeaderTextSplitter(headers)
    docs = splitter.split_text(content)

    # 初步提取 metadata（兜底用）
    for doc in docs:
        section_text = doc.page_content
        doc.metadata["title"] = extract_policy_name(section_text)
        doc.metadata["policy_basis"] = extract_policy_basis(section_text)

    return docs

# 构建 RAG 检索器
def build_retriever(md_path: str, summary_path: str):
    docs = load_markdown_sections(md_path)

    with open(summary_path, 'r', encoding='utf-8') as f:
        summaries = json.load(f)

    vectorstore = Chroma(collection_name="summaries_2", embedding_function=embeddings)
    store = InMemoryByteStore()
    id_key = "doc_id"

    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        byte_store=store,
        id_key=id_key,
        search_kwargs={"k": 3}
    )

    doc_ids = [str(uuid.uuid4()) for _ in docs]

    # 构造摘要文档并绑定 metadata
    summary_docs = []
    for item in summaries:
        doc_id = doc_ids[item["doc_index"]]
        summary_docs.append(Document(
            page_content=item["summary"],
            metadata={
                id_key: doc_id,
                "title": item.get("title", "未命名政策"),
                "policy_basis": item.get("policy_basis", "未知政策依据")
            }
        ))

    # ✅ 同步 metadata 到原文 doc（retrieved_docs 实际返回用的是这个）
    for item in summaries:
        i = item["doc_index"]
        docs[i].metadata["title"] = item.get("title", "未命名政策")
        docs[i].metadata["policy_basis"] = item.get("policy_basis", "未知政策依据")

    retriever.vectorstore.add_documents(summary_docs)
    retriever.docstore.mset(list(zip(doc_ids, docs)))

    return retriever
