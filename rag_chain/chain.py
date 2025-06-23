# chain.py


import os
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from .retriever_config import build_retriever
from langchain_deepseek import ChatDeepSeek

# ✅ 加载 retriever
retriever = build_retriever(
    md_path="惠企政策_去除<br>.md",
    summary_path="summaries_2.json"
)

# ✅ DeepSeek 配置：从环境变量加载 key
llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=None,
    api_key = "sk-524bc7e636d34a6a8c102be54acbc16a"
    # api_key=os.getenv("DEEPSEEK_API_KEY")
)

# ========== 工具函数 ==========
def format_enterprise_context(info: dict) -> str:
    return (
        f"企业背景信息如下：该企业属于{info.get('region', '未知地区')}的{info.get('industry', '未知行业')}，"
        f"成立已有{info.get('years', '未知年限')}，目前员工约{info.get('employees', '未知人数')}，"
        f"注册资本为{info.get('capital', '未知资本')}。\n"
    )

def get_missing_fields_prompt(info: dict) -> str:
    required_fields = {
        "industry": "所属行业",
        "region": "注册地区",
        "years": "成立年限",
        "employees": "员工规模",
        "capital": "注册资本"
    }
    missing = [v for k, v in required_fields.items() if not info.get(k)]
    return "为了更准确推荐政策，您可以补充以下信息：" + "、".join(missing) if missing else ""

def build_background_aware_query(x):
    info = x.get("enterprise_info", {})
    query = x["question"]
    context = format_enterprise_context(info) if info else ""
    prompt = get_missing_fields_prompt(info)
    return context + query + ("\n" + prompt if prompt else "")

def format_docs(docs):
    return "\n\n".join(
        f"《{doc.metadata.get('title', '未命名政策')}》：\n{doc.page_content[:500]}" for doc in docs
    )

# ========== Prompt 模板 ==========
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "你是一个基于RAG架构实现的服务于企业用户的惠企政策问答助手，你将会看到用户的问题和检索到的政策文件。请严格遵循以下规范进行回答：\n\n"
     "1. 判断用户输入是否为有效提问。若为“你好”“1”等无意义内容，请简洁回复，不要生成政策解读。\n"
     "2. 回答必须基于以下政策内容（context），禁止编造。若无法回答，请回复：“很抱歉，我未能在当前政策中找到相关内容。\n"
     "3. 你会与客户进行多轮对话，回答的时候注意与客户聊天的上下文”\n"
     "4. 回答应自然、简洁、适合网页展示。\n"
     "5. 请勿主动在回答中生成“参考政策”或“政策来源”等信息，系统会自动添加。\n\n"
     "以下是可用政策内容：\n{context}"
     ),
    ("placeholder", "{history}"),
    ("human", "{question}")
])

# ========== 构建链 ==========
# ① 基础输入清洗
prepare_inputs = RunnableParallel({
    "question": RunnableLambda(lambda x: x["question"]),
    "history": RunnableLambda(lambda x: x["history"]),
    "enterprise_info": RunnableLambda(lambda x: x.get("enterprise_info", {})),
})

# ② 检索 & 格式化
with_retrieval = prepare_inputs | RunnableLambda(lambda x: {
    **x,
    "retrieved_docs": retriever.invoke(x["question"]),
    "question_for_llm": build_background_aware_query(x)
})

# ③ 拼接最终 LLM 输入并调用模型
rag_chain = with_retrieval | RunnableParallel({
    "answer": RunnableLambda(lambda x: {
        "context": format_docs(x["retrieved_docs"]),
        "question": x["question_for_llm"],
        "history": x["history"]
    }) | prompt | llm | StrOutputParser(),
    "retrieved_docs": RunnableLambda(lambda x: x["retrieved_docs"])
})
