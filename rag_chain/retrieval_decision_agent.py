from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain_deepseek import ChatDeepSeek

llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=None,
    api_key="sk-524bc7e636d34a6a8c102be54acbc16a"
)

# ✅ 判断是否需要检索的提示词（优化版本）
prompt = ChatPromptTemplate.from_messages([
    ("system",
    """你是一个用于判断是否需要重新进行知识库检索的助手。

请严格遵循以下判断逻辑，仅输出“需要”或“不需要”两个字：

1. 如果当前问题是对上一个问题中某个内容的进一步追问（如“请讲第一个政策”“它怎么申请”“补贴是多少”“还有其他吗”等指代性提问），则回答：不需要。
2. 如果当前问题引入了新实体、新话题（如“关于高校毕业生的补贴政策”“还有哪些适用于武汉地区的政策”等），则回答：需要。
3. 如果当前问题是寒暄、无效输入（如“你好”“明白了”“再说一遍”），则回答：不需要。

注意：
- 如果你能根据历史对话看出当前问题是接着上一个内容问的，请判断为“不需要”。
- 请不要进行任何解释或添加其他内容，结果必须是：需要 或 不需要。
"""),
    ("human",
     "最近对话记录（最多5轮）：\n{chat_history}\n\n上一个问题：{last_query}\n当前问题：{query}")
])

chain: Runnable = prompt | llm | StrOutputParser()

def should_retrieve(query: str, last_query: str, history: list[str]) -> bool:
    # 只保留最近5轮有效对话（10条消息），过滤掉非 str
    clean_history = [msg for msg in history if isinstance(msg, str)]
    last_turns = clean_history[-10:]

    # 将成对（用户-助手）组成对话
    pairs = [
        (last_turns[i], last_turns[i + 1])
        for i in range(0, len(last_turns) - 1, 2)
    ]

    chat_history = "\n".join([f"用户：{q}\n助手：{a}" for q, a in pairs])

    result = chain.invoke({
        "query": query,
        "last_query": last_query,
        "chat_history": chat_history
    })

    cleaned = result.strip().replace("。", "")
    return cleaned == "需要"
