from rag_chain.chain import rag_chain
from rag_chain.memory import build_messages_from_history, update_history
from pprint import pprint

history = []

while True:
    query = input("\n👤 用户提问：")
    query = query.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")  # ✅ 清洗非法字符
    if query.lower() in ["exit", "quit"]:
        break

    messages = build_messages_from_history(history, query)
    print("\n🔍 即将传入 rag_chain 的数据：")
    print({"question": query, "history": messages})
    print("📜 历史消息类型：", [type(m) for m in messages])

    result = rag_chain.invoke(
        {"question": query, "history": messages},
        config={"run_name": "Zhipu-RAG-Session"}
    )

    print("\n🤖 AI 回答：", result)
    history = update_history(history, query, result)
