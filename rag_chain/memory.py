from langchain_core.messages import HumanMessage, AIMessage

def build_messages_from_history(history, current_question):
    return history + [HumanMessage(content=current_question)]

def update_history(history, query, response, max_turns=4):
    history.extend([
        HumanMessage(content=query),
        AIMessage(content=response)
    ])
    return history[-(2 * max_turns):]
