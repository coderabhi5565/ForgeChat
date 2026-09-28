from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from tools import tools
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

llm_with_tools = llm.bind_tools(tools)

def chat(state: State):

    response = llm_with_tools.invoke(
        state["messages"]
    )

    return {
        "messages": [response]
    }

graph = StateGraph(State)

graph.add_node("chat", chat)

graph.add_node(
    "tools",
    ToolNode(tools)
)


graph.add_edge(
    START,
    "chat"
)


graph.add_conditional_edges(
    "chat",
    tools_condition
)


graph.add_edge(
    "tools",
    "chat"
)

checkpoint = InMemorySaver()

app = graph.compile(
    checkpointer=checkpoint
)

config = {
    "configurable": {
        "thread_id": "chat-1"
    }
}

while True:

    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    result = app.invoke(
        {
            "messages": [
                HumanMessage(content=user_input)
            ]
        },
        config=config
    )

    print(
        "AI:",
        result["messages"][-1].content
    )