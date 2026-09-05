# %%
import asyncio
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from openinference.instrumentation.ollama import OllamaInstrumentor
from phoenix.otel import register

tracer_provider = register(project_name="langgraph_test")
OllamaInstrumentor().instrument(tracer_provider=tracer_provider)

# %%
llm = ChatOllama(
    model="huihui_ai/qwen3.5-abliterated:0.8b",
    temperature=0.7,
    reasoning=False,
)


@tool
def add(num1: float, num2: float):
    """adds two numbers"""
    return num1 + num2


@tool
def subtract(num1: float, num2: float):
    """subtracts two numbers [num1 - num2]"""
    return num1 - num2


@tool
def multiply(num1: float, num2: float):
    """multiplies two numbers"""
    return num1 * num2


@tool
def divide(num1: float, num2: float):
    """divides two numbers"""
    return num1 / num2


tools = [add, subtract, multiply, divide]
llm = llm.bind_tools(tools)


# %%
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


async def process_node(state: AgentState) -> AgentState:
    system_prompt = SystemMessage("You are a helpful assistant, ensure to use tools to have accurate results when appropiate")
    response = await llm.ainvoke([system_prompt, *state["messages"]])
    return AgentState(messages=[response])


def should_call_tools(state: AgentState) -> str:
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "yes"
    else:
        return "no"


graph = StateGraph(AgentState)
graph.add_node("process", process_node)

tool_node = ToolNode(tools)
graph.add_node("tool", tool_node)

graph.add_edge(START, "process")
graph.add_conditional_edges(
    "process",
    should_call_tools,
    {
        "yes": "tool",
        "no": END,
    },
)
graph.add_edge("tool", "process")

app = graph.compile()

# %%


async def main():
    agent_state = AgentState(
        messages=[
            HumanMessage(
                "Hey what is 34 + 53, taking the result dividing it by 7 then multiplying result by 20, and after that tell me a nice joke"
            )
        ]
    )
    response = await app.ainvoke(agent_state)
    print(response["messages"][-1].content)

    for msg in response["messages"]:
        print(msg)


if __name__ == "__main__":
    asyncio.run(main())