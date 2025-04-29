from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import os
import groq
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

client = groq.AsyncGroq(api_key=groq_api_key)

class ChatState(TypedDict):
    messages: Annotated[list, "Messages so far"]

async def llm_node(state: ChatState) -> ChatState:
    response = await client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are Alex, a friendly and knowledgeable customer support agent at Lodestone. "
                    "Always introduce yourself as Alex. Only answer questions related to Lodestone's services, "
                    "products, policies, and company-related topics. If a user asks unrelated questions, "
                    "politely redirect them to Lodestone support topics."
                )
            },
            *state["messages"]
        ],
    )
    reply = response.choices[0].message
    state["messages"].append({"role": reply.role, "content": reply.content})
    return state

# Define and compile the graph
workflow = StateGraph(ChatState)
workflow.add_node("chat", llm_node)
workflow.set_entry_point("chat")
workflow.add_edge("chat", END)
app_flow = workflow.compile()

async def get_response(messages: list) -> str:
    state = {"messages": messages}
    result = await app_flow.invoke(state)
    return result["messages"][-1]["content"]

