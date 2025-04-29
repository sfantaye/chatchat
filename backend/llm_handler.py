from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import os
import groq
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Groq API key
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize the Groq client
client = groq.AsyncGroq(api_key=groq_api_key)

# Define the structure of the chat state
class ChatState(TypedDict):
    messages: Annotated[list, "Messages so far"]

# Define the LangGraph node that will interact with the Groq API
async def llm_node(state: ChatState) -> ChatState:
    """
    This function interacts with the Groq API to generate a response based on the state.
    It adds the user's message to the state and appends the AI's response.
    """
    # Send the conversation history to Groq for processing
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
    # Extract the AI response
    reply = response.choices[0].message

    # Update the conversation state with the AI's reply
    state["messages"].append({"role": reply.role, "content": reply.content})

    return state

# Define and compile the state machine (workflow)
workflow = StateGraph(ChatState)
workflow.add_node("chat", llm_node)
workflow.set_entry_point("chat")
workflow.add_edge("chat", END)

# Compile the workflow into a callable function
app_flow = workflow.compile()

# Define a helper function to get a response from the AI
async def get_response(messages: list) -> str:
    """
    This function takes a list of messages and returns the AI's response.
    """
    state = {"messages": messages}
    result = await app_flow.invoke(state)
    return result["messages"][-1]["content"]
