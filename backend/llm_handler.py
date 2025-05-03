import os
import logging
from dotenv import load_dotenv
import groq

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List, Tuple

# --- Setup ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    logging.error("GROQ_API_KEY not found in environment variables.")
    raise ValueError("GROQ_API_KEY is not set.")

try:
    groq_client = groq.AsyncGroq(api_key=groq_api_key)
    logging.info("Groq client initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize Groq client: {e}")
    raise

# --- LangGraph State Definition ---
class ChatState(TypedDict):
    # List of messages, where each message is {"role": "user" | "assistant", "content": str}
    messages: Annotated[List[dict], "Conversation history"]

# --- LangGraph Node ---
async def llm_node(state: ChatState) -> ChatState:
    """Calls Groq API with the current conversation history + system prompt."""
    logging.info(f"llm_node executing with {len(state['messages'])} messages in history.")
    # print(f"DEBUG llm_node messages in: {state['messages']}") # Uncomment for deep debugging
    try:
        system_prompt = {
            "role": "system",
            "content": (
                "You are Sintayehu, a friendly and highly knowledgeable customer support agent for Lodestone. "
                "Your personality is helpful, patient, and slightly enthusiastic. "
                "Introduce yourself as Sintayehu initially if relevant, but don't repeat it excessively. "
                "Your primary goal is to answer questions ONLY about Lodestone's services, products, policies, pricing, "
                "and other company-related matters. Be concise and clear. "
                "If asked about anything outside this scope (e.g., general knowledge, coding help, personal opinions), "
                "politely steer the conversation back. Example refusal: 'That's an interesting question! However, "
                "my expertise is focused on Lodestone Software. How can I help you with our services today?' or "
                "'I can only assist with inquiries related to Lodestone. Do you have any questions about our offerings?'"
            )
        }
        # Combine system prompt with actual conversation history for the API call
        messages_for_api = [system_prompt] + state["messages"]

        response = await groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=messages_for_api,
            temperature=0.7, # Adjust for creativity vs consistency
            max_tokens=1024, # Limit response length
        )
        ai_message = response.choices[0].message
        logging.info(f"llm_node received reply from Groq: {ai_message.content[:80]}...")

        # Return a *new* state dictionary containing the updated messages list
        updated_messages = state["messages"] + [{"role": ai_message.role, "content": ai_message.content}]
        return {"messages": updated_messages}

    except groq.APIError as e:
        logging.error(f"Groq API Error in llm_node: {e.status_code} - {e.message}")
        error_message = "My apologies, I'm having a bit of trouble connecting to my knowledge base right now. Could you please try again in a moment?"
        # Append error message as assistant response
        updated_messages = state["messages"] + [{"role": "assistant", "content": error_message}]
        return {"messages": updated_messages}
    except Exception as e:
        logging.exception("Unexpected error in llm_node.") # Log full traceback
        error_message = "I seem to have encountered an unexpected technical glitch. Please try rephrasing your question or try again later."
         # Append error message as assistant response
        updated_messages = state["messages"] + [{"role": "assistant", "content": error_message}]
        return {"messages": updated_messages}

# --- LangGraph Workflow Definition & Compilation ---
workflow = StateGraph(ChatState)
workflow.add_node("chat", llm_node)
workflow.set_entry_point("chat")
workflow.add_edge("chat", END) # Simple graph: entry -> chat node -> end

try:
    app_flow = workflow.compile()
    logging.info("LangGraph workflow compiled successfully.")
except Exception as e:
    logging.error(f"Failed to compile LangGraph workflow: {e}")
    raise

# --- Interaction Function for FastAPI ---
async def get_response(current_messages: List[dict]) -> Tuple[str, List[dict]]:
    """
    Takes the current message list, invokes the LangGraph flow,
    and returns the latest AI response text and the full updated message list.
    """
    if not isinstance(current_messages, list):
        logging.error(f"Invalid input: get_response requires a list, got {type(current_messages)}")
        error_msg = "Internal error: Invalid message format received."
        return error_msg, [{"role": "assistant", "content": error_msg}]

    initial_state: ChatState = {"messages": current_messages}
    logging.info(f"get_response invoking graph with {len(current_messages)} user/assistant messages.")

    try:
        # Asynchronously invoke the compiled graph
        result_state = await app_flow.ainvoke(initial_state)

        updated_messages = result_state.get("messages", [])

        # Extract the content of the very last message (should be the assistant's response)
        if updated_messages and updated_messages[-1].get("role") == "assistant":
            ai_reply_content = updated_messages[-1].get("content", "").strip()
            if not ai_reply_content:
                logging.warning("Received empty content from assistant.")
                ai_reply_content = "I don't have a specific response for that right now. Could you ask differently?"
                # Correct the last message if it was empty
                updated_messages[-1]["content"] = ai_reply_content

            logging.info(f"get_response returning AI reply and updated history ({len(updated_messages)} total messages).")
            return ai_reply_content, updated_messages
        else:
            # This case indicates an issue in the llm_node or graph flow
            logging.error(f"Graph finished, but last message not from assistant. Result: {result_state}")
            error_message = "I wasn't able to generate a response for that. Please try again."
            # Ensure an assistant message is added, even if it's an error
            final_messages = updated_messages + [{"role": "assistant", "content": error_message}]
            return error_message, final_messages

    except Exception as e:
        logging.exception("Exception during LangGraph invocation in get_response.")
        error_message = "I encountered an error while processing your request. Please try again later."
        # Return the error and an updated list containing the error message
        final_messages = current_messages + [{"role": "assistant", "content": error_message}]
        return error_message, final_messages
