import os
import asyncio
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.exceptions import RequestValidationError
import uvicorn

# --- Import the LLM handler ---
try:
    # Assumes llm_handler.py is in the same directory
    from llm_handler import get_response
except ImportError:
    logging.critical("FATAL: Could not import 'get_response' from llm_handler.py.", exc_info=True)
    # Provide a dummy function so the server *might* start, but websockets will fail
    async def get_response(messages: list):
        logging.error("CRITICAL: Using dummy get_response function due to import error.")
        error_message = "Error: Chat logic handler is not available."
        return error_message, messages + [{"role": "assistant", "content": error_message}]

# --- Basic Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="ChatChat API",
    description="FastAPI WebSocket server for Lodestone Support Chatbot using LangGraph",
    version="1.1.0"
)

# --- Static Files Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
frontend_dir = os.path.join(script_dir, "frontend")

if not os.path.isdir(frontend_dir):
    logger.critical(f"CRITICAL: Frontend directory not found at: {frontend_dir}. Exiting.")
    # Exit if frontend is essential and missing
    exit(1)
else:
    logger.info(f"Serving static files from: {frontend_dir}")
    # Mount the directory containing index.html and any potential CSS/JS files
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

# --- Global Exception Handler (Optional) ---
@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return PlainTextResponse("Internal Server Error", status_code=500)

# --- Root Route for serving the HTML ---
@app.get("/", response_class=FileResponse)
async def read_index():
    index_path = os.path.join(frontend_dir, "index.html")
    if not os.path.isfile(index_path):
         logger.error(f"index.html not found at path: {index_path}")
         return PlainTextResponse("Frontend not found.", status_code=404)
    logger.info("Serving index.html")
    return FileResponse(index_path, media_type='text/html')


# --- WebSocket Endpoint ---
@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_addr = f"{websocket.client.host}:{websocket.client.port}"
    logger.info(f"WebSocket connection accepted: {client_addr}")

    # --- IMPORTANT: Conversation history ISOLATED per connection ---
    conversation_history = [] # List of {"role": "user/assistant", "content": "..."}

    try:
        while True:
            # 1. Receive user message
            user_message = await websocket.receive_text()
            logger.info(f"Received from {client_addr}: '{user_message[:100]}...'") # Log truncated message

            # 2. Add user message to this connection's history
            conversation_history.append({"role": "user", "content": user_message})

            # 3. Send typing indicator to frontend
            try:
                await websocket.send_text("__typing__")
                logger.debug(f"Sent typing indicator to {client_addr}")
            except Exception as e:
                # Log error but continue, typing indicator isn't critical
                logger.warning(f"Could not send typing indicator to {client_addr}: {e}")

            # 4. Get response using the LLM handler function
            # Pass the current history for this specific connection
            ai_reply_text, updated_history = await get_response(conversation_history)
            logger.info(f"Sending to {client_addr}: '{ai_reply_text[:100]}...'")

            # 5. CRITICAL: Update the history for this connection
            conversation_history = updated_history

            # 6. Send AI's response back to the client
            await websocket.send_text(ai_reply_text)

    except WebSocketDisconnect:
        logger.info(f"WebSocket connection closed by client: {client_addr}")
    except ConnectionResetError:
         logger.warning(f"Connection reset by client: {client_addr}")
    except Exception as e:
        # Log any other exceptions that occur during the loop
        logger.error(f"Error during WebSocket communication with {client_addr}: {e}", exc_info=True)
        # Attempt to close gracefully if possible
        try:
            await websocket.close(code=1011) # Internal Error code
        except Exception:
            pass # Ignore if closing fails
    finally:
        # Optional: Log total messages exchanged or other cleanup tasks
        logger.info(f"Finished WebSocket session for {client_addr}. History length: {len(conversation_history)}")


# --- Main execution block to run the server ---
if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")  # Listen on all network interfaces by default
    port = int(os.getenv("PORT", 8000)) # Use port 8000 by default
    reload_flag = os.getenv("RELOAD", "true").lower() == "true" # Enable reload by default (good for dev)

    logger.info(f"Starting Uvicorn server on {host}:{port} with reload={'ON' if reload_flag else 'OFF'}...")
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload_flag # Set reload=False for production
        # You might add workers=N for production based on your CPU cores
    )