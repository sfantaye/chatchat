# main.py

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os
from llm_handler import get_response
import asyncio


app = FastAPI()
# Use absolute path to the frontend folder
frontend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")

# Serve static files from the 'frontend' folder
app.mount("/static", StaticFiles(directory=frontend_path), name="static")

# Serve the index.html at the root URL
@app.get("/", response_class=HTMLResponse)
async def read_root():
    file_path = os.path.join(frontend_path, "index.html")  # Absolute path to index.html
    with open(file_path, "r") as f:
        return HTMLResponse(content=f.read())

# WebSocket route to handle the chat messages
@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()

    # Send a welcome message
    await websocket.send_text("Hello! I'm Alex, your support agent at Lodestone. How can I assist you today?")

    messages = []  # Initialize the messages list to keep track of the conversation

    try:
        while True:
            data = await websocket.receive_text()  # Receive the message from the user
            messages.append({"role": "user", "content": data})  # Append user message to the messages list

            # Get AI response from LangGraph + Groq API
            response = await get_response(messages)

            # Send the AI's response back to the user
            await websocket.send_text(response)

            # Optionally, simulate typing indicator by waiting before replying
            await asyncio.sleep(1)  # You can adjust the sleep time for the typing effect

    except WebSocketDisconnect:
        print("Client disconnected")
