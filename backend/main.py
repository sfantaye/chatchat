from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from llm_handler import get_response

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

connections = {}

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    connections[client_id] = websocket
    messages = []

    try:
        while True:
            user_input = await websocket.receive_text()
            messages.append({"role": "user", "content": user_input})

            await websocket.send_text("__typing__")
            reply = await get_response(messages)
            messages.append({"role": "assistant", "content": reply})
            await websocket.send_text(reply)
    except WebSocketDisconnect:
        connections.pop(client_id, None)

