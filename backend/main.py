import uvicorn
import socketio
from fastapi import FastAPI

# 1. Create a Socket.IO server (async_mode="asgi" is important for FastAPI/ASGI apps).
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins=[])

# 2. Create a FastAPI app.
app = FastAPI()

# 3. Wrap our FastAPI app with socketio's ASGI app:
socket_app = socketio.ASGIApp(sio, other_asgi_app=app, socketio_path="socket.io")

# A simple test endpoint.
@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI + Socket.IO"}

# Socket.IO event handlers:
@sio.event
async def connect(sid, environ):
    print(f"Client connected: {sid}")

@sio.event
async def disconnect(sid):
    print(f"Client disconnected: {sid}")

@sio.event
async def message(sid, data):
    print(f"Message from {sid}: {data}")
    # Echo the message back to the sender
    await sio.emit("message", f"Server says: {data}", to=sid)

if __name__ == "__main__":
    # Note: we run `socket_app`, not just `app`
    uvicorn.run(socket_app, host="127.0.0.1", port=8000)
