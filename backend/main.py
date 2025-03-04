import uvicorn
import socketio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from yolo_tracker import YOLOVideoTracker

# 1. Create a Socket.IO server (async_mode="asgi" is important for FastAPI/ASGI apps).
sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins="*"  # <-- allows all origins
)

# 2. Create a FastAPI app.
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
async def start(sid):
    video_tracker = YOLOVideoTracker(video_path='videos/hd_0.mp4', sio=sio)
    await video_tracker.run()

if __name__ == "__main__":
    # Note: we run `socket_app`, not just `app`
    uvicorn.run(socket_app, host="127.0.0.1", port=8000)
