import uvicorn
import socketio
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from frame_processing.multi_camera_processor import MultiCameraProcessor

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
socket_app = socketio.ASGIApp(
    sio, other_asgi_app=app, socketio_path="socket.io")

# We'll keep a global reference here for simplicity:
multi_video_tracker = None


@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI + Socket.IO"}

# ----------------------------------------------------
# Socket.IO event handlers
# ----------------------------------------------------


@sio.event
async def connect(sid, environ):
    print(f"Client connected: {sid}")


@sio.event
async def disconnect(sid):
    print(f"Client disconnected: {sid}")
    global multi_video_tracker
    multi_video_tracker.stop()


@sio.event
async def start(sid):
    """
    START event: instantiate MultiVideoSingleLoop, 
    and spawn its run() method as a background task.
    """
    global multi_video_tracker
    video_paths = [
        'videos/hd_00_00.mp4',
        # 'videos/hd_00_01.mp4',
        # 'videos/hd_00_02.mp4',
        'videos/hd_00_03.mp4'
    ]

    # Create a new MultiVideoSingleLoop instance
    multi_video_tracker = MultiCameraProcessor(
        video_paths=video_paths,
        sio=sio
    )

    # Kick off its run() in the background
    asyncio.create_task(multi_video_tracker.run())


@sio.event
async def pause(sid):
    """
    PAUSE event: tell the tracker to stop processing frames (but not exit).
    """
    global multi_video_tracker
    if multi_video_tracker:
        multi_video_tracker.pause()


@sio.event
async def stop(sid):
    """
    PAUSE event: tell the tracker to stop processing frames (but not exit).
    """
    global multi_video_tracker
    if multi_video_tracker:
        multi_video_tracker.stop()


@sio.event
async def resume(sid):
    """
    RESUME event: unpause the loop, letting it continue from where it left off.
    """
    global multi_video_tracker
    if multi_video_tracker:
        multi_video_tracker.resume()


@sio.event
async def reset(sid):
    """
    RESET event: 
      1. Stop the current loop if it’s running.
      2. Reinitialize all trackers from frame=0.
      3. Start the run loop again in a fresh task.
    """
    global multi_video_tracker
    if multi_video_tracker:
        asyncio.create_task(multi_video_tracker.reset())


if __name__ == "__main__":
    # Run `socket_app`, not just `app`
    uvicorn.run("main:socket_app", host="0.0.0.0", port=8000, reload=True)
