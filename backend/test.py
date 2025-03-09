import numpy as np
import base64
from ultralytics import YOLO
import torch
import cv2
import asyncio
from pathlib import Path


class MultiVideoSingleLoop:
    def __init__(self, video_paths, sio, model_path='yolov8n.pt', skip_interval=5, resized_shape=(640, 360)):
        self.video_paths = video_paths
        self.sio = sio
        self.model_path = model_path
        self.skip_interval = skip_interval
        self.resized_shape = resized_shape

        # Control flags
        self.paused = False
        self.stopped = False

        # Build initial trackers
        self._init_trackers()

    def _init_trackers(self):
        """Helper to create a YOLOVideoTracker for each path."""
        self.trackers = []
        for vp in self.video_paths:
            video_id = Path(vp).stem
            tracker = YOLOVideoTracker(
                video_path=vp,
                sio=self.sio,
                model_path=self.model_path,
                skip_interval=self.skip_interval,
                resized_shape=self.resized_shape,
                video_id=video_id
            )
            self.trackers.append(tracker)

    async def run(self):
        self.stopped = False
        while not self.stopped:
            if self.paused:
                await asyncio.sleep(0.05)
                continue

            any_frame_ok = False

            for tracker in self.trackers:
                if not tracker.cap.isOpened():
                    continue

                success, frame = tracker.cap.read()
                if not success:
                    tracker.cap.release()
                    continue

                any_frame_ok = True
                tracker.frame_counter += 1

                # Resize
                frame = cv2.resize(frame, tracker.resized_shape)

                # Run YOLO detection or reuse old boxes
                if tracker.frame_counter % tracker.skip_interval == 0:
                    processed_frame = tracker.process_frame(frame)
                else:
                    processed_frame = tracker.draw_last_boxes(frame)

                await tracker.send_image_to_frontend(processed_frame)

            if not any_frame_ok:
                break

        for tracker in self.trackers:
            if tracker.cap.isOpened():
                tracker.cap.release()

    def stop(self):
        self.stopped = True

    def pause(self):
        """Pause the loop (no new frames are read)."""
        self.paused = True

    def resume(self):
        """Resume the loop (start reading frames again)."""
        self.paused = False

    async def reset(self):
        """
        Reset videos to frame=0 by:
          1) Stopping the current loop.
          2) Re-init trackers.
          3) Starting run() again in a fresh task.
        """
        # Signal the current loop to stop
        self.stopped = True
        # Let the loop exit gracefully
        await asyncio.sleep(0.1)

        # Re-initialize trackers
        self._init_trackers()

        # Clear flags
        self.stopped = False
        self.paused = False

        # Start again
        asyncio.create_task(self.run())


class YOLOVideoTracker:
    def __init__(self, video_path, sio, model_path, skip_interval, resized_shape, video_id):
        self.video_path = video_path
        self.sio = sio
        self.model_path = model_path
        self.skip_interval = skip_interval
        self.resized_shape = resized_shape
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.font_size = 0.5

        self.model = YOLO(self.model_path).to(self.device)
        self.cap = cv2.VideoCapture(self.video_path)
        self.frame_counter = -1
        self.video_id = video_id

        # For storing last known boxes
        self.last_xyxy = []
        self.last_ids = []

    def process_frame(self, frame):
        results = self.model.track(
            frame, persist=True, verbose=False, device=self.device
        )

        self.last_xyxy = []
        self.last_ids = []

        if results and len(results) > 0 and results[0].boxes:
            for i, box in enumerate(results[0].boxes.xyxy):
                x1, y1, x2, y2 = map(int, box[:4])
                # Get center of bounding box
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # Get unique tracking ID
                track_id = (results[0].boxes.id[i]
                            if results[0].boxes.id is not None else i)

                self.last_xyxy.append((x1, y1, x2, y2))
                self.last_ids.append(track_id)

            # Draw the new boxes on this frame
            for (x1, y1, x2, y2), track_id in zip(self.last_xyxy, self.last_ids):
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(frame, f'ID: {int(track_id)}',
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, self.font_size, (0, 255, 0), 1)

        return frame

    def draw_last_boxes(self, frame):
        """
        Draw the old bounding boxes (from self.last_xyxy/self.last_ids)
        on the given frame without running YOLO again.
        """
        for (x1, y1, x2, y2), track_id in zip(self.last_xyxy, self.last_ids):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(frame, f'ID: {int(track_id)}',
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, self.font_size, (0, 255, 0), 1)

        return frame

    async def send_image_to_frontend(self, image):
        _, buffer = cv2.imencode('.jpg', image)
        base64_jpg = base64.b64encode(buffer).decode('utf-8')
        await self.sio.emit('image', {"video_id": self.video_id, "image": base64_jpg})
