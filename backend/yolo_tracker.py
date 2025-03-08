import cv2
import torch
from ultralytics import YOLO
import base64
from pathlib import Path


class YOLOVideoTracker:
    def __init__(self, video_path, sio, model_path, skip_interval, resized_shape, calib):
        self.video_path = video_path
        self.sio = sio
        self.model_path = model_path
        self.skip_interval = skip_interval
        self.resized_shape = resized_shape
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.calib = calib  # Camera calibration object
        self.font_size = 0.5

        self.model = YOLO(self.model_path).to(self.device)
        self.cap = cv2.VideoCapture(self.video_path)
        self.frame_counter = -1
        self.video_id = Path(video_path).stem

        # For storing last known boxes and world coordinates
        self.last_xyxy = []
        self.last_ids = []
        self.last_world_coords = []  # Store global positions

    def process_frame(self, frame):
        """
        Perform YOLO detection/tracking, update self.last_xyxy/self.last_ids/self.last_world_coords,
        and draw the bounding boxes.
        """
        results = self.model.track(
            frame, persist=True, verbose=False, device=self.device
        )

        self.last_xyxy = []
        self.last_ids = []
        self.last_world_coords = []

        if results and len(results) > 0 and results[0].boxes:
            for i, box in enumerate(results[0].boxes.xyxy):
                x1, y1, x2, y2 = map(int, box[:4])
                # Get center of bounding box
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # Get unique tracking ID
                track_id = (results[0].boxes.id[i]
                            if results[0].boxes.id is not None else i)

                # Convert to global world coordinates
                world_pos = self.calib.image_to_world(self.video_id, (cx, cy))

                if world_pos is not None:
                    x_world, y_world = world_pos
                    self.last_world_coords.append((x_world, y_world))

                self.last_xyxy.append((x1, y1, x2, y2))
                self.last_ids.append(track_id)

            # Draw the new boxes on this frame
            for (x1, y1, x2, y2), track_id, world_pos in zip(self.last_xyxy, self.last_ids, self.last_world_coords):
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(frame, f'ID: {int(track_id)} X:{world_pos[0]:.2f} Y:{world_pos[1]:.2f}',
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, self.font_size, (0, 255, 0), 1)

        return frame

    def draw_last_boxes(self, frame):
        """
        Draw the old bounding boxes (from self.last_xyxy/self.last_ids)
        on the given frame without running YOLO again.
        """
        for (x1, y1, x2, y2), track_id, (world_x, world_y) in zip(self.last_xyxy, self.last_ids, self.last_world_coords):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(frame, f'ID: {int(track_id)} X:{world_x:.2f} Y:{world_y:.2f}',
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, self.font_size, (0, 255, 0), 1)

        return frame

    async def send_image_to_frontend(self, image):
        _, buffer = cv2.imencode('.jpg', image)
        base64_jpg = base64.b64encode(buffer).decode('utf-8')
        await self.sio.emit('image', {"video_id": self.video_id, "image": base64_jpg})
