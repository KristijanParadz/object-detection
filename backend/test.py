import numpy as np
import base64
from ultralytics import YOLO
import torch
import cv2
import asyncio
from pathlib import Path
import random

import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms

# ReIDModel for computing embeddings


class ReIDModel:
    def __init__(self, device):
        self.device = device
        # Load a pre-trained ResNet50 model and remove its final classification layer
        base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # output shape: [B, 2048, 1, 1]
        self.model = nn.Sequential(*(list(base_model.children())[:-1]))
        self.model.eval()
        self.model.to(self.device)
        # Typical pre-processing for ImageNet models; reid models may use different sizes
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),  # common size for reid
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def get_embedding(self, image):
        """
        Given a BGR image crop (numpy array), return a normalized embedding vector.
        """
        # Convert BGR to RGB for PIL
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model(input_tensor)  # shape: [1, 2048, 1, 1]
        embedding = embedding.view(
            embedding.size(0), -1)  # flatten to [1, 2048]
        embedding = embedding.cpu().numpy()[0]
        # Normalize the embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding


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

        # For storing last known boxes and embeddings
        self.last_xyxy = []
        self.last_ids = []
        # Dictionary to store embeddings and class labels: {track_id: (embedding, class_id)}
        self.embeddings = {}
        # ReID model instance
        self.reid_model = ReIDModel(self.device)

    def process_frame(self, frame):
        results = self.model.track(
            frame, persist=True, verbose=False, device=self.device
        )

        self.last_xyxy = []
        self.last_ids = []

        if results and len(results) > 0 and results[0].boxes:
            for i, box in enumerate(results[0].boxes.xyxy):
                x1, y1, x2, y2 = map(int, box[:4])
                # Get unique tracking ID
                track_id = int(results[0].boxes.id[i]
                               if results[0].boxes.id is not None else i)
                # Attempt to retrieve class id (if provided by YOLO)
                class_id = int(
                    results[0].boxes.cls[i]) if results[0].boxes.cls is not None else None

                self.last_xyxy.append((x1, y1, x2, y2))
                self.last_ids.append(track_id)

                # Only compute the embedding the first time this track_id is seen
                if track_id not in self.embeddings:
                    # Crop the detection from the frame; note: adjust if needed for padding etc.
                    crop = frame[y1:y2, x1:x2]
                    embedding = self.reid_model.get_embedding(crop)
                    # Save a tuple of (embedding, class_id) for later use (e.g. cosine similarity comparisons)
                    color = (random.randint(0, 255), random.randint(
                        0, 255), random.randint(0, 255))
                    self.embeddings[track_id] = (embedding, class_id, color)
            # Filter embeddings: remove entries for tracks not present in this frame
            active_ids = set(self.last_ids)
            self.embeddings = {
                tid: emb for tid, emb in self.embeddings.items() if tid in active_ids}

            # Draw the new boxes on this frame (and optionally display the class)
            for (x1, y1, x2, y2), track_id in zip(self.last_xyxy, self.last_ids):
                color = self.embeddings[track_id][2]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                cv2.putText(frame, f'ID: {int(track_id)}',
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, self.font_size, color, 1)

        return frame

    def draw_last_boxes(self, frame):
        """
        Draw the old bounding boxes (from self.last_xyxy/self.last_ids)
        on the given frame without running YOLO again.
        """
        for (x1, y1, x2, y2), track_id in zip(self.last_xyxy, self.last_ids):
            color = self.embeddings[track_id][2]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            cv2.putText(frame, f'ID: {int(track_id)}',
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, self.font_size, color, 1)
        return frame

    # def compute_cosine_similarity(self, track_id1, track_id2):
    #     """
    #     Example helper: Compute cosine similarity between embeddings for two track_ids,
    #     but only if they belong to the same class. Returns None if classes differ or
    #     if embeddings are not available.
    #     """
    #     if track_id1 in self.embeddings and track_id2 in self.embeddings:
    #         emb1, class1 = self.embeddings[track_id1]
    #         emb2, class2 = self.embeddings[track_id2]
    #         if class1 == class2:
    #             # Cosine similarity (embeddings assumed to be normalized)
    #             similarity = np.dot(emb1, emb2)
    #             return similarity
    #     return None

    async def send_image_to_frontend(self, image):
        _, buffer = cv2.imencode('.jpg', image)
        base64_jpg = base64.b64encode(buffer).decode('utf-8')
        await self.sio.emit('image', {"video_id": self.video_id, "image": base64_jpg})
