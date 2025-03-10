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

# -------------------
# ReIDModel for computing embeddings
# -------------------


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


# -------------------
# MultiVideoSingleLoop - manager for multiple cameras
# -------------------
class MultiVideoSingleLoop:
    def __init__(self, video_paths, sio, model_path='yolov8n.pt',
                 skip_interval=5, resized_shape=(640, 360)):
        self.video_paths = video_paths
        self.sio = sio
        self.model_path = model_path
        self.skip_interval = skip_interval
        self.resized_shape = resized_shape

        # Control flags
        self.paused = False
        self.stopped = False

        # Create trackers for each camera
        self._init_trackers()

        # Similarity threshold
        self.similarity_threshold = 0.3  # can tune as needed

    def _init_trackers(self):
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
            processed_frames = []

            # 1) For each camera, read a frame and run YOLO detection/tracking
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

                # Store (tracker, processed_frame) so we can re-draw after cross-camera matching
                processed_frames.append((tracker, processed_frame))

            if not any_frame_ok:
                break

            # 2) Cross-camera matching
            self.cross_camera_match()

            # 3) Re-draw bounding boxes with updated display ID/color
            #    and send to the frontend
            for (tracker, cached_frame) in processed_frames:
                final_frame = tracker.draw_last_boxes(cached_frame)
                await tracker.send_image_to_frontend(final_frame)

        # Release
        for tracker in self.trackers:
            if tracker.cap.isOpened():
                tracker.cap.release()

    def stop(self):
        self.stopped = True

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    async def reset(self):
        self.stopped = True
        await asyncio.sleep(0.1)
        self._init_trackers()
        self.stopped = False
        self.paused = False
        asyncio.create_task(self.run())

    # ------------------------------------
    # Cross-camera matching step
    # ------------------------------------
    def cross_camera_match(self):
        """
        1. Collect all embeddings from each camera for the current frame.
        2. Compare each pair from different cameras with the same class.
        3. If similarity > threshold, set cameraB's display_id/color to cameraA's.
        """
        # Gather all
        all_detections = []  # list of (tracker, local_id, embedding, class_id)
        for tracker in self.trackers:
            for local_id, (emb, c_id, local_color, disp_id, disp_color) in tracker.embeddings.items():
                all_detections.append((tracker, local_id, emb, c_id))

        # Compare each pair
        for i in range(len(all_detections)):
            trackerA, idA, embA, clsA = all_detections[i]
            for j in range(i + 1, len(all_detections)):
                trackerB, idB, embB, clsB = all_detections[j]

                # Different camera, same class
                if trackerA is trackerB:
                    continue
                if clsA != clsB:
                    continue

                # Cosine similarity
                sim = float(np.dot(embA, embB))
                if sim > self.similarity_threshold:
                    # They match => cameraB's display = cameraA's display

                    # Retrieve camera A's display_id and color
                    # (embedding, class_id, local_color, display_id, display_color)
                    _, _, _, a_disp_id, a_disp_color = trackerA.embeddings[idA]

                    # Overwrite cameraB's display id/color
                    embB_orig, clsB_orig, locB, _, _ = trackerB.embeddings[idB]
                    trackerB.embeddings[idB] = (
                        embB_orig,  # same embedding
                        clsB_orig,  # same class
                        locB,       # same local color
                        a_disp_id,  # new display ID
                        a_disp_color  # new display color
                    )


# -------------------
# YOLOVideoTracker - single camera
# -------------------
class YOLOVideoTracker:
    def __init__(self, video_path, sio, model_path,
                 skip_interval, resized_shape, video_id):
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

        # last known boxes & local track IDs
        self.last_xyxy = []
        self.last_ids = []
        # embeddings: track_id -> (embedding, class_id, local_color, display_id, display_color)
        self.embeddings = {}
        self.reid_model = ReIDModel(self.device)

    def process_frame(self, frame):
        """Run YOLO tracking, compute embeddings, store them, then do a preliminary draw."""
        self.last_xyxy = []
        self.last_ids = []

        results = self.model.track(
            frame, persist=True, verbose=False, device=self.device)

        if results and len(results) > 0 and results[0].boxes:
            for i, box in enumerate(results[0].boxes.xyxy):
                x1, y1, x2, y2 = map(int, box[:4])
                # YOLO track ID
                track_id = int(results[0].boxes.id[i]
                               ) if results[0].boxes.id is not None else i
                # YOLO class
                class_id = int(
                    results[0].boxes.cls[i]) if results[0].boxes.cls is not None else 0

                self.last_xyxy.append((x1, y1, x2, y2))
                self.last_ids.append(track_id)

                # Compute embedding

                # If new track, init with local color = display color, local ID = display ID
                if track_id not in self.embeddings:
                    crop = frame[y1:y2, x1:x2]
                    embedding = self.reid_model.get_embedding(crop)
                    color = (
                        random.randint(0, 255),
                        random.randint(0, 255),
                        random.randint(0, 255)
                    )
                    self.embeddings[track_id] = (
                        embedding,        # embedding
                        class_id,         # class
                        color,            # local_color
                        # display_id (initially same as YOLO ID)
                        track_id,
                        # display_color (same as local color initially)
                        color
                    )

            # Remove embeddings for IDs not seen this frame
            active_ids = set(self.last_ids)
            self.embeddings = {
                tid: v for tid, v in self.embeddings.items() if tid in active_ids
            }

        return self.draw_last_boxes(frame)

    def draw_last_boxes(self, frame):
        """
        Draw bounding boxes using the 'display_id' and 'display_color'
        from self.embeddings, not the original YOLO ID/color (unless they match).
        """
        for (x1, y1, x2, y2), track_id in zip(self.last_xyxy, self.last_ids):
            if track_id not in self.embeddings:
                continue

            # Unpack
            embedding, class_id, local_color, display_id, display_color = self.embeddings[
                track_id]

            cv2.rectangle(frame, (x1, y1), (x2, y2), display_color, 2)
            cv2.putText(frame, f'ID: {display_id}',
                        (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        self.font_size, display_color, 1)
        return frame

    async def send_image_to_frontend(self, image):
        _, buffer = cv2.imencode('.jpg', image)
        base64_jpg = base64.b64encode(buffer).decode('utf-8')
        await self.sio.emit('image', {"video_id": self.video_id, "image": base64_jpg})
