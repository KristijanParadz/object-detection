import numpy as np
import base64
import torch
import random
import cv2
from ultralytics import YOLO
from pathlib import Path
import asyncio


import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms


# -----------------------------------------------------------
#               ReID Model
# -----------------------------------------------------------
class ReIDModel:
    """
    Lightweight ReID model (ResNet50 -> final avgpool -> 2048-dim).
    Embeddings are L2-normalized.
    """

    def __init__(self, device):
        self.device = device
        # Load a pre-trained ResNet50 and remove its final classification layer
        base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model = nn.Sequential(*(list(base_model.children())[:-1]))
        self.model.eval()
        self.model.to(self.device)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def get_embedding(self, image_bgr):
        """
        Convert a BGR NumPy array (crop) to a normalized embedding.
        Only called once per new track in this example.
        """
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        inp = self.transform(image_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(inp)  # shape: [1, 2048, 1, 1]
        feat = feat.view(feat.size(0), -1).cpu().numpy()[0]  # [2048]
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat /= norm
        return feat


# -----------------------------------------------------------
#               Global ID Manager
# -----------------------------------------------------------
class GlobalIDManager:
    """
    Maintains a dictionary of global_id -> (embedding, class_id, color).
    Returns a global ID for each new track, reusing an existing global ID if
    the embedding is sufficiently similar (same class).
    """

    def __init__(self, similarity_threshold=0.7):
        self.similarity_threshold = similarity_threshold
        # global_id -> (representative_embedding, class_id, color)
        self.global_tracks = {}
        self.next_global_id = 1

    def match_or_create(self, embedding, class_id):
        """
        Compare 'embedding' to existing global tracks of the same class.
        If best similarity >= threshold, reuse that global_id.
        Otherwise, create a new global_id and return it.
        """
        best_g_id = None
        best_sim = -1.0

        for g_id, (g_emb, g_cls, g_color) in self.global_tracks.items():
            if g_cls != class_id:
                continue
            sim = float(np.dot(embedding, g_emb))
            if sim > best_sim:
                best_sim = sim
                best_g_id = g_id

        if best_g_id is not None and best_sim >= self.similarity_threshold:
            return best_g_id
        else:
            # create a new global ID
            color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )
            self.global_tracks[self.next_global_id] = (
                embedding, class_id, color)
            assigned_id = self.next_global_id
            self.next_global_id += 1
            return assigned_id

    def get_color(self, global_id):
        """
        Retrieve the color for a global_id, or fallback if not found.
        """
        if global_id in self.global_tracks:
            return self.global_tracks[global_id][2]
        else:
            return (255, 255, 255)

    def update_embedding(self, global_id, new_embedding, alpha=0.7):
        """
        Optional: update the stored embedding by blending in 'new_embedding'
        with weight alpha for the old embedding. Re-normalize.
        """
        if global_id not in self.global_tracks:
            return
        old_emb, cls_id, color = self.global_tracks[global_id]
        blended = alpha * old_emb + (1.0 - alpha) * new_embedding
        norm = np.linalg.norm(blended)
        if norm > 0:
            blended /= norm
        self.global_tracks[global_id] = (blended, cls_id, color)


# -----------------------------------------------------------
#               Per-Camera YOLO Tracker
# -----------------------------------------------------------
class YOLOVideoTracker:
    """
    Each camera has:
      - YOLO tracker (persist=True).
      - ReIDModel for embeddings.
      - A local dictionary: local_track_id -> (class_id, global_id)
        plus a single embedding (computed once).
    """

    def __init__(self, video_path, sio, model_path, skip_interval,
                 resized_shape, video_id, global_manager):
        self.video_path = video_path
        self.sio = sio
        self.model_path = model_path
        self.skip_interval = skip_interval
        self.resized_shape = resized_shape
        self.video_id = video_id
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = YOLO(self.model_path).to(self.device)
        self.cap = cv2.VideoCapture(self.video_path)
        self.frame_counter = -1

        # local dictionary:
        # local_id -> {"class_id": c, "global_id": g_id, "embedding": emb}
        self.tracks = {}

        # bounding boxes from last detection
        self.last_xyxy = []
        self.last_ids = []

        # ReID
        self.reid_model = ReIDModel(self.device)

        # reference to the global manager
        self.global_manager = global_manager

    def process_frame(self, frame):
        """
        YOLO track. If we see a brand-new local_id, we compute the embedding once,
        match/create global ID, and store it. If we see an existing local_id, we skip
        embedding computation and reuse what's stored.
        """
        self.last_xyxy.clear()
        self.last_ids.clear()

        results = self.model.track(
            frame, persist=True, device=self.device, verbose=False)
        if results and len(results) > 0 and results[0].boxes:
            boxes = results[0].boxes.xyxy
            ids = results[0].boxes.id
            clss = results[0].boxes.cls
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box[:4])
                local_id = int(ids[i]) if ids is not None else i
                class_id = int(clss[i]) if clss is not None else 0

                self.last_xyxy.append((x1, y1, x2, y2))
                self.last_ids.append(local_id)

                # If it's the first time we see this local track, compute embedding once
                if local_id not in self.tracks:
                    crop = frame[y1:y2, x1:x2]
                    emb = self.reid_model.get_embedding(crop)

                    # match or create global ID
                    g_id = self.global_manager.match_or_create(emb, class_id)

                    self.tracks[local_id] = {
                        "class_id": class_id,
                        "global_id": g_id,
                        "embedding": emb
                    }
                else:
                    # otherwise do nothing, we already have embedding & global_id
                    pass

            # Prune local IDs not in this frame
            active_ids = set(self.last_ids)
            self.tracks = {
                tid: data for tid, data in self.tracks.items() if tid in active_ids
            }

        # Draw bounding boxes
        return self.draw_last_boxes(frame)

    def draw_last_boxes(self, frame):
        """
        Use the global manager's color for each global_id. 
        Show "GID: #".
        """
        for (x1, y1, x2, y2), local_id in zip(self.last_xyxy, self.last_ids):
            if local_id not in self.tracks:
                continue
            data = self.tracks[local_id]
            g_id = data["global_id"]
            color = self.global_manager.get_color(g_id)
            text = f"GID: {g_id}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, text, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return frame

    async def send_image_to_frontend(self, image):
        _, buffer = cv2.imencode('.jpg', image)
        base64_jpg = base64.b64encode(buffer).decode('utf-8')
        await self.sio.emit('image', {"video_id": self.video_id, "image": base64_jpg})


# -----------------------------------------------------------
#              MultiVideoSingleLoop
# -----------------------------------------------------------
class MultiVideoSingleLoop:
    """
    Manages multiple YOLOVideoTracker (one per camera).
    Uses a single GlobalIDManager so that if an object
    appears in multiple cameras, they share the same global ID.
    """

    def __init__(self, video_paths, sio,
                 model_path='yolov8n.pt',
                 skip_interval=5,
                 resized_shape=(640, 360)):
        self.video_paths = video_paths
        self.sio = sio
        self.model_path = model_path
        self.skip_interval = skip_interval
        self.resized_shape = resized_shape

        self.paused = False
        self.stopped = False

        # One global manager across all cameras
        self.global_manager = GlobalIDManager(similarity_threshold=0.3)

        self._init_trackers()

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
                video_id=video_id,
                global_manager=self.global_manager
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

                # Only run process_frame on skip interval
                # otherwise just draw last known boxes
                if tracker.frame_counter % tracker.skip_interval == 0:
                    processed_frame = tracker.process_frame(frame)
                else:
                    processed_frame = tracker.draw_last_boxes(frame)

                await tracker.send_image_to_frontend(processed_frame)

            if not any_frame_ok:
                break

        # Cleanup
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
        """
        Reset videos to frame=0 by:
          1) Stopping the current loop.
          2) Re-init trackers.
          3) Start run() again in a fresh task.
        """
        self.stopped = True
        await asyncio.sleep(0.1)

        self._init_trackers()
        self.stopped = False
        self.paused = False

        asyncio.create_task(self.run())
