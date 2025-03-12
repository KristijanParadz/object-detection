import asyncio
import base64
import cv2
import numpy as np
import random
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from pathlib import Path
from ultralytics import YOLO


class ReIDModel:
    def __init__(self, device):
        self.device = device
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
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        inp = self.transform(image_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(inp)
        feat = feat.view(feat.size(0), -1).cpu().numpy()[0]
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat /= norm
        return feat


class GlobalIDManager:
    def __init__(self, similarity_threshold):
        self.similarity_threshold = similarity_threshold
        self.global_tracks = {}
        self.global_id_to_class = {}
        self.next_global_id = 1

    def match_or_create(self, embedding, class_id):
        if class_id not in self.global_tracks:
            self.global_tracks[class_id] = {}

        best_g_id = None
        best_sim = -1.0
        for g_id, (g_emb, g_color) in self.global_tracks[class_id].items():
            sim = float(np.dot(embedding, g_emb))
            if sim > best_sim:
                best_sim = sim
                best_g_id = g_id

        if best_g_id is not None and best_sim >= self.similarity_threshold:
            return best_g_id

        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
        new_g_id = self.next_global_id
        self.next_global_id += 1

        self.global_tracks[class_id][new_g_id] = (embedding, color)
        self.global_id_to_class[new_g_id] = class_id

        return new_g_id

    def get_color(self, global_id):
        if global_id not in self.global_id_to_class:
            return (255, 255, 255)  # default/fallback
        cls_id = self.global_id_to_class[global_id]
        if cls_id in self.global_tracks and global_id in self.global_tracks[cls_id]:
            return self.global_tracks[cls_id][global_id][1]
        return (255, 255, 255)

    def update_embedding(self, global_id, new_embedding, alpha=0.7):
        if global_id not in self.global_id_to_class:
            return
        cls_id = self.global_id_to_class[global_id]
        if cls_id not in self.global_tracks or global_id not in self.global_tracks[cls_id]:
            return

        old_emb, color = self.global_tracks[cls_id][global_id]
        blended = alpha * old_emb + (1.0 - alpha) * new_embedding
        norm = np.linalg.norm(blended)
        if norm > 0:
            blended /= norm
        self.global_tracks[cls_id][global_id] = (blended, color)

    def reset(self):
        self.global_tracks.clear()
        self.global_id_to_class.clear()
        self.next_global_id = 1


class YOLOVideoTracker:
    def __init__(self, video_path, sio, model_path, skip_interval,
                 resized_shape, video_id, global_manager,
                 embedding_update_interval):
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
        self.tracks = {}  # local_id -> {"class_id", "global_id", "embedding", "last_update_frame"}
        self.last_xyxy = []
        self.last_ids = []
        self.reid_model = ReIDModel(self.device)
        self.global_manager = global_manager
        self.embedding_update_interval = embedding_update_interval

    def process_frame(self, frame):
        self.last_xyxy.clear()
        self.last_ids.clear()
        results = self.model.track(
            frame, persist=True, device=self.device, verbose=False
        )
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

                crop = frame[y1:y2, x1:x2]

                if local_id not in self.tracks:
                    emb = self.reid_model.get_embedding(crop)
                    g_id = self.global_manager.match_or_create(emb, class_id)
                    self.tracks[local_id] = {
                        "class_id": class_id,
                        "global_id": g_id,
                        "embedding": emb,
                        "last_update_frame": self.frame_counter
                    }
                else:
                    track_data = self.tracks[local_id]
                    # Update embedding only if enough frames have passed
                    if (self.frame_counter - track_data["last_update_frame"]) >= self.embedding_update_interval:
                        new_emb = self.reid_model.get_embedding(crop)
                        self.global_manager.update_embedding(
                            track_data["global_id"], new_emb)
                        track_data["embedding"] = new_emb
                        track_data["last_update_frame"] = self.frame_counter

            # Remove any tracks not seen this frame
            active_ids = set(self.last_ids)
            self.tracks = {
                tid: data
                for tid, data in self.tracks.items()
                if tid in active_ids
            }

            # ----------------------------------------------------
            #   REMOVE DUPLICATES (same global ID) IN THIS FRAME
            # ----------------------------------------------------
            from collections import defaultdict

            # Build map: global_id -> [local_ids that have this gID in this frame]
            g_id_map = defaultdict(list)
            for l_id in self.last_ids:
                g_id = self.tracks[l_id]["global_id"]
                g_id_map[g_id].append(l_id)

            # For each global ID that appears multiple times, keep the first local ID
            # and remove the others from self.tracks, last_xyxy, last_ids
            for g_id, local_ids_for_gid in g_id_map.items():
                if len(local_ids_for_gid) > 1:
                    # Keep the first local ID, remove the rest
                    first_id = local_ids_for_gid[0]
                    for duplicate_id in local_ids_for_gid[1:]:
                        self.tracks.pop(duplicate_id, None)

            # Now rebuild self.last_xyxy / self.last_ids to include only surviving local IDs
            new_last_xyxy = []
            new_last_ids = []
            for (xy, l_id) in zip(self.last_xyxy, self.last_ids):
                if l_id in self.tracks:
                    new_last_xyxy.append(xy)
                    new_last_ids.append(l_id)

            self.last_xyxy = new_last_xyxy
            self.last_ids = new_last_ids
            # ----------------------------------------------------

        return self.draw_last_boxes(frame)

    def draw_last_boxes(self, frame):
        for (x1, y1, x2, y2), local_id in zip(self.last_xyxy, self.last_ids):
            if local_id not in self.tracks:
                continue
            data = self.tracks[local_id]
            g_id = data["global_id"]
            color = self.global_manager.get_color(g_id)
            text = f"ID:{g_id}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, text, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return frame

    def replace_global_id(self, old_g_id, new_g_id):
        """
        Called when duplicates are merged in the GlobalIDManager.
        If a track references old_g_id, switch it to new_g_id.
        """
        for track_data in self.tracks.values():
            if track_data["global_id"] == old_g_id:
                track_data["global_id"] = new_g_id

    async def send_image_to_frontend(self, image):
        _, buffer = cv2.imencode('.jpg', image)
        base64_jpg = base64.b64encode(buffer).decode('utf-8')
        await self.sio.emit('image', {"video_id": self.video_id, "image": base64_jpg})


class MultiVideoProcessor:
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

        # Lower threshold means more merges.
        # You can adjust or even separate "create-threshold" from "merge-threshold".
        self.global_manager = GlobalIDManager(similarity_threshold=0.3)

        self._init_trackers()

    def _init_trackers(self):
        self.trackers = []
        for vp in self.video_paths:
            video_id = Path(vp).stem
            t = YOLOVideoTracker(
                video_path=vp,
                sio=self.sio,
                model_path=self.model_path,
                skip_interval=self.skip_interval,
                resized_shape=self.resized_shape,
                video_id=video_id,
                global_manager=self.global_manager,
                embedding_update_interval=60
            )
            self.trackers.append(t)

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

                frame = cv2.resize(frame, tracker.resized_shape)

                if tracker.frame_counter % tracker.skip_interval == 0:
                    processed_frame = tracker.process_frame(frame)
                else:
                    # Just draw known boxes
                    processed_frame = tracker.draw_last_boxes(frame)
                await tracker.send_image_to_frontend(processed_frame)
            if not any_frame_ok:
                break
        for tracker in self.trackers:
            if tracker.cap.isOpened():
                tracker.cap.release()

    def stop(self):
        self.stopped = True
        for tracker in self.trackers:
            if tracker.cap.isOpened():
                tracker.cap.release()
        self.trackers = []

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    async def reset(self):
        self.stop()
        await asyncio.sleep(0.1)
        self.global_manager.reset()
        self._init_trackers()
        self.stopped = False
        self.paused = False
        asyncio.create_task(self.run())
