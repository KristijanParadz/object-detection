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
    """
    Now storing a *gallery* of embeddings for each ID instead of a single embedding.
    """

    def __init__(self, similarity_threshold, max_gallery_size=5):
        self.similarity_threshold = similarity_threshold
        self.max_gallery_size = max_gallery_size
        # Structure:
        # self.global_tracks[class_id][global_id] = ([emb1, emb2, ...], color)
        self.global_tracks = {}
        self.global_id_to_class = {}
        self.next_global_id = 1

    def match_or_create(self, new_embedding, class_id):
        """
        1) For each existing global_id in class_id, compute max similarity vs. that ID’s gallery.
        2) If max similarity >= threshold, return that ID.
        3) Else create a new ID (with an empty gallery to start) and return it.
        """
        if class_id not in self.global_tracks:
            self.global_tracks[class_id] = {}

        best_g_id = None
        best_sim = -1.0
        for g_id, (gallery, _) in self.global_tracks[class_id].items():
            # Compute similarity to each embedding in that gallery
            sim_list = [float(np.dot(new_embedding, emb)) for emb in gallery]
            max_sim = max(sim_list) if sim_list else -1.0
            if max_sim > best_sim:
                best_sim = max_sim
                best_g_id = g_id

        # Decide whether to match or create new
        if best_g_id is not None and best_sim >= self.similarity_threshold:
            return best_g_id

        # Create new ID
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
        new_g_id = self.next_global_id
        self.next_global_id += 1

        # Initialize with a 1-entry gallery
        self.global_tracks[class_id][new_g_id] = ([new_embedding], color)
        self.global_id_to_class[new_g_id] = class_id
        return new_g_id

    def get_color(self, global_id):
        if global_id not in self.global_id_to_class:
            return (255, 255, 255)  # fallback
        cls_id = self.global_id_to_class[global_id]
        if cls_id in self.global_tracks and global_id in self.global_tracks[cls_id]:
            return self.global_tracks[cls_id][global_id][1]
        return (255, 255, 255)

    def update_gallery(self, global_id, new_embedding):
        """
        Append new_embedding to the gallery for that global_id (if it exists),
        and remove oldest if we exceed max_gallery_size.
        """
        if global_id not in self.global_id_to_class:
            return
        cls_id = self.global_id_to_class[global_id]
        if cls_id not in self.global_tracks or global_id not in self.global_tracks[cls_id]:
            return

        gallery, color = self.global_tracks[cls_id][global_id]
        gallery.append(new_embedding)

        # If the gallery is too big, drop the oldest
        if len(gallery) > self.max_gallery_size:
            gallery.pop(0)  # remove from front (oldest)

        self.global_tracks[cls_id][global_id] = (gallery, color)

    def reset(self):
        self.global_tracks.clear()
        self.global_id_to_class.clear()
        self.next_global_id = 1


class YOLOVideoTracker:
    def __init__(self, video_path, sio, model_path, skip_interval,
                 resized_shape, video_id, global_manager,
                 embedding_update_interval=60):
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
        # local_id -> {"class_id", "global_id", "last_embedding", "last_update_frame"}
        self.tracks = {}
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

                # If we haven't seen this local_id before, create it
                if local_id not in self.tracks:
                    new_emb = self.reid_model.get_embedding(crop)
                    g_id = self.global_manager.match_or_create(
                        new_emb, class_id)
                    self.tracks[local_id] = {
                        "class_id": class_id,
                        "global_id": g_id,
                        "last_embedding": new_emb,
                        "last_update_frame": self.frame_counter
                    }
                else:
                    # Possibly update the gallery if enough frames passed
                    track_data = self.tracks[local_id]
                    if (self.frame_counter - track_data["last_update_frame"]) >= self.embedding_update_interval:
                        new_emb = self.reid_model.get_embedding(crop)
                        # Append new_emb to that ID's gallery
                        self.global_manager.update_gallery(
                            track_data["global_id"], new_emb)
                        track_data["last_embedding"] = new_emb
                        track_data["last_update_frame"] = self.frame_counter

            # Remove tracks not seen this frame
            active_ids = set(self.last_ids)
            self.tracks = {
                tid: data
                for tid, data in self.tracks.items()
                if tid in active_ids
            }

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
        If a track references old_g_id, switch it to new_g_id.
        (Used in deduplication merges.)
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

        # We use 0.3 for creation matching, and up to 5 embeddings per ID
        self.global_manager = GlobalIDManager(
            similarity_threshold=0.3, max_gallery_size=5)

        self._init_trackers()
        self.global_frame_counter = 0

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

    def unify_ids(self, cls_id, keep_id, remove_id):
        """
        Merge 'remove_id' into 'keep_id' for that class.
        Now we unify the *galleries* of keep_id and remove_id.
        """
        keep_gallery, keep_color = self.global_manager.global_tracks[cls_id][keep_id]
        remove_gallery, _ = self.global_manager.global_tracks[cls_id][remove_id]

        # Concatenate
        merged_gallery = keep_gallery + remove_gallery
        # Possibly slice if it exceeds max_gallery_size
        if len(merged_gallery) > self.global_manager.max_gallery_size:
            # keep the newest
            merged_gallery = merged_gallery[-self.global_manager.max_gallery_size:]

        # Update the keep_id with merged gallery
        self.global_manager.global_tracks[cls_id][keep_id] = (
            merged_gallery, keep_color)

        # Remove the old ID
        del self.global_manager.global_tracks[cls_id][remove_id]
        del self.global_manager.global_id_to_class[remove_id]

        # Update references in all trackers
        for tracker in self.trackers:
            tracker.replace_global_id(remove_id, keep_id)

    def deduplicate_global_tracks(self, similarity=0.5):
        """
        Example dedup logic: For each class, check all pairs of IDs. If
        max similarity among their galleries is above 'similarity', unify them.
        """
        for cls_id, id_dict in self.global_manager.global_tracks.items():
            all_ids = list(id_dict.keys())
            i = 0
            while i < len(all_ids):
                keep_id = all_ids[i]
                if keep_id not in id_dict:
                    i += 1
                    continue
                keep_gallery, _ = id_dict[keep_id]

                j = i + 1
                while j < len(all_ids):
                    check_id = all_ids[j]
                    if check_id not in id_dict:
                        j += 1
                        continue
                    check_gallery, _ = id_dict[check_id]

                    # Compare each embedding in keep_gallery to each in check_gallery
                    # Then look at the max similarity
                    # For performance, you might do an NxM dot product. We'll just do a double loop.
                    max_sim = -1.0
                    for emb1 in keep_gallery:
                        for emb2 in check_gallery:
                            sim = float(np.dot(emb1, emb2))
                            if sim > max_sim:
                                max_sim = sim

                    if max_sim > similarity:
                        # unify them
                        id_to_keep = min(keep_id, check_id)
                        id_to_remove = max(keep_id, check_id)
                        self.unify_ids(cls_id, id_to_keep, id_to_remove)

                        all_ids.remove(id_to_remove)

                        # If we changed the keep_id, fix references
                        if keep_id != id_to_keep:
                            keep_id, id_to_remove = id_to_keep, keep_id
                            keep_gallery, _ = id_dict[keep_id]

                        # Do not increment j yet, re-check with new keep_id
                    else:
                        j += 1
                i += 1

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
                self.global_frame_counter += 1

                frame = cv2.resize(frame, tracker.resized_shape)

                if tracker.frame_counter % tracker.skip_interval == 0:
                    processed_frame = tracker.process_frame(frame)
                else:
                    processed_frame = tracker.draw_last_boxes(frame)

                await tracker.send_image_to_frontend(processed_frame)

            # Occasionally run deduplication
            # (Tune how often you want to do this for performance)
            if self.global_frame_counter % 200 == 0:
                self.deduplicate_global_tracks(similarity=0.5)

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
        self.global_frame_counter = 0
        asyncio.create_task(self.run())
