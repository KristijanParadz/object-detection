import base64
import cv2
import torch
from ultralytics import YOLO
from frame_processing.re_id_model import ReIDModel
from collections import defaultdict


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
        self.tracks = {}
        self.last_xyxy = []
        self.last_ids = []
        self.reid_model = ReIDModel(self.device)
        self.global_manager = global_manager
        self.embedding_update_interval = embedding_update_interval

    def _remove_local_duplicates(self):
        active_ids = set(self.last_ids)
        self.tracks = {
            tid: data
            for tid, data in self.tracks.items()
            if tid in active_ids
        }

        g_id_map = defaultdict(list)
        for l_id in self.last_ids:
            g_id = self.tracks[l_id]["global_id"]
            g_id_map[g_id].append(l_id)

        for g_id, local_ids_for_gid in g_id_map.items():
            if len(local_ids_for_gid) > 1:
                for duplicate_id in local_ids_for_gid[1:]:
                    self.tracks.pop(duplicate_id, None)

        new_last_xyxy = []
        new_last_ids = []
        for (xy, l_id) in zip(self.last_xyxy, self.last_ids):
            if l_id in self.tracks:
                new_last_xyxy.append(xy)
                new_last_ids.append(l_id)

        self.last_xyxy = new_last_xyxy
        self.last_ids = new_last_ids

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
                    if (self.frame_counter - track_data["last_update_frame"]) >= self.embedding_update_interval:
                        new_emb = self.reid_model.get_embedding(crop)
                        self.global_manager.update_embedding(
                            track_data["global_id"], new_emb)
                        track_data["embedding"] = new_emb
                        track_data["last_update_frame"] = self.frame_counter

            self._remove_local_duplicates()

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

    async def send_image_to_frontend(self, image):
        _, buffer = cv2.imencode('.jpg', image)
        base64_jpg = base64.b64encode(buffer).decode('utf-8')
        await self.sio.emit('image', {"video_id": self.video_id, "image": base64_jpg})
