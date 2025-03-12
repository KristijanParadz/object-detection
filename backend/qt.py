import sys
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from ultralytics import YOLO
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QGridLayout
)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap


###############################################################################
# 1) ReID Model
###############################################################################
class ReIDModel:
    def __init__(self, device):
        self.device = device
        base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model = nn.Sequential(*(list(base_model.children())[:-1]))
        self.model.eval().to(self.device)

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
        """Convert BGR image to embedding vector."""
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        inp = self.transform(image_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(inp)
        feat = feat.view(feat.size(0), -1).cpu().numpy()[0]
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat /= norm
        return feat


###############################################################################
# 2) Global ID Manager
###############################################################################
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
            # Match with existing global ID
            return best_g_id

        # Create new global ID
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
            return (255, 255, 255)
        cls_id = self.global_id_to_class[global_id]
        if cls_id in self.global_tracks and global_id in self.global_tracks[cls_id]:
            return self.global_tracks[cls_id][global_id][1]
        return (255, 255, 255)

    def update_embedding(self, global_id, new_embedding, alpha=0.7):
        """Blend new embedding into the old one."""
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


###############################################################################
# 3) YOLO + ReID Tracker for a Single Video
###############################################################################
class YOLOVideoTracker:
    def __init__(
            self,
            video_path,
            model_path,
            skip_interval,
            resized_shape,
            global_manager,
            embedding_update_interval,
            update_embeddings,
            display_label=None,
    ):
        self.video_path = video_path
        self.model_path = model_path
        self.skip_interval = skip_interval
        self.resized_shape = resized_shape
        self.embedding_update_interval = embedding_update_interval
        self.global_manager = global_manager
        self.update_embeddings = update_embeddings

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(self.model_path).to(self.device)

        self.cap = cv2.VideoCapture(self.video_path)
        self.frame_counter = -1

        # ReID model
        self.reid_model = ReIDModel(self.device)

        # Local track data
        self.tracks = {}
        self.last_xyxy = []
        self.last_ids = []

        # Where we display frames in the GUI
        self.display_label = display_label

        # For pause/resume
        self.video_ended = False

    def process_frame(self, frame):
        """Run YOLO tracking + ReID on a frame."""
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

                # Crop & get embedding
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
                    if (self.update_embeddings and
                                self.frame_counter -
                            track_data["last_update_frame"]
                            ) >= self.embedding_update_interval:
                        new_emb = self.reid_model.get_embedding(crop)
                        self.global_manager.update_embedding(
                            track_data["global_id"], new_emb)
                        track_data["embedding"] = new_emb
                        track_data["last_update_frame"] = self.frame_counter

            # Remove tracks not seen in this frame
            active_ids = set(self.last_ids)
            self.tracks = {
                tid: data
                for tid, data in self.tracks.items()
                if tid in active_ids
            }

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

        return self.draw_last_boxes(frame)

    def draw_last_boxes(self, frame):
        """Draw bounding boxes + global IDs on the frame."""
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

    def read_and_process_one_frame(self):
        """
        Reads one frame, processes it if needed,
        then updates the display label (if provided).
        Returns True if a frame was read successfully, else False.
        """
        if not self.cap.isOpened() or self.video_ended:
            return False

        success, frame = self.cap.read()
        if not success:
            # End of video or error
            self.video_ended = True
            self.cap.release()
            return False

        self.frame_counter += 1
        frame = cv2.resize(frame, self.resized_shape)

        if self.frame_counter % self.skip_interval == 0:
            processed_frame = self.process_frame(frame)
        else:
            processed_frame = self.draw_last_boxes(frame)

        if self.display_label is not None:
            self._display_in_label(processed_frame)

        return True

    def _display_in_label(self, frame_bgr):
        """Convert BGR -> QImage -> QPixmap -> put in QLabel."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(
            frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
        )
        pixmap = QPixmap.fromImage(qimg)
        self.display_label.setPixmap(pixmap)

    def release(self):
        if self.cap.isOpened():
            self.cap.release()


###############################################################################
# 4) MultiVideoProcessor to handle multiple YOLOVideoTracker
###############################################################################
class MultiVideoProcessor:
    def __init__(
            self,
            video_paths,
            model_path='yolov8n.pt',
            skip_interval=5,
            resized_shape=(640, 360),
            embedding_update_interval=60,
            similarity_threshold=0.3,
            update_embeddings=True,
            display_labels=None
    ):
        """
        :param display_labels: list of QLabel, one for each video path.
        """
        self.video_paths = video_paths
        self.model_path = model_path
        self.skip_interval = skip_interval
        self.resized_shape = resized_shape
        self.embedding_update_interval = embedding_update_interval

        # State flags
        self.paused = False
        self.stopped = False

        # Global manager shared across all trackers
        self.global_manager = GlobalIDManager(
            similarity_threshold=similarity_threshold)

        self.global_frame_counter = 0

        # If user didn't pass labels, we'll store None for each
        if display_labels is None:
            display_labels = [None] * len(video_paths)

        # Create YOLOVideoTracker objects
        self.trackers = []
        for vp, lbl in zip(video_paths, display_labels):
            t = YOLOVideoTracker(
                video_path=vp,
                model_path=self.model_path,
                skip_interval=self.skip_interval,
                resized_shape=self.resized_shape,
                global_manager=self.global_manager,
                embedding_update_interval=self.embedding_update_interval,
                display_label=lbl,
                update_embeddings=update_embeddings,
            )
            self.trackers.append(t)

    def run_one_cycle(self):
        """
        Call this periodically (e.g. via QTimer) to process one frame from each tracker.
        """
        if self.stopped:
            return
        if self.paused:
            return

        any_frame_ok = False
        for tracker in self.trackers:
            ok = tracker.read_and_process_one_frame()
            if ok:
                any_frame_ok = True
                self.global_frame_counter += 1

        if not any_frame_ok:
            self.stop()

    def stop(self):
        self.stopped = True
        for tracker in self.trackers:
            tracker.release()

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def reset(self):
        """
        Stop everything, clear the global manager, and re-init trackers.
        This is effectively "start from scratch."
        """
        self.stop()
        self.global_manager.reset()
        self.trackers = []
        # Re-init trackers (labels won't change unless you want them to)
        # If you want to truly "restart," you'd open new captures or re-create with the same paths:
        # self.__init__(...)
        # but watch out for re-creating if you have the same QLabels.
        # Example:
        # self.__init__(video_paths=self.video_paths, model_path=self.model_path, ...)
        # For now we'll do nothing or re-init if desired.


###############################################################################
# 5) A Simple PyQt6 Main Window to demonstrate multiple videos
###############################################################################
class MainWindow(QMainWindow):
    def __init__(self, video_paths):
        super().__init__()
        self.setWindowTitle("Multi-Camera YOLO + ReID (PyQt6)")

        # Main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # A grid (or horizontal box) to hold multiple video labels
        self.video_grid = QGridLayout()
        main_layout.addLayout(self.video_grid)

        # Create a label for each video
        self.labels = []
        for i, vp in enumerate(video_paths):
            label = QLabel(f"Video: {Path(vp).name}")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setFixedSize(640, 360)  # or adapt to your needs
            self.video_grid.addWidget(label, i // 2, i %
                                      2)  # for a 2-column layout
            self.labels.append(label)

        # Buttons for controlling playback
        button_layout = QHBoxLayout()
        main_layout.addLayout(button_layout)

        self.btn_pause = QPushButton("Pause")
        self.btn_pause.clicked.connect(self.pause_videos)
        button_layout.addWidget(self.btn_pause)

        self.btn_resume = QPushButton("Resume")
        self.btn_resume.clicked.connect(self.resume_videos)
        button_layout.addWidget(self.btn_resume)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self.stop_videos)
        button_layout.addWidget(self.btn_stop)

        # Create multi-video processor
        self.multiproc = MultiVideoProcessor(
            video_paths=video_paths,
            model_path='yolov8n.pt',
            skip_interval=5,
            resized_shape=(640, 360),
            embedding_update_interval=60,
            similarity_threshold=0.3,
            display_labels=self.labels
        )

        # QTimer to periodically process one frame from each video
        self.timer = QTimer()
        self.timer.timeout.connect(self.multiproc.run_one_cycle)
        # ~20 FPS is 50 ms interval
        self.timer.start(15)

    def pause_videos(self):
        self.multiproc.pause()

    def resume_videos(self):
        self.multiproc.resume()

    def stop_videos(self):
        self.multiproc.stop()

    def closeEvent(self, event):
        # Cleanup resources
        self.multiproc.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)

    # Replace these with your multiple video paths
    video_paths = [
        'videos/hd_00_00.mp4',
        # 'videos/hd_00_01.mp4',
        # 'videos/hd_00_02.mp4',
        'videos/hd_00_03.mp4'
    ]

    window = MainWindow(video_paths)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
