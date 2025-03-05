import sys
import cv2
import torch
import numpy as np
import json
from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QPushButton, QGridLayout, QScrollArea
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer, Qt
from ultralytics import YOLO
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms


def cosine_similarity(a, b):
    """Compute cosine similarity between two 1D numpy arrays."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)


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


class VideoProcessor:
    def __init__(self, video_path, camera_params, reid_model, device, model_path="yolov8n.pt"):
        self.device = device
        # Instantiate YOLO without the device parameter
        self.model = YOLO(model_path)

        self.model.model.to(self.device)

        self.cap = cv2.VideoCapture(video_path)

        # Load camera parameters
        self.K = np.array(camera_params['K'])
        self.distCoef = np.array(camera_params['distCoef'])
        self.R = np.array(camera_params['R'])
        self.t = np.array(camera_params['t'])

        self.reid_model = reid_model  # Use a shared ReID model

    def undistort_frame(self, frame):
        return cv2.undistort(frame, self.K, self.distCoef)

    def transform_point(self, point):
        """
        Transform a 2D point using camera extrinsics with a division by zero check.
        Returns None if the transformation cannot be performed.
        """
        point_h = np.array([point[0], point[1], 1]).reshape(3, 1)
        world_point = np.dot(self.R, point_h) + self.t
        denom = world_point[2, 0]
        # Check for division by zero or near-zero denominator
        if np.abs(denom) < 1e-6:
            return None
        return world_point[:2] / denom

    def process_frame(self):
        if not self.cap.isOpened():
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None  # End of video

        # Undistort the frame once
        frame = self.undistort_frame(frame)

        # Run detection and tracking with YOLO
        results = self.model.track(frame, persist=True, verbose=False)
        detections = []
        if results and results[0].boxes:
            for i, box in enumerate(results[0].boxes.xyxy):
                x1, y1, x2, y2 = map(int, box[:4])
                track_id = results[0].boxes.id[i] if results[0].boxes.id is not None else i
                world_pos = self.transform_point(
                    [(x1 + x2) // 2, (y1 + y2) // 2])
                if world_pos is None:
                    continue  # Skip this detection if transformation fails

                world_pos_x = world_pos[0, 0]
                world_pos_y = world_pos[1, 0]

                # Draw bounding box and basic info on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'ID: {int(track_id)}, X:{world_pos_x:.3f}, Y:{world_pos_y:.3f}',
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Crop the detection region for ReID embedding extraction.
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                embedding = self.reid_model.get_embedding(crop)
                detections.append({
                    "track_id": track_id,
                    "bbox": (x1, y1, x2, y2),
                    "embedding": embedding,
                    "world_pos": (world_pos_x, world_pos_y)
                })

        # Optionally resize the frame for display
        frame = cv2.resize(frame, (1280, 720))
        # Return both the processed frame and the list of detections with their embeddings.
        return frame, detections

    def release(self):
        if self.cap:
            self.cap.release()


class ObjectTrackingApp(QWidget):
    def __init__(self, video_paths, calibration_data, reid_model, device):
        super().__init__()
        self.video_paths = video_paths
        self.calibration_data = calibration_data
        self.reid_model = reid_model
        self.device = device
        self.initUI()

        self.processors = [
            VideoProcessor(path, calibration_data[i], reid_model, self.device)
            for i, path in enumerate(video_paths)
        ]
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frames)

    def initUI(self):
        self.setWindowTitle("YOLO Object Tracking with ReID - PyQt6")
        self.setGeometry(100, 100, 1920, 1080)

        main_layout = QVBoxLayout()
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        main_layout.addWidget(self.scroll_area)

        self.video_widget = QWidget()
        self.video_layout = QGridLayout()
        self.video_widget.setLayout(self.video_layout)
        self.scroll_area.setWidget(self.video_widget)

        self.video_labels = []
        for i, _ in enumerate(self.video_paths):
            label = QLabel(self)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            row, col = divmod(i, 2)
            self.video_layout.addWidget(label, row, col)
            self.video_labels.append(label)

        self.start_button = QPushButton("Start Tracking", self)
        self.start_button.clicked.connect(self.start_tracking)
        main_layout.addWidget(self.start_button)

        self.setLayout(main_layout)

    def start_tracking(self):
        if any(proc.cap.isOpened() for proc in self.processors):
            self.timer.start(30)

    def update_frames(self):
        all_detections = []
        # Process each camera stream individually
        for i, processor in enumerate(self.processors):
            ret = processor.process_frame()
            if ret is not None:
                frame, detections = ret
                self.display_frame(frame, self.video_labels[i])
                # Tag each detection with its camera index for cross-camera matching.
                for det in detections:
                    det["camera_index"] = i
                    all_detections.append(det)
        # Now perform cross-camera matching based on the ReID embeddings.
        matches = self.match_detections(all_detections, threshold=0.5)
        # For demonstration, print out matches (or integrate with your tracking logic)
        for det1, det2, sim in matches:
            print(f"Match found between Cam {det1['camera_index']} (ID {det1['track_id']}) "
                  f"and Cam {det2['camera_index']} (ID {det2['track_id']}) with similarity {sim:.3f}")

    def match_detections(self, detections, threshold=0.5):
        """
        Compare all detections across cameras using cosine similarity.
        Returns a list of tuples: (detection_from_cam_A, detection_from_cam_B, similarity)
        """
        matches = []
        n = len(detections)
        for i in range(n):
            for j in range(i + 1, n):
                # Only compare detections from different cameras.
                if detections[i]["camera_index"] != detections[j]["camera_index"]:
                    sim = cosine_similarity(
                        detections[i]["embedding"], detections[j]["embedding"])
                    if sim > threshold:
                        matches.append((detections[i], detections[j], sim))
        return matches

    def display_frame(self, frame, label):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame_rgb.shape
        bytes_per_line = 3 * width
        qimg = QImage(frame_rgb.data, width, height,
                      bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        label.setPixmap(pixmap)

    def closeEvent(self, event):
        for processor in self.processors:
            processor.release()
        event.accept()


if __name__ == "__main__":
    # Determine device to use based on CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load calibration data (ensure the file exists and is properly formatted)
    with open("calibration/calibration.json", "r") as f:
        calib_data = json.load(f)

    video_paths = ["videos/hd_0.mp4", "videos/hd_3.mp4"]
    calibration_data = calib_data['cameras']

    # Initialize the ReID model with the appropriate device
    reid_model = ReIDModel(device=device)

    app = QApplication(sys.argv)
    window = ObjectTrackingApp(
        video_paths, calibration_data, reid_model, device)
    window.show()
    sys.exit(app.exec())
