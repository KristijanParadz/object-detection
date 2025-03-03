import cv2
import torch
from ultralytics import YOLO

class YOLOVideoTracker:
    def __init__(self, video_path, model_path='yolov8n.pt', skip_interval=3, resized_shape=(990, 540)):
        self.video_path = video_path
        self.model_path = model_path
        self.skip_interval = skip_interval
        self.resized_shape = resized_shape
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        self.model = YOLO(self.model_path).to(self.device)
        self.cap = cv2.VideoCapture(self.video_path)
        self.frame_counter = -1
    
    def process_frame(self, frame):
        frame = cv2.resize(frame, self.resized_shape)
        results = self.model.track(frame, persist=True, verbose=False, device=self.device)
        
        if results and results[0].boxes:
            for i, box in enumerate(results[0].boxes.xyxy):
                x1, y1, x2, y2 = map(int, box[:4])
                track_id = results[0].boxes.id[i] if results[0].boxes.id is not None else i
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(frame, f'ID: {int(track_id)}', 
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 0), 1)
        return frame
    
    def run(self):
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                break
            
            self.frame_counter += 1
            if self.frame_counter % self.skip_interval != 0:
                cv2.imshow("YOLOv8 Tracking", frame)
                continue
            
            frame = self.process_frame(frame)
            cv2.imshow("YOLOv8 Tracking", frame)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    video_tracker = YOLOVideoTracker(video_path='videos/hd_0.mp4')
    video_tracker.run()
