import cv2
import torch
from ultralytics import YOLO

# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load the YOLOv8 model on the appropriate device
model = YOLO('yolov8n.pt').to(device)  # Load a pretrained YOLOv8n model

# Open the video file
video_path = 'videos/hd_0.mp4'  # Replace with your video file path
cap = cv2.VideoCapture(video_path)
skip_interval = 3
frame_counter = -1
resized_shape = (1980 // 2, 1080 // 2)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if not success:
        break
    
    frame = cv2.resize(frame, resized_shape)
    frame_counter += 1

    if frame_counter % skip_interval != 0:
        cv2.imshow("YOLOv8 Tracking", frame)
        continue

    # Convert frame to tensor and move to the selected device
    results = model.track(frame, persist=True, verbose=False, device=device)

    # Visualize the results on the frame
    if results and results[0].boxes:
        for i, box in enumerate(results[0].boxes.xyxy):
            x1, y1, x2, y2 = map(int, box[:4])
            track_id = results[0].boxes.id[i] if results[0].boxes.id is not None else i

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(frame, f'ID: {int(track_id)}', 
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 0), 1)

    cv2.imshow("YOLOv8 Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()