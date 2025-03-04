import cv2
from yolo_tracker import YOLOVideoTracker

class MultiVideoSingleLoop:
    def __init__(
        self, 
        video_paths, 
        sio, 
        model_path='yolov8n.pt', 
        skip_interval=3, 
        resized_shape=(640, 360)
    ):
     
        self.video_paths = video_paths
        self.sio = sio
        self.model_path = model_path
        self.skip_interval = skip_interval
        self.resized_shape = resized_shape
        
        
        self.trackers = []
        for vp in self.video_paths:
            tracker = YOLOVideoTracker(
                video_path=vp, 
                sio=self.sio,
                model_path=self.model_path,
                skip_interval=self.skip_interval,
                resized_shape=self.resized_shape
            )
            self.trackers.append(tracker)

    async def run(self):
        """
        Single loop that attempts to read from each video tracker 
        and processes frames until all are done.
        """
        while True:
            any_frame_ok = False

            # Loop over each tracker
            for tracker in self.trackers:
                cap = tracker.cap
                if not cap.isOpened():
                    continue  # This video is finished
                
                success, frame = cap.read()
                if not success:
                    # End of this video
                    cap.release()
                    continue
                
                any_frame_ok = True
                
                # Update the tracker's frame counter
                tracker.frame_counter += 1
                
                # Resize frame
                frame = cv2.resize(frame, tracker.resized_shape)
                
                # Check skip interval
                if tracker.frame_counter % tracker.skip_interval != 0:
                    # Optionally emit the raw frame or do nothing
                    continue
                
                # Process frame with YOLO
                processed_frame = tracker.process_frame(frame)

                
                # Emit with the video_id so the frontend can tell them apart
                await tracker.send_image_to_frontend(processed_frame)

            # If we couldn't read a frame from any video, break out
            if not any_frame_ok:
                break
