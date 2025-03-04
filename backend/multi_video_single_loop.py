import cv2
import asyncio
from yolo_tracker import YOLOVideoTracker

class MultiVideoSingleLoop:
    def __init__(
        self, 
        video_paths, 
        sio, 
        model_path='yolov8n.pt', 
        skip_interval=5, 
        resized_shape=(640, 360)
    ):
     
        self.video_paths = video_paths
        self.sio = sio
        self.model_path = model_path
        self.skip_interval = skip_interval
        self.resized_shape = resized_shape
        
        # Control flags
        self.paused = False
        self.stopped = False

        # Build initial trackers
        self._init_trackers()

    def _init_trackers(self):
        """Helper to create a YOLOVideoTracker for each path."""
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
                    # No more frames
                    tracker.cap.release()
                    continue

                any_frame_ok = True
                tracker.frame_counter += 1

                # Resize
                frame = cv2.resize(frame, tracker.resized_shape)

                # Decide whether to run YOLO or just reuse old boxes
                if tracker.frame_counter % tracker.skip_interval == 0:
                    # Run actual YOLO detection, draw bounding boxes, and store them
                    processed_frame = tracker.process_frame(frame)
                else:
                    # Reuse last boxes on this new frame (no new detection)
                    processed_frame = tracker.draw_last_boxes(frame)

                # Send every frame to the frontend
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
        """Pause the loop (no new frames are read)."""
        self.paused = True

    def resume(self):
        """Resume the loop (start reading frames again)."""
        self.paused = False

    async def reset(self):
        """
        Reset videos to frame=0 by:
          1) Stopping the current loop.
          2) Re-init trackers.
          3) Starting run() again in a fresh task.
        """
        # Signal the current loop to stop
        self.stopped = True
        # Let the loop exit gracefully
        await asyncio.sleep(0.1)

        # Re-initialize trackers
        self._init_trackers()

        # Clear flags
        self.stopped = False
        self.paused = False

        # Start again
        asyncio.create_task(self.run())
