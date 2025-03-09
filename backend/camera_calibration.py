import cv2
import numpy as np
import json


class CameraCalibration:
    def __init__(self, calib_file):
        """ Load calibration data from JSON. """
        with open(calib_file, 'r') as f:
            self.calib_data = json.load(f)

        self.cameras = {}
        for cam in self.calib_data["cameras"]:
            name = f"{cam['type']}_{cam['name']}"
            self.cameras[name] = {
                "K": np.array(cam["K"]),
                "distCoef": np.array(cam["distCoef"]),
                "R": np.array(cam["R"]),
                "t": np.array(cam["t"]),
            }

    def image_to_world(self, cam_name, bbox_center):
        """
        Converts a bounding box center from image space to global world coordinates on the ground plane (Z=0).
        """
        if cam_name not in self.cameras:
            print(f"Camera {cam_name} not found in calibration data!")
            return None

        cam = self.cameras[cam_name]
        K, distCoef, R, t = cam["K"], cam["distCoef"], cam["R"], cam["t"]

        # Undistort pixel coordinates
        undistorted_pts = cv2.undistortPoints(
            np.array([[[bbox_center[0], bbox_center[1]]]], dtype=np.float32),
            K, distCoef, None, K
        )
        u, v = undistorted_pts[0][0]

        # Compute the ray in camera coordinates
        ray = np.linalg.inv(K) @ np.array([[u], [v], [1]])

        # Invert rotation matrix once for later use
        R_inv = np.linalg.inv(R)

        # Transform ray and translation into world space
        w = R_inv @ ray
        d = R_inv @ t

        # Compute scale factor such that the world Z coordinate becomes 0
        if w[2][0] == 0:
            print(
                "Invalid ray: division by zero encountered when computing scale factor.")
            return None
        s = d[2][0] / w[2][0]

        # Compute world coordinates with proper scaling
        world_coords = R_inv @ (s * ray - t)

        return world_coords[0][0], world_coords[1][0]


class CalibrationParameters:
    def __init__(self, K, distCoef, R, t):
        self.K = K
        self.distCoef = distCoef
        self.R = R
        self.t = t
