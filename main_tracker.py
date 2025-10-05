# main_tracker.py

import argparse
from collections import deque
import cv2
import numpy as np

# Import the custom modules
from ball_detector import detect_ball
from perspective_mapper import PerspectiveMapper
from vidstab import VidStab # <-- VIDSTAB IMPORTED

# Constants
FRAME_WIDTH = 600
SMOOTHING_WINDOW = 30 # Default smoothing recommended by vidstab

def draw_trajectory(frame, pts, color=(0, 0, 255), buffer_size=64):
    """Draws the trajectory trail on a frame."""
    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            continue
        # Line thickness decreases over time
        thickness = int(np.sqrt(buffer_size / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], color, thickness)


def main():
    # --- Setup Arguments ---
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", required=True,
        help="path to the video file (e.g., rgb.avi)")
    ap.add_argument("-b", "--buffer", type=int, default=64,
        help="max buffer size for tracking past points")
    args = vars(ap.parse_args())

    # --- Initialization ---
    vs = cv2.VideoCapture(args["video"])
    if not vs.isOpened():
        print(f"[ERROR] Could not open video file: {args['video']}")
        return

    # 1. Initialize the perspective mapper
    mapper = PerspectiveMapper(frame_width=FRAME_WIDTH)
    
    # 2. Initialize the video stabilizer
    stabilizer = VidStab() 

    # 3. Initialize tracking lists for both views
    pts_original = deque(maxlen=args["buffer"])
    pts_warped = deque(maxlen=args["buffer"])
    
    print(f"[INFO] Starting video processing for: {args['video']}...")

    # --- Main Loop ---
    while True:
        ret, frame = vs.read()
        if not ret:
            break

        # Pre-resize the frame for consistent processing and mapping
        height, width, _ = frame.shape
        scale = FRAME_WIDTH / float(width)
        dim = (FRAME_WIDTH, int(height * scale))
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        # --- STEP 1: STABILIZATION ---
        frame_stabilized = stabilizer.stabilize_frame(
            input_frame=frame, 
            smoothing_window=SMOOTHING_WINDOW
        )

        if frame_stabilized is None:
            # Skip frame until the stabilizer is warmed up
            continue 

        # --- STEP 2: PERSPECTIVE WARPING ---
        # Perspective transform is applied to the stabilized frame
        frame_warped = mapper.warp_frame(frame_stabilized)


        # --- TEMPORARY DEBUG VISUALIZATION START ---
        # Get the source points from the mapper
        pts_src_int = np.int32(mapper.pts_src) 

        # Draw the polygon outline
        cv2.polylines(frame, [pts_src_int], isClosed=True, color=(0, 255, 0), thickness=2)

        # Draw circles at each corner for clarity (p1: Red, p2: Blue, p3: Green, p4: Yellow)
        colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (0, 255, 255)]
        for i, pt in enumerate(pts_src_int):
            cv2.circle(frame, tuple(pt), 5, colors[i], -1)
        # --- TEMPORARY DEBUG VISUALIZATION END ---

        # 3. BALL DETECTION (on the stabilized frame)
        center, radius, mask = detect_ball(frame_stabilized)
        
        # 4. DRAWING AND TRAJECTORY
        center_warped = None

        if center is not None:
            # Draw detection on the STABILIZED frame
            x, y = center
            cv2.circle(frame_stabilized, (int(x), int(y)), int(radius), (0, 255, 255), 2) # Yellow circle
            cv2.circle(frame_stabilized, center, 5, (0, 0, 255), -1) # Red dot
            
            # Transform point for the warped view
            center_warped = mapper.transform_point(center)
            
            # Draw the current position on the warped view
            if center_warped is not None:
                cv2.circle(frame_warped, center_warped, 5, (0, 255, 255), -1)

        # 5. TRAJECTORY UPDATES
        pts_original.appendleft(center)
        pts_warped.appendleft(center_warped)
        
        # Draw trajectories
        draw_trajectory(frame_stabilized, pts_original, color=(0, 0, 255), buffer_size=args["buffer"])
        draw_trajectory(frame_warped, pts_warped, color=(0, 0, 255), buffer_size=args["buffer"])

        # Display results in three separate windows
        cv2.imshow("1. STABILIZED View (Ball + Trajectory)", frame_stabilized)
        cv2.imshow("2. 2D Top-Down Map View (Stabilized)", frame_warped)
        cv2.imshow("3. Raw Detection Mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # Cleanup
    vs.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()