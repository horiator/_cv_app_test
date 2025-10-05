# perspective_mapper.py

import numpy as np
import cv2

class PerspectiveMapper:
    """
    Handles the calculation and application of the 
    perspective transform for a top-down view.
    
    NOTE: The pts_src coordinates MUST be adjusted based on your video 
    to accurately trace a rectangle on the floor in the 600px wide frame.
    """
    
    def __init__(self, frame_width=600):
        
        """
        Initializes the mapper by calculating the transformation matrix (M).
        """
        self.WARPED_WIDTH = 400
        #  HEIGHT (e.g., to 900) to map the area 
        # from the ball's start position (Y=590) up to the far end (Y=300).
        self.WARPED_HEIGHT = 900 
        
        # --- Source Points (ADJUSTED TO CAPTURE BALL'S START AREA) ---
        # The far points (Y=300) define the TOP of the map (distant gray area).
        # The near points (Y=590) define the BOTTOM of the map (where the ball starts).
        # Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
        self.pts_src = np.float32([
            [220, 100],  # Far Left Corner (TOP of the 2D map)
            [420, 100],  # Far Right Corner 
            [500, 590],  # Near Right Corner (Lower Y-coordinate to reach ball's start)
            [140, 590]   # Near Left Corner (Lower Y-coordinate to reach ball's start)
        ])
        
        # NOTE: I widened the near points (500 and 140) slightly to keep the lines parallel 
        # as they approach the bottom of the frame. You MUST adjust these four points 
        # using the debug visualization to match the physical rectangle on the floor!

        # --- Destination Points (Perfect Rectangle - Based on new WARPED_HEIGHT) ---
        self.pts_dst = np.float32([
            [0, 0],
            [self.WARPED_WIDTH - 1, 0],
            [self.WARPED_WIDTH - 1, self.WARPED_HEIGHT - 1],
            [0, self.WARPED_HEIGHT - 1]
        ])


        # Calculate the perspective transform matrix (M)
        self.M = cv2.getPerspectiveTransform(self.pts_src, self.pts_dst)

    def warp_frame(self, frame):
        """Applies the perspective transformation to an image frame."""
        return cv2.warpPerspective(frame, self.M, (self.WARPED_WIDTH, self.WARPED_HEIGHT))

    def transform_point(self, point):
        """Applies the perspective transform matrix to a single point (x, y)."""
        if point is None:
            return None
        
        # Convert point to homogeneous coordinates (x, y, 1)
        src_pt = np.float32([[[point[0], point[1]]]])
        
        # Apply the perspective transform
        dst_pt = cv2.perspectiveTransform(src_pt, self.M)
        
        # Return the new point (x', y')
        return (int(dst_pt[0][0][0]), int(dst_pt[0][0][1]))