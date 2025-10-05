# ball_detector.py

import numpy as np
import cv2
import math

def detect_ball(frame):
    """
    Detects the red ball in the frame, filters out non-circular/large objects, 
    and returns its center, radius, and the final binary mask.

    :param frame: The input frame (expected to be already resized and stabilized).
    :return: A tuple (center, radius, mask) or (None, 0, None) if no valid ball is found.
    """
    
    # --- HSV Boundaries for RED Color (Two ranges for robust red detection) ---
    redLower1 = (0, 100, 50)
    redUpper1 = (10, 255, 255)
    redLower2 = (170, 100, 50)
    redUpper2 = (179, 255, 255)

    # 1. Blur and Convert to HSV
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # 2. Combine Masks
    mask1 = cv2.inRange(hsv, redLower1, redUpper1)
    mask2 = cv2.inRange(hsv, redLower2, redUpper2)
    mask = cv2.bitwise_or(mask1, mask2)
    
    # 3. Clean up the mask (Erosion then Dilation)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # 4. Find Contours
    cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_c = None
    max_area = 0

    # 5. Filter Contours (Size and Circularity check)
    for c in cnts:
        area = cv2.contourArea(c)
        
        if area < 100: # Ignore small noise
            continue
            
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        
        # Use a safe check to prevent division by zero
        if radius == 0:
            continue
            
        circle_area = math.pi * (radius ** 2)
        
        # Check 1: Circularity (ratio of contour area to enclosing circle area > 0.7)
        if area / circle_area > 0.7:
            # Check 2: Max size (to ignore the large t-shirt, area < 8000 is an estimate)
            if area > max_area and area < 8000: 
                max_area = area
                best_c = c

    if best_c is not None:
        ((x, y), radius) = cv2.minEnclosingCircle(best_c)
        M = cv2.moments(best_c)
        
        if M["m00"] != 0:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            return center, radius, mask
    
    return None, 0, mask # Return mask even if detection fails for visualization