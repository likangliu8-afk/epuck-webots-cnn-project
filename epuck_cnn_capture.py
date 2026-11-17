# Applied Robotics and AI Project - Assignment 2 (Part 2)
# Controller: e-puck_search.py
# Description: Built-in HSV red-frame target detection (filters out brown wooden floor),
#              visual homing, auto-centering, and camera image capture for CNN prediction.

from controller import Robot
import colorsys
import os
import random

TIME_STEP = 32
MAX_SPEED = 6.28
DIST_LIMIT = 75.0

# Require the red frame to occupy >= 50% of camera width before stopping to take photo
MIN_FRAME_WIDTH_RATIO = 0.50

robot = Robot()

# Initialize wheel motors
leftMotor = robot.getDevice('left wheel motor')
rightMotor = robot.getDevice('right wheel motor')
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))
leftMotor.setVelocity(0.0)
rightMotor.setVelocity(0.0)

# Enable 8 IR proximity sensors (ps0 - ps7)
ps = []
for i in range(8):
    s = robot.getDevice(f'ps{i}')
    s.enable(TIME_STEP)
    ps.append(s)

# Enable camera (configured to 128x128 in Scene Tree)
cam = robot.getDevice('camera')
cam.enable(TIME_STEP)
W = cam.getWidth()
H = cam.getHeight()

# Create 'images' folder next to this script
SAVE_FOLDER = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(SAVE_FOLDER, exist_ok=True)

image_count = 0
captured_flag = False


def detect_red_frame_hsv():
    """
    Uses Python's built-in colorsys HSV conversion to detect ONLY the vivid pure-red
    board border while completely ignoring the brown wooden checkerboard floor.
    Returns:
      (found, width_ratio, horiz_bias)
    """
    img = cam.getImage()
    if img is None:
        return False, 0.0, None

    red_count = 0
    min_x = W
    max_x = 0
    sum_x = 0.0

    for x in range(0, W, 2):
        for y in range(0, H, 2):
            r = cam.imageGetRed(img, W, x, y)
            g = cam.imageGetGreen(img, W, x, y)
            b = cam.imageGetBlue(img, W, x, y)

            # Quick pre-check before HSV conversion for fast execution
            if r > 90 and g < 45 and b < 45:
                h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
                # Pure red: Hue near 0 or 1, Saturation >= 0.82 (wooden floor is ~0.58)
                if (h <= 0.04 or h >= 0.96) and s >= 0.82 and v >= 0.35:
                    red_count += 1
                    sum_x += x
                    if x < min_x:
                        min_x = x
                    if x > max_x:
                        max_x = x

    # Require at least 12 vivid red pixels so distant noise is ignored
    if red_count < 12:
        return False, 0.0, None

    box_width = max_x - min_x
    width_ratio = box_width / float(W)
    avg_x = sum_x / float(red_count)
    horiz_bias = (avg_x - (W / 2.0)) / (W / 2.0)

    return True, width_ratio, horiz_bias


def save_camera_photo():
    """Saves high-quality JPEG directly from the e-puck onboard camera."""
    global image_count
    image_count += 1
    filepath = os.path.join(SAVE_FOLDER, f"webots_full_{image_count:03d}.jpg")
    status = cam.saveImage(filepath, 100)
    if status == 0:
        print(f"SUCCESS! Saved e-puck camera photo to: {filepath}", flush=True)
    else:
        print("Failed to save camera image.", flush=True)


def stop():
    leftMotor.setVelocity(0.0)
    rightMotor.setVelocity(0.0)


def main():
    global captured_flag
    turn_steps = 0
    turn_dir = 1
    settle_steps = 0

    print(f"E-puck HSV controller started (Camera: {W}x{H}).")
    print("Searching for vivid red-framed frog board (wooden floor filtered out)...", flush=True)

    while robot.step(TIME_STEP) != -1:
        if captured_flag:
            stop()
            continue

        found, width_ratio, bias = detect_red_frame_hsv()

        # Priority 1: Reached close range to the red-framed board (width >= 50% of view)
        if found and width_ratio >= MIN_FRAME_WIDTH_RATIO:
            # Fine-tune rotation until the frog board is dead-center in the camera
            if abs(bias) > 0.08:
                settle_steps = 0
                spin = 0.18 * MAX_SPEED if bias > 0 else -0.18 * MAX_SPEED
                leftMotor.setVelocity(spin)
                rightMotor.setVelocity(-spin)
            else:
                stop()
                settle_steps += 1
                # Wait 3 steps after stopping so camera frame is completely steady
                if settle_steps >= 3:
                    print(f"Frog board centered at close range (width={width_ratio*100:.1f}%)! Taking photo...", flush=True)
                    save_camera_photo()
                    captured_flag = True
            continue

        settle_steps = 0

        # Priority 2: Visual Homing if red board is visible in camera FOV
        if found and bias is not None:
            base = 0.60 * MAX_SPEED
            steer = 0.45 * bias * MAX_SPEED
            leftMotor.setVelocity(max(-MAX_SPEED, min(MAX_SPEED, base + steer)))
            rightMotor.setVelocity(max(-MAX_SPEED, min(MAX_SPEED, base - steer)))
            continue

        # Priority 3: IR Obstacle Avoidance & Arena Scanning
        right_blk = ps[0].getValue() > DIST_LIMIT or ps[1].getValue() > DIST_LIMIT or ps[2].getValue() > DIST_LIMIT
        left_blk = ps[7].getValue() > DIST_LIMIT or ps[6].getValue() > DIST_LIMIT or ps[5].getValue() > DIST_LIMIT

        if turn_steps > 0:
            turn_steps -= 1
            leftMotor.setVelocity(-turn_dir * 0.55 * MAX_SPEED)
            rightMotor.setVelocity(turn_dir * 0.55 * MAX_SPEED)
        elif right_blk:
            turn_dir = 1
            turn_steps = 12
        elif left_blk:
            turn_dir = -1
            turn_steps = 12
        else:
            # Gentle arc patrol so the camera sweeps across the entire arena quickly
            leftMotor.setVelocity(0.65 * MAX_SPEED)
            rightMotor.setVelocity(0.40 * MAX_SPEED)


if __name__ == '__main__':
    main()
