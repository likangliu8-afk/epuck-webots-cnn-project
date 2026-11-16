"""
Controller Name: e-puck_search.py
Course: Applied Robotics and AI Project (Assignment 2 - Project Part 1)
Description:
    Autonomous e-puck controller for searching and detecting red, blue, and green
    blocks (0.3m x 0.3m x 0.1m) in a Webots simulation environment.
    Features:
      1. Reactive infrared obstacle avoidance (ps0 - ps7) + escape backing.
      2. Long-range visual target homing + close-range ROI RGB encounter confirmation.
      3. First-encounter message output ("I see red/blue/green") and encounter summary.
      4. Search duration timer for each encounter and total time to find all 3 blocks.
      5. Multi-round re-search support: pauses for a standby window after finding all
         3 blocks so the user can pause, reposition blocks, and run subsequent rounds.
"""

import random
from controller import Robot

# ==============================================================================
# 1. Constants and Hyperparameters
# ==============================================================================
TIME_STEP = 64                  # Basic simulation step in milliseconds
MAX_SPEED = 6.28                # Maximum angular velocity of e-puck wheels (rad/s)
CRUISE_SPEED = 0.80 * MAX_SPEED # Standard forward exploration speed

# Infrared proximity sensor threshold for obstacle detection
OBSTACLE_THRESHOLD = 75.0

# Color detection thresholds (RGB space, 0-255, shadow-tolerant)
MIN_CHANNEL_INTENSITY = 45       # Allows detection on shaded faces of the blocks
DOMINANCE_RATIO = 2.0            # Dominant channel must be >= 2.0x the other two channels
ROI_MATCH_RATIO_THRESHOLD = 0.38 # Requires >=38% of central ROI (robot must be close, ~0.15m-0.25m)

# Standby duration (in simulation seconds) before starting the next search round
ROUND_STANDBY_DURATION = 5.0


def classify_pixel_color(r, g, b):
    """
    Classifies a single RGB pixel into 'red', 'blue', 'green', or None
    based on channel intensity and relative dominance ratio.
    """
    if r > MIN_CHANNEL_INTENSITY and r > DOMINANCE_RATIO * g and r > DOMINANCE_RATIO * b:
        return "red"
    elif b > MIN_CHANNEL_INTENSITY and b > DOMINANCE_RATIO * r and b > DOMINANCE_RATIO * g:
        return "blue"
    elif g > MIN_CHANNEL_INTENSITY and g > DOMINANCE_RATIO * r and g > DOMINANCE_RATIO * b:
        return "green"
    return None


def analyze_camera_frame(camera, width, height, encountered_blocks):
    """
    Scans the camera image to:
      1. Confirm a close-range block encounter when an unvisited block fills >= 38%
         of the central Region of Interest (ROI).
      2. Locate the horizontal offset of any unvisited target block seen at a distance
         to guide the robot toward it (visual homing).
    Returns:
      (detected_color, homing_bias)
    """
    image = camera.getImage()
    if image is None:
        return None, None

    # Central Region of Interest (ROI) for confirming close-range block encounter
    x_roi_start = int(width * 0.20)
    x_roi_end = int(width * 0.80)
    y_roi_start = int(height * 0.15)
    y_roi_end = int(height * 0.85)

    roi_color_counts = {"red": 0, "blue": 0, "green": 0}
    total_roi_pixels = 0

    # Visual homing accumulators for unvisited blocks across the entire frame
    unvisited_x_sum = 0.0
    unvisited_pixel_count = 0

    for x in range(0, width, 2):
        for y in range(0, height, 2):
            r = camera.imageGetRed(image, width, x, y)
            g = camera.imageGetGreen(image, width, x, y)
            b = camera.imageGetBlue(image, width, x, y)

            color = classify_pixel_color(r, g, b)

            # Check within central ROI for close-range encounter confirmation
            if x_roi_start <= x < x_roi_end and y_roi_start <= y < y_roi_end:
                total_roi_pixels += 1
                if color in roi_color_counts:
                    roi_color_counts[color] += 1

            # Track horizontal position of unvisited target blocks for long-range homing
            if color is not None and color not in encountered_blocks:
                unvisited_x_sum += x
                unvisited_pixel_count += 1

    detected_color = None
    if total_roi_pixels > 0:
        # Prioritize checking unvisited colors in the ROI first
        for candidate_color in ["red", "blue", "green"]:
            if candidate_color not in encountered_blocks:
                match_ratio = roi_color_counts[candidate_color] / float(total_roi_pixels)
                if match_ratio >= ROI_MATCH_RATIO_THRESHOLD:
                    detected_color = candidate_color
                    break

    # Compute normalized horizontal steering bias [-1.0 (left) to +1.0 (right)]
    homing_bias = None
    if unvisited_pixel_count >= 4:
        avg_x = unvisited_x_sum / float(unvisited_pixel_count)
        homing_bias = (avg_x - (width / 2.0)) / (width / 2.0)

    return detected_color, homing_bias


def print_encounter_report(color, encountered_blocks, round_num, elapsed_time):
    """
    Displays the required first-encounter message, the elapsed search time,
    and a summary of which items have been encountered before and so far.
    """
    # 1. Required first-encounter response message
    print(f"I see {color}")

    # 2. Format previous encounters (strictly prior to this encounter)
    if len(encountered_blocks) == 0:
        prev_str = "None (this is the first block encountered)"
    else:
        prev_str = ", ".join([f"{c.capitalize()} block" for c in encountered_blocks])

    # Record current block into the encountered history list
    encountered_blocks.append(color)

    # 3. Format cumulative encounters (all items encountered up to now)
    cumulative_str = ", ".join([f"{c.capitalize()} block" for c in encountered_blocks])

    # Display structured summary with elapsed time
    print(f"--- Encounter Summary (Round {round_num}) ---")
    print(f"Elapsed search time: {elapsed_time:.2f} seconds")
    print(f"Previously encountered before this item: {prev_str}")
    print(f"Total items encountered so far ({len(encountered_blocks)}/3): {cumulative_str}")
    print(f"-----------------------------------------", flush=True)


def run_robot():
    """
    Main execution loop for the e-puck search and detection controller.
    """
    robot = Robot()

    # --------------------------------------------------------------------------
    # 1. Initialize Motors (Differential Drive)
    # --------------------------------------------------------------------------
    left_motor = robot.getDevice("left wheel motor")
    right_motor = robot.getDevice("right wheel motor")
    left_motor.setPosition(float("inf"))
    right_motor.setPosition(float("inf"))
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

    # --------------------------------------------------------------------------
    # 2. Initialize 8 Infrared Proximity Sensors (ps0 - ps7)
    # --------------------------------------------------------------------------
    ps_sensors = []
    for i in range(8):
        sensor = robot.getDevice(f"ps{i}")
        sensor.enable(TIME_STEP)
        ps_sensors.append(sensor)

    # --------------------------------------------------------------------------
    # 3. Initialize Onboard Camera
    # --------------------------------------------------------------------------
    camera = robot.getDevice("camera")
    camera.enable(TIME_STEP)
    cam_width = camera.getWidth()
    cam_height = camera.getHeight()

    # --------------------------------------------------------------------------
    # 4. State Tracking & Multi-Round Timer Variables
    # --------------------------------------------------------------------------
    encountered_blocks = []          # Stores colors in order of first encounter in current round
    round_number = 1                 # Tracks current test round (Round 1, Round 2, ...)
    round_start_time = robot.getTime() # Simulation timestamp when current round started
    in_standby_mode = False          # True when waiting between rounds for user to move blocks
    standby_start_time = 0.0         # Timestamp when standby started
    round_durations = []             # Stores total completion time for each finished round

    step_counter = 0
    wander_bias = 0.0
    turn_counter = 0
    turn_direction = 1
    stuck_counter = 0

    print("=========================================================")
    print(f"=== Starting Search Round {round_number} (Timer set to 0.00s) ===")
    print("E-puck autonomous search initialized. Searching for red, blue, and green blocks...")
    print("=========================================================", flush=True)

    # --------------------------------------------------------------------------
    # 5. Main Control Loop
    # --------------------------------------------------------------------------
    while robot.step(TIME_STEP) != -1:
        current_sim_time = robot.getTime()
        step_counter += 1

        # ======================================================================
        # Mode A: Inter-Round Standby Mode (Waiting to start next search round)
        # ======================================================================
        if in_standby_mode:
            wait_elapsed = current_sim_time - standby_start_time
            if wait_elapsed < ROUND_STANDBY_DURATION - 1.2:
                # Keep wheels stopped while user clicks PAUSE and moves blocks
                left_motor.setVelocity(0.0)
                right_motor.setVelocity(0.0)
            elif wait_elapsed < ROUND_STANDBY_DURATION:
                # Spin in place during the last 1.2s so the robot faces a new direction
                left_motor.setVelocity(-0.7 * MAX_SPEED)
                right_motor.setVelocity(0.7 * MAX_SPEED)
            else:
                # Standby finished: reset state and timer for the new round
                in_standby_mode = False
                round_number += 1
                encountered_blocks.clear()
                round_start_time = current_sim_time
                turn_counter = 0
                stuck_counter = 0
                print("\n=========================================================")
                print(f"=== Starting Search Round {round_number} (Timer Reset to 0.00s) ===")
                print("Searching for repositioned red, blue, and green blocks...")
                print("=========================================================", flush=True)
            continue

        # ======================================================================
        # Mode B: Active Search & Detection Mode
        # ======================================================================
        elapsed_round_time = current_sim_time - round_start_time

        # --- Step 1: Visual Block Detection & First-Encounter Reporting ---
        detected_color, homing_bias = analyze_camera_frame(
            camera, cam_width, cam_height, encountered_blocks
        )

        if detected_color is not None and detected_color not in encountered_blocks:
            print_encounter_report(
                detected_color, encountered_blocks, round_number, elapsed_round_time
            )

            # Check if all 3 blocks have been found in this round
            if len(encountered_blocks) == 3:
                round_durations.append((round_number, elapsed_round_time))
                left_motor.setVelocity(0.0)
                right_motor.setVelocity(0.0)

                print("=========================================================")
                print(f"[ROUND {round_number} COMPLETED] All three target blocks (Red, Blue, Green) found!")
                print(f"Total Time Taken in Round {round_number}: {elapsed_round_time:.2f} seconds")
                if len(round_durations) > 1:
                    history_str = ", ".join([f"Round {r}: {t:.2f}s" for r, t in round_durations])
                    print(f"All Rounds Time Summary: {history_str}")
                print("---------------------------------------------------------")
                print(f"Robot is now stopping for {ROUND_STANDBY_DURATION:.0f}s before Round {round_number + 1}.")
                print(">>> Click PAUSE now, drag the blocks to new locations, then click PLAY! <<<")
                print("=========================================================", flush=True)

                in_standby_mode = True
                standby_start_time = current_sim_time
                continue

        # --- Step 2: Read Proximity Sensors for Obstacle Avoidance ---
        ps_values = [sensor.getValue() for sensor in ps_sensors]

        # Front-right sensors: ps0, ps1, ps2 | Front-left sensors: ps5, ps6, ps7
        right_obstacle = (
            ps_values[0] > OBSTACLE_THRESHOLD or
            ps_values[1] > OBSTACLE_THRESHOLD or
            ps_values[2] > OBSTACLE_THRESHOLD
        )
        left_obstacle = (
            ps_values[5] > OBSTACLE_THRESHOLD or
            ps_values[6] > OBSTACLE_THRESHOLD or
            ps_values[7] > OBSTACLE_THRESHOLD
        )

        # --- Step 3: Navigation & Motor Speed Calculation ---
        if right_obstacle or left_obstacle:
            stuck_counter += 1
        else:
            stuck_counter = max(0, stuck_counter - 1)

        # Escape maneuver if trapped in a tight corner for too long
        if stuck_counter > 25:
            left_speed = -0.7 * MAX_SPEED
            right_speed = -0.3 * MAX_SPEED
            if stuck_counter > 35:
                stuck_counter = 0
                turn_counter = 12
                turn_direction = random.choice([-1, 1])
        elif turn_counter > 0:
            turn_counter -= 1
            left_speed = -turn_direction * 0.65 * MAX_SPEED
            right_speed = turn_direction * 0.65 * MAX_SPEED
        elif right_obstacle and left_obstacle:
            turn_direction = random.choice([-1, 1])
            turn_counter = random.randint(10, 18)
            left_speed = -turn_direction * 0.65 * MAX_SPEED
            right_speed = turn_direction * 0.65 * MAX_SPEED
        elif right_obstacle:
            turn_direction = 1
            turn_counter = random.randint(6, 12)
            left_speed = -0.55 * MAX_SPEED
            right_speed = 0.55 * MAX_SPEED
        elif left_obstacle:
            turn_direction = -1
            turn_counter = random.randint(6, 12)
            left_speed = 0.55 * MAX_SPEED
            right_speed = -0.55 * MAX_SPEED
        elif homing_bias is not None:
            # Visual homing: steer smoothly toward an unvisited block seen in camera
            steer = 0.45 * homing_bias * MAX_SPEED
            left_speed = max(-MAX_SPEED, min(MAX_SPEED, CRUISE_SPEED + steer))
            right_speed = max(-MAX_SPEED, min(MAX_SPEED, CRUISE_SPEED - steer))
        else:
            # Free space: cruise forward with periodic wandering bias
            if step_counter % 45 == 0:
                wander_bias = random.uniform(-0.22, 0.22) * MAX_SPEED

            left_speed = max(-MAX_SPEED, min(MAX_SPEED, CRUISE_SPEED + wander_bias))
            right_speed = max(-MAX_SPEED, min(MAX_SPEED, CRUISE_SPEED - wander_bias))

        # Apply calculated velocities to wheel actuators
        left_motor.setVelocity(left_speed)
        right_motor.setVelocity(right_speed)


if __name__ == "__main__":
    run_robot()
