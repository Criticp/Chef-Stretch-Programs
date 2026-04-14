cat > /app/food_viewer_yolo.py <<'PY'
#!/usr/bin/env python3

# Import argparse so we can pass the robot IP and thresholds from the command line.
import argparse

# Import logging so the script prints clear status messages while it runs.
import logging

# Import time so we can sleep briefly when a frame is not ready yet.
import time

# Import datetime so snapshot filenames are unique and readable.
from datetime import datetime

# Import Path so we can safely create a snapshots folder.
from pathlib import Path

# Import OpenCV for the live camera window and drawing overlays.
import cv2

# Import the Stretch AI simple client.
from stretch.agent import RobotClient

# Import YOLO from ultralytics for first-pass food detection.
from ultralytics import YOLO


# Define the OpenCV window title once so it is easy to change later.
WINDOW_NAME = "Stretch Head Camera + Food Detection"

# Define the food labels we want to keep from the detector output.
# This is intentionally small for the first version.
FOOD_LABELS = {
    "apple",
    "banana",
    "orange",
    "sandwich",
    "pizza",
    "hot dog",
    "donut",
    "cake",
    "carrot",
    "broccoli",
}


def draw_overlay(frame, detections, detection_enabled):
    # Draw all saved detections on the current frame.
    for x1, y1, x2, y2, label, conf in detections:
        # Draw the bounding box around the detected food item.
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Build the text label with the confidence score.
        text = f"{label} {conf:.2f}"

        # Draw the text slightly above the top-left corner of the box.
        cv2.putText(
            frame,
            text,
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    # Build a status message for the top-left corner of the window.
    if detections:
        # Show the currently detected food labels when at least one food item is found.
        labels = ", ".join(sorted({label for _, _, _, _, label, _ in detections}))
        status_text = f"Detected: {labels}"
    else:
        # Show a helpful message when nothing food-related is found.
        status_text = "Detected: none"

    # Add whether detection is currently on or off.
    toggle_text = f"Detection: {'ON' if detection_enabled else 'OFF'}"

    # Draw the first status line.
    cv2.putText(
        frame,
        status_text,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Draw the second status line with the keyboard controls.
    cv2.putText(
        frame,
        "Keys: q=quit  s=save snapshot  d=toggle detection",
        (15, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Draw the third status line.
    cv2.putText(
        frame,
        toggle_text,
        (15, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def main():
    # Create the command-line parser.
    parser = argparse.ArgumentParser(description="Show Stretch head camera and detect food items.")

    # Add the robot IP argument so you can override it if needed.
    parser.add_argument("--robot_ip", type=str, default="192.168.20.203", help="Stretch robot IP")

    # Add the YOLO model argument so you can swap models later if desired.
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Ultralytics YOLO model path")

    # Add the confidence threshold argument.
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")

    # Add the frame-skip argument so inference does not run on every frame.
    parser.add_argument("--infer_every", type=int, default=5, help="Run detection every N frames")

    # Parse the command-line arguments.
    args = parser.parse_args()

    # Configure logging once at startup.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    # Create a folder for snapshots.
    snapshot_dir = Path("/app/snapshots")
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # Initialize placeholders so cleanup in finally always works.
    robot = None
    detector = None

    # Keep the most recent detections so we can draw them between inference frames.
    last_food_detections = []

    # Track the frame count so we only run the model every few frames.
    frame_count = 0

    # Start with detection enabled.
    detection_enabled = True

    try:
        # Announce that we are connecting to Stretch.
        logging.info("Connecting to Stretch at %s", args.robot_ip)

        # Create the Stretch AI client and disable rerun since we only want the OpenCV window here.
        robot = RobotClient(robot_ip=args.robot_ip, enable_rerun_server=False)

        # Announce that we are loading the detector.
        logging.info("Loading YOLO model: %s", args.model)

        # Load the detector weights.
        detector = YOLO(args.model)

        # Create a resizable OpenCV window.
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        # Announce that the live loop is starting.
        logging.info("Starting live camera + food detection loop")

        # Loop as long as the Stretch AI client is running.
        while robot.is_running():
            # Ask Stretch AI for the head RGB image and the aligned depth image.
            rgb, depth = robot.get_head_rgbd()

            # If no image has arrived yet, wait briefly and try again.
            if rgb is None:
                logging.warning("No head RGB frame yet; waiting...")
                time.sleep(0.05)
                continue

            # Convert the frame from RGB to BGR because OpenCV expects BGR for display.
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # Make a copy for drawing boxes and labels.
            display = bgr.copy()

            # Only run the detector every N frames to keep the UI responsive.
            if detection_enabled and (frame_count % max(1, args.infer_every) == 0):
                # Run the model on the current frame.
                results = detector.predict(source=display, conf=args.conf, verbose=False)

                # Clear the previous detections before storing fresh ones.
                last_food_detections = []

                # Extract the first result because we only passed one frame.
                result = results[0]

                # Iterate through every box the detector found.
                for box in result.boxes:
                    # Read the class index from the result.
                    class_id = int(box.cls[0].item())

                    # Convert the detector class name to lowercase for consistent matching.
                    class_name = str(detector.names[class_id]).lower()

                    # Read the confidence score for this detection.
                    confidence = float(box.conf[0].item())

                    # Keep only food-related detections for this first version.
                    if class_name in FOOD_LABELS:
                        # Convert the box coordinates to integers for OpenCV drawing.
                        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

                        # Save the detection so it can be drawn on this and subsequent frames.
                        last_food_detections.append((x1, y1, x2, y2, class_name, confidence))

            # Draw the overlays on the display frame.
            draw_overlay(display, last_food_detections, detection_enabled)

            # Show the frame in the OpenCV window.
            cv2.imshow(WINDOW_NAME, display)

            # Read one keyboard event.
            key = cv2.waitKey(1) & 0xFF

            # Quit when q is pressed.
            if key == ord("q"):
                logging.info("Quit requested by user")
                break

            # Save a snapshot when s is pressed.
            elif key == ord("s"):
                # Build a timestamped filename.
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # Build the full output path.
                out_path = snapshot_dir / f"stretch_food_snapshot_{stamp}.png"

                # Save the currently displayed frame.
                cv2.imwrite(str(out_path), display)

                # Log where the snapshot was saved.
                logging.info("Saved snapshot to %s", out_path)

            # Toggle detection on and off when d is pressed.
            elif key == ord("d"):
                # Flip the detection state.
                detection_enabled = not detection_enabled

                # Log the new detection state.
                logging.info("Detection enabled = %s", detection_enabled)

            # Increment the frame counter at the end of the loop.
            frame_count += 1

    except KeyboardInterrupt:
        # Handle Ctrl+C cleanly.
        logging.info("Interrupted by user")

    finally:
        # Always close the OpenCV windows on exit.
        cv2.destroyAllWindows()

        # Stop the Stretch AI client if it was created.
        if robot is not None:
            robot.stop()

        # Log final shutdown.
        logging.info("Shutdown complete")


if __name__ == "__main__":
    # Run the program.
    main()
PY