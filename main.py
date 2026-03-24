import cv2
from video.capture import start_video_capture
from processing.frame_processor import process_frame
from detection.weapon_detection import WeaponDetector
import time
from config import HIGH_RISK_THRESHOLD, MEDIUM_RISK_THRESHOLD, FRAME_SKIP

def detect_behavior(frame):
    return False

def main():
    cap = start_video_capture()
    detector = WeaponDetector()
    frame_count = 0
    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # frame skipping (performance)
        if frame_count % FRAME_SKIP != 0:
            continue

        # Process frame
        processed_frame, frame_rgb = process_frame(frame)

        # Detection directly on RGB frame
        weapon_detected, detections, best_conf = detector.detect_weapon(frame_rgb)
       
        behavior_detected = detect_behavior(frame_rgb)

        # Defaults
        label = "SAFE"
        score = int(best_conf * 100) if weapon_detected else 0

        # Update Overall Risk Level
        if weapon_detected:
            if best_conf > HIGH_RISK_THRESHOLD:
                label = "HIGH RISK"
            elif best_conf > MEDIUM_RISK_THRESHOLD:
                label = "MEDIUM RISK"
            else:
                label = "LOW RISK"

        # Draw all detections
        if weapon_detected and detections:
            for box, conf in detections:
                x1, y1, x2, y2 = box

                # Bounding box
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0,0,255), 2)

                # Label near box
                y_text = max(y1 - 30, 20)
                cv2.putText(processed_frame, "WEAPON DETECTED",
                            (x1, y_text),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0,0,255), 2)

                # Confidence near box
                cv2.putText(processed_frame, f"{conf:.2f}",
                            (x1, y_text + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0,0,255), 2)

        # Status logic
        status = "SAFE"
        color = (0, 255, 0)  # Green

        if weapon_detected:
            status = "DANGER"
            color = (0, 0, 255)  # Red
        elif behavior_detected:
            status = "WARNING"
            color = (0, 255, 255)  # Yellow

        # FPS calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time

        # CLEAN UI 
        cv2.putText(processed_frame, f"Status: {status}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9, color, 2)

        cv2.putText(processed_frame, f"Threat: {score}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9, (0,255,255), 2)

        # Label near box (status)
        cv2.putText(processed_frame, f"Risk: {label}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9, (0,0,255), 2)

        cv2.putText(processed_frame, f"FPS: {int(fps)}",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (255,255,255), 2)
        # Display
        cv2.imshow("Live Feed", processed_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()