import cv2
from video.capture import start_video_capture
from processing.frame_processor import process_frame
from detection.weapon_detection import detect_weapon
import time

def detect_behavior(frame):
    return False

def main():
    cap = start_video_capture()
    frame_count = 0
    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # # frame skipping (performance)
        # if frame_count % 2 != 0:
        #     continue

        # Process frame
        processed_frame, frame_rgb = process_frame(frame)
        small_frame = cv2.resize(frame_rgb, (320, 320))
        processed_frame = frame.copy()

        # Detection
        weapon_detected, box, conf = detect_weapon(small_frame)
       
        behavior_detected = detect_behavior(frame_rgb)

        # Defaults
        label = "SAFE"
        score = int(conf * 100) if weapon_detected else 0

        # Draw detection
        if weapon_detected and box is not None:
            x1, y1, x2, y2 = box

            # 🔥 SCALE BACK
            h, w, _ = frame_rgb.shape
            scale_x = w / 320
            scale_y = h / 320

            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)

            # 🔥 Risk level
            if conf > 0.8:
                label = "HIGH RISK"
            elif conf > 0.65:
                label = "MEDIUM RISK"
            else:
                label = "LOW RISK"

            # Bounding box
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0,0,255), 2)

            # Label near box
            cv2.putText(processed_frame, "WEAPON DETECTED",
                        (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0,0,255), 2)

            # Confidence near box
            cv2.putText(processed_frame, f"{conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0,0,255), 2)

        # Status logic
        status = "SAFE"
        if weapon_detected:
            status = "DANGER"
        elif behavior_detected:
            status = "WARNING"

        color = (0,255,0) if status == "SAFE" else (0,0,255)

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