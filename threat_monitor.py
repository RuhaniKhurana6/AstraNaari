import cv2
from video.capture import start_video_capture
from processing.frame_processor import process_frame
from detection.weapon_detection import detect_weapon
from alert.buzzer import play_buzzer
from processing.audio_processor import AudioDetector
import time

def detect_behavior(frame, prev_gray):
    """Detects unusual/rapid motion by comparing current frame with the previous one."""
    if frame is None or frame.size == 0:
        return False, prev_gray
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Blur to reduce noise
    gray_blur = cv2.GaussianBlur(gray, (21, 21), 0)

    if prev_gray is None:
        return False, gray_blur
    # Blur to reduce noise
    gray_blur = cv2.GaussianBlur(gray, (21, 21), 0)
    
    frame_delta = cv2.absdiff(prev_gray, gray_blur)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    # Calculate motion percentage
    motion_score = (cv2.countNonZero(thresh) / (frame.shape[0] * frame.shape[1])) * 100
    
    # Return detection if motion > 40% of frame (unusual behavior like punches)
    return motion_score > 40, gray_blur

def main():
    import processing.audio_processor as ap
    print(f"DEBUG: Using Audio Processor from: {ap.__file__}")
    print("--- AstraNaari System VERSION: 2.0 (Raw Audio Refined) ---")
    cap = start_video_capture()
    if not cap.isOpened():
        print("Error: Could not open video source. Please check if your camera is connected and not in use by another app.")
        return

    # Initialize Audio Detector
    audio_threshold = 0.1 # Raw baseline
    audio_monitor = AudioDetector(threshold=audio_threshold) 
    
    frame_count = 0
    prev_time = 0
    prev_gray = None
    status_timer = 0
    status = "SAFE"

    # State variables for detection persistence
    weapon_detected = False
    box = None
    conf = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Process frame
        processed_frame, frame_rgb = process_frame(frame)
        if frame_rgb is None or frame_rgb.size == 0:
            continue
            
        small_frame = cv2.resize(frame_rgb, (320, 320))
        processed_frame = frame.copy()

        # AI Detection (every 5th frame for speed)
        if frame_count % 5 == 0:
            weapon_detected, box, conf = detect_weapon(small_frame)
       
        # Behavior Detection (every frame for responsiveness)
        behavior_detected, prev_gray = detect_behavior(frame, prev_gray)
        audio_detected, audio_level = audio_monitor.check_for_scream()

        # Defaults
        label = "SAFE"
        score = int(conf * 100) if weapon_detected else 0
        
        # INCREASED THRESHOLD: 45% for "Unusual" behavior like punches
        if behavior_detected and not weapon_detected:
            label = "UNUSUAL MOTION"
            score = 50
        elif audio_detected:
            label = "LOUD NOISE / SCREAM"
            score = 70
            behavior_detected = True # Trigger behavior logic
        
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

        # Status logic with persistence
        current_event = "SAFE"
        if weapon_detected:
            current_event = "DANGER"
        elif behavior_detected:
            current_event = "WARNING"

        # Update persistent status
        if current_event == "DANGER":
            status = "DANGER"
            status_timer = time.time()
        elif current_event == "WARNING":
            if status != "DANGER": # Don't downgrade Danger to Warning
                status = "WARNING"
                status_timer = time.time()
        else:
            # Check if we should revert to SAFE
            if time.time() - status_timer > 2.0: # 2 second persistence
                status = "SAFE"

        if status == "DANGER":
            play_buzzer(frequency=2500, duration=500) # Fast repeating high beep
        elif status == "WARNING":
            play_buzzer(frequency=1000, duration=300) # Repeating low beep

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

        # Display Calibration status
        if audio_monitor.is_calibrating:
            cv2.putText(processed_frame, "CALIBRATING AUDIO... (BE QUIET)",
                (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        cv2.putText(processed_frame, f"FPS: {int(fps)}",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (255,255,255), 2)

        # Visual Sound Meter (Horizontal Bar)
        bar_x1, bar_y1 = 20, 200
        bar_length = 200
        bar_height = 20
        # Scaling for RAW Signal (Gain removed)
        current_threshold = audio_monitor.threshold
        fill_width = int(min(audio_level * 5, 1) * bar_length)
        threshold_line = int(min(current_threshold * 5, 1) * bar_length)
        
        cv2.rectangle(processed_frame, (bar_x1, bar_y1), (bar_x1 + bar_length, bar_y1 + bar_height), (100,100,100), 1)
        cv2.rectangle(processed_frame, (bar_x1, bar_y1), (bar_x1 + fill_width, bar_y1 + bar_height), (0,255,255), -1)
        # Draw dynamic threshold line (Red)
        cv2.line(processed_frame, (bar_x1 + threshold_line, bar_y1), (bar_x1 + threshold_line, bar_y1 + bar_height), (0,0,255), 2)
        
        # Display numeric level for debugging
        cv2.putText(processed_frame, f"Sound: {audio_level:.3f} (Aim for > {current_threshold:.2f})", (bar_x1, bar_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # Display
        cv2.imshow("Live Feed", processed_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()