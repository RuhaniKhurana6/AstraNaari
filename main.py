import cv2
import time
import numpy as np
from detection.weapon_detection import detect_weapon
from processing.audio_processor import AudioDetector
from alert.buzzer import play_buzzer

# ─── Video Setup ──────────────────────────────────────────────────────────────
def start_video_capture(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        return cap
    print(f"Successfully opened camera {camera_index}")
    return cap

# ─── Unusual Movement Detection ───────────────────────────────────────────────
def detect_behavior(frame, prev_gray):
    """Returns True if SUDDEN / UNUSUAL movement is detected."""
    if frame is None or frame.size == 0:
        return False, prev_gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (21, 21), 0)
    if prev_gray is None:
        return False, gray_blur
    frame_delta = cv2.absdiff(prev_gray, gray_blur)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    motion_score = (np.sum(thresh) / (thresh.shape[0] * thresh.shape[1] * 255)) * 100
    # 45% pixel change = very sudden movement (e.g. running, attacking)
    return motion_score > 45, gray_blur

# ─── Main Loop ────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  AstraNaari Security System — FINAL BUILD v3.0")
    print("  Detects: Sharp Objects | High Volume | Movement")
    print("=" * 55)

    cap = start_video_capture()
    if not cap.isOpened():
        return

    # Audio: only alert at 110+ dB (concert / siren level and above)
    audio_monitor = AudioDetector(threshold_db=110)

    prev_gray = None
    status = "SAFE"
    status_reason = ""
    status_timer = 0
    frame_count = 0
    buzzer_cooldown = 0  # prevent repeating buzzers every frame

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        processed_frame = frame.copy()
        h, w = processed_frame.shape[:2]

        # ── 1. Sharp Object / Weapon Detection (every 3 frames) ──────────────
        weapon_detected = False
        box, conf = None, 0
        if frame_count % 3 == 0:
            small = cv2.resize(frame, (320, 320))
            weapon_detected, box, conf = detect_weapon(small)

        if weapon_detected and box:
            x1, y1, x2, y2 = box
            x1 = int(x1 * (w / 320))
            y1 = int(y1 * (h / 320))
            x2 = int(x2 * (w / 320))
            y2 = int(y2 * (h / 320))
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            label = f"SHARP OBJECT {conf:.0%}"
            cv2.putText(processed_frame, label, (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # ── 2. Unusual Movement Detection ────────────────────────────────────
        behavior_detected, prev_gray = detect_behavior(frame, prev_gray)

        # ── 3. High Volume Detection ──────────────────────────────────────────
        _, current_db, _ = audio_monitor.get_levels()

        # Map dB → label
        if current_db >= 140:
            db_label = "SEVERE DANGER (Explosion/Gunshot)"
            db_color = (0, 0, 180)
        elif current_db >= 120:
            db_label = "PAIN THRESHOLD (120dB+)"
            db_color = (0, 0, 255)
        elif current_db >= 110:
            db_label = "DANGEROUS (Siren/Concert)"
            db_color = (30, 80, 255)
        elif current_db >= 85:
            db_label = "LOUD"
            db_color = (0, 165, 255)
        elif current_db >= 70:
            db_label = "MODERATE"
            db_color = (0, 220, 220)
        else:
            db_label = "SAFE"
            db_color = (0, 220, 60)

        # ── 4. Decision Engine ────────────────────────────────────────────────
        current_event = "SAFE"

        if weapon_detected or current_db >= 120:
            current_event = "DANGER"
            if weapon_detected:
                status_reason = "Sharp Object Detected"
            else:
                status_reason = f"Extreme Volume ({current_db:.0f} dB)"
        elif behavior_detected or current_db >= 110:
            current_event = "WARNING"
            if behavior_detected:
                status_reason = "Unusual Movement"
            else:
                status_reason = f"Dangerous Volume ({current_db:.0f} dB)"

        if current_event != "SAFE":
            status = current_event
            status_timer = time.time()
            if time.time() > buzzer_cooldown:
                if status == "DANGER":
                    play_buzzer(frequency=2500, duration=300)
                else:
                    play_buzzer(frequency=1200, duration=200)
                buzzer_cooldown = time.time() + 1.5  # 1.5s between buzzers
        elif time.time() - status_timer > 3.0:
            status = "SAFE"
            status_reason = ""

        # ── 5. Premium UI ─────────────────────────────────────────────────────

        # Top status bar
        bar_col = (0, 200, 50) if status == "SAFE" else (0, 0, 230) if status == "DANGER" else (0, 140, 230)
        cv2.rectangle(processed_frame, (0, 0), (w, 70), (25, 25, 25), -1)
        cv2.rectangle(processed_frame, (0, 0), (8, 70), bar_col, -1)  # color stripe
        cv2.putText(processed_frame, f"STATUS: {status}", (22, 35),
                    cv2.FONT_HERSHEY_DUPLEX, 1.1, bar_col, 2)
        if status_reason:
            cv2.putText(processed_frame, status_reason, (22, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # dB Info box — top right
        bx = w - 310
        cv2.rectangle(processed_frame, (bx, 75), (w - 5, 155), (35, 35, 35), -1)
        cv2.putText(processed_frame, f"VOLUME: {current_db:.1f} dB", (bx + 10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
        cv2.putText(processed_frame, db_label, (bx + 10, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, db_color, 2)

        # Indicator dots (bottom-left)
        dot_y = h - 50
        def dot(label, detected, x):
            col = (0, 60, 255) if detected else (0, 180, 60)
            cv2.circle(processed_frame, (x, dot_y), 10, col, -1)
            cv2.putText(processed_frame, label, (x - 20, dot_y + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

        dot("SHARP", weapon_detected, 50)
        dot("MOVE", behavior_detected, 130)
        dot("LOUD", current_db >= 110, 210)

        # Calibrating notice
        if audio_monitor.is_calibrating:
            cv2.rectangle(processed_frame, (0, h - 30), (w, h), (30, 30, 0), -1)
            cv2.putText(processed_frame, "CALIBRATING ROOM VOLUME — PLEASE STAY SILENT",
                        (20, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)

        cv2.imshow("AstraNaari Security v3.0", processed_frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    audio_monitor.stop()


if __name__ == "__main__":
    main()