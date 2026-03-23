import streamlit as st
import cv2
import time
from datetime import datetime
from streamlit_option_menu import option_menu
from detection.weapon_detection import WeaponDetector
from video.capture import ThreadedCamera

# Page config
st.set_page_config(
    page_title="AstraNaari",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CUSTOM CSS (UI MAGIC)
st.markdown("""
<style>
.main {
    background-color: #0E1117;
    font-family: 'Inter', sans-serif;
}
.stMetric {
    background-color: #1c1f26;
    padding: 15px;
    border-radius: 8px;
    border: 1px solid #2d3139;
}
</style>
""", unsafe_allow_html=True)

# Initialize Session State for Alerts
if 'alert_history' not in st.session_state:
    st.session_state['alert_history'] = []

# SIDEBAR MENU
with st.sidebar:
    selected = option_menu(
        "AstraNaari",
        ["Dashboard", "Location", "Alerts", "AI Assistant"],
        icons=["camera", "geo-alt", "bell", "robot"],
        menu_icon="shield-lock",
        default_index=0,
    )

# ---------------- DASHBOARD ----------------
if selected == "Dashboard":
    st.title("🔒 AstraNaari Dashboard")

    col1, col2, col3 = st.columns(3)

    status_box = col1.empty()
    threat_box = col2.empty()
    fps_box = col3.empty()

    run = st.toggle("Start Camera")

    FRAME_WINDOW = st.image([])

    if run:
        # ASYNC THREADED CAMERA INIT
        cap = ThreadedCamera().start()
        detector = WeaponDetector()
        prev_time = time.time()
        
        # CAMERA LOOP
        while run:
            ret, frame = cap.read()
            
            # Error Handling Example
            if not ret or frame is None:
                st.error("⚠️ Camera disconnected or blocked. Attempting to restart...")
                time.sleep(1)
                continue

            weapon_detected, valid_boxes, best_conf = detector.detect_weapon(frame)

            if weapon_detected and valid_boxes:
                for box, conf in valid_boxes:
                    x1, y1, x2, y2 = box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
                    cv2.putText(frame, f"WEAPON {conf:.2f}",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0,0,255), 2)

            # Status & Logging
            if weapon_detected:
                status = "DANGER"
                now = datetime.now().strftime("%H:%M:%S")
                if not st.session_state['alert_history'] or st.session_state['alert_history'][0]['time'] != now:
                    st.session_state['alert_history'].insert(0, {
                        "time": now,
                        "location": "Room 1",
                        "threat_level": f"WEAPON DETECTED ({best_conf:.2f})"
                    })
            else:
                status = "SAFE"

            # FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
            prev_time = curr_time

            # METRIC CARDS
            status_box.metric("Status", status)
            threat_box.metric("Confidence", f"{best_conf:.2f}" if weapon_detected else "0.00")
            fps_box.metric("FPS", int(fps))

            FRAME_WINDOW.image(frame, channels="BGR")

        cap.stop()

# ---------------- LOCATION ----------------
elif selected == "Location":
    st.title("📍 Location Tracking")
    st.info("Camera Location: Room 1")

# ---------------- ALERTS ----------------
elif selected == "Alerts":
    st.title("🚨 Alert History")
    if st.session_state['alert_history']:
        st.warning(f"Recent Alerts ({len(st.session_state['alert_history'])})")
        st.dataframe(
            st.session_state['alert_history'], 
            use_container_width=True,
            column_config={
                "time": "Timestamp",
                "location": "Location",
                "threat_level": "Threat Level"
            }
        )
    else:
        st.success("✅ No alerts recorded yet. System is safe.")

# ---------------- AI ASSISTANT ----------------
elif selected == "AI Assistant":
    st.title("🤖 AI Assistant")
    query = st.text_input("Ask about system status")
    if query:
        if "danger" in query.lower():
            st.error("⚠️ Threat detected. Stay alert.")
        elif "safe" in query.lower():
            st.success("✅ System is safe.")
        else:
            st.info("Monitoring system is active.")