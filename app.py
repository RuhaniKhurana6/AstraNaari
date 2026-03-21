import streamlit as st
import cv2
from ultralytics import YOLO
import time
from streamlit_option_menu import option_menu

# Load model
model = YOLO("models/best.pt")

# Page config
st.set_page_config(
    page_title="AI Security System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CUSTOM CSS (UI MAGIC)
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
.stMetric {
    background-color: #1c1f26;
    padding: 10px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# SIDEBAR MENU
with st.sidebar:
    selected = option_menu(
        "AI Security",
        ["Dashboard", "Location", "Alerts", "AI Assistant"],
        icons=["camera", "geo-alt", "bell", "robot"],
        menu_icon="shield-lock",
        default_index=0,
    )

# ---------------- DASHBOARD ----------------
if selected == "Dashboard":
    st.title("🔒 AI Weapon Detection Dashboard")

    col1, col2, col3 = st.columns(3)

    status_box = col1.empty()
    threat_box = col2.empty()
    fps_box = col3.empty()

    run = st.toggle("Start Camera")

    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)

    prev_time = 0

    while run:
        ret, frame = cap.read()
        if not ret:
            break

        small_frame = cv2.resize(frame, (320, 320))

        results = model(small_frame)

        weapon_detected = False
        best_conf = 0
        box_coords = None

        for box in results[0].boxes:
            conf = float(box.conf)

            if conf > best_conf:
                best_conf = conf
                box_coords = box

        if best_conf > 0.6 and box_coords is not None:
            weapon_detected = True

            x1, y1, x2, y2 = map(int, box_coords.xyxy[0])

            h, w, _ = frame.shape
            x1 = int(x1 * w / 320)
            y1 = int(y1 * h / 320)
            x2 = int(x2 * w / 320)
            y2 = int(y2 * h / 320)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
            cv2.putText(frame, f"WEAPON {best_conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0,0,255), 2)

        # Status
        status = "DANGER" if weapon_detected else "SAFE"
        color = "red" if weapon_detected else "green"

        # FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time

        # METRIC CARDS
        status_box.metric("Status", status)
        threat_box.metric("Confidence", f"{best_conf:.2f}")
        fps_box.metric("FPS", int(fps))

        FRAME_WINDOW.image(frame, channels="BGR")

    cap.release()

# ---------------- LOCATION ----------------
elif selected == "Location":
    st.title("📍 Location Tracking")

    st.info("Camera Location: Room 1")

# ---------------- ALERTS ----------------
elif selected == "Alerts":
    st.title("🚨 Alert History")

    st.warning("Recent Alerts")

    st.markdown("""
    - 🔴 [18:45] Room - HIGH RISK  
    - 🟠 [18:47] Kitchen - MEDIUM RISK  
    """)

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