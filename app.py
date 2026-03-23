from flask import Flask, render_template, Response
import cv2
import time
import csv
import os
from datetime import datetime
import numpy as np

from recognizer import recognize_frame, student_info

app = Flask(__name__)

camera = cv2.VideoCapture(0)

# Stability control
name_buffer = {}

# Liveness
prev_gray = None
movement_counter = 0

# Unknown save
last_unknown_save = 0
UNKNOWN_COOLDOWN = 5

# Display card
display_card_until = 0
display_student = None


# FINAL FIXED ATTENDANCE FUNCTION
def mark_attendance(name):
    print("Trying:", name)

    if name == "UNKNOWN" or name not in student_info:
        print("Skipped:", name)
        return

    branch, section = student_info[name]

    date = datetime.now().strftime("%Y-%m-%d")
    os.makedirs("attendance", exist_ok=True)

    file_name = f"attendance/{branch}_{section}_{date}.csv"

    file_exists = os.path.exists(file_name)

    existing_names = []

    if file_exists:
        with open(file_name, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            existing_names = [row[0] for row in reader]

    if name in existing_names:
        print("Already marked:", name)
        return

    with open(file_name, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["Name", "Time"])

        writer.writerow([name, datetime.now().strftime("%H:%M:%S")])

    print("SUCCESS:", name)


# Get student image
def get_student_image(name):
    branch, section = student_info.get(name, ("", ""))
    folder = f"images/{branch}/{section}"

    if not os.path.exists(folder):
        return None

    for file in os.listdir(folder):
        if file.upper().startswith(name):
            return os.path.join(folder, file)

    return None


# Display Card
def create_display_card(frame, name):
    img_path = get_student_image(name)

    if img_path is None:
        return frame

    h, w, _ = frame.shape

    blurred = cv2.GaussianBlur(frame, (35, 35), 0)
    overlay = blurred.copy()
    dark_layer = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.addWeighted(overlay, 0.4, dark_layer, 0.6, 0, overlay)

    student_img = cv2.imread(img_path)
    student_img = cv2.resize(student_img, (250, 250))

    cx, cy = w // 2, h // 2
    x1 = cx - 125
    y1 = cy - 180

    cv2.rectangle(overlay, (x1-5, y1-5), (x1+255, y1+255), (0,255,200), 3)
    overlay[y1:y1+250, x1:x1+250] = student_img

    branch, section = student_info[name]

    cv2.putText(overlay, name,
                (cx - 120, cy + 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,200), 2)

    cv2.putText(overlay, f"{branch} - {section}",
                (cx - 100, cy + 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

    cv2.putText(overlay, "Attendance Marked ✓",
                (cx - 140, cy + 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    return overlay


# Save unknown face
def save_unknown_face(frame, x1, y1, x2, y2):
    global last_unknown_save

    current_time = time.time()
    if current_time - last_unknown_save < UNKNOWN_COOLDOWN:
        return

    folder = "unknown_faces"
    os.makedirs(folder, exist_ok=True)

    h, w, _ = frame.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    face = frame[y1:y2, x1:x2]

    if face.size == 0 or (x2 - x1) < 80:
        return

    filename = f"{folder}/unknown_{datetime.now().strftime('%H%M%S')}.jpg"
    cv2.imwrite(filename, face)

    last_unknown_save = current_time


# Main generator
def generate_frames():
    global prev_gray, movement_counter, display_card_until, display_student

    while True:
        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            movement = cv2.countNonZero(diff)

            if movement > 2000:
                movement_counter += 1
            else:
                movement_counter = max(0, movement_counter - 1)

        prev_gray = gray

        results = recognize_frame(frame)

        for name, (x1, y1, x2, y2) in results:

            if name not in name_buffer:
                name_buffer[name] = 0

            name_buffer[name] += 1

            display_name = ""

            if name != "UNKNOWN" and name in student_info:
                branch, section = student_info[name]
                display_name = f"{name} ({branch}-{section})"

                mark_attendance(name)

                # Delay popup
                if name_buffer[name] == 10:
                    display_student = name
                    display_card_until = time.time() + 3

            else:
                display_name = ""
                save_unknown_face(frame, x1, y1, x2, y2)

            color = (0,255,0) if display_name != "" else (0,0,255)

            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)

            if display_name != "":
                cv2.putText(frame, display_name, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Show overlay
        if time.time() < display_card_until and display_student is not None:
            frame = create_display_card(frame, display_student)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)