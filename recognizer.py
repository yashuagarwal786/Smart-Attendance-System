import cv2
import face_recognition
import numpy as np
import os

base_path = 'images'

encodeListKnown = []
namesList = []
student_info = {}

# ✅ Load images (branch → section → students)
for branch in os.listdir(base_path):
    branch_path = os.path.join(base_path, branch)

    if not os.path.isdir(branch_path):
        continue

    for section in os.listdir(branch_path):
        section_path = os.path.join(branch_path, section)

        if not os.path.isdir(section_path):
            continue

        for file in os.listdir(section_path):
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):

                img_path = os.path.join(section_path, file)
                img = cv2.imread(img_path)

                if img is None:
                    continue

                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                encodes = face_recognition.face_encodings(rgb)

                if len(encodes) == 0:
                    continue

                encoding = encodes[0]
                name = os.path.splitext(file)[0].upper()

                encodeListKnown.append(encoding)
                namesList.append(name)

                student_info[name] = (branch, section)


def recognize_frame(frame):
    imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faces = face_recognition.face_locations(imgS)
    encodes = face_recognition.face_encodings(imgS, faces)

    results = []

    for encodeFace, faceLoc in zip(encodes, faces):
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if faceDis[matchIndex] < 0.6:
            name = namesList[matchIndex]
        else:
            name = "UNKNOWN"

        y1, x2, y2, x1 = faceLoc
        results.append((name, (x1*4, y1*4, x2*4, y2*4)))

    return results