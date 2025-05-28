import cv2
import numpy as np
from collections import Counter

def get_dominant_color(image):
    pixels = image.reshape(-1, 3)
    pixels = [tuple(pixel) for pixel in pixels if np.any(pixel > 30)]
    if not pixels:
        return "unknown"
    most_common = Counter(pixels).most_common(1)[0][0]
    b, g, r = most_common
    return classify_color(r, g, b)

def classify_color(r, g, b):
    if r > 200 and g < 100 and b < 100:
        return "red"
    elif g > 200 and r < 100 and b < 100:
        return "green"
    elif b > 200 and r < 100 and g < 100:
        return "blue"
    elif r > 200 and g > 200 and b < 100:
        return "yellow"
    elif r > 200 and g > 200 and b > 200:
        return "white"
    elif r < 50 and g < 50 and b < 50:
        return "black"
    elif r > 150 and g > 100 and b < 100:
        return "orange"
    elif r > 150 and b > 150 and g < 100:
        return "pink"
    elif r > 100 and g > 100 and b > 100:
        return "gray"
    else:
        return "unknown"

def generate_suggestion(top_color, bottom_color):
    if top_color == "black" and bottom_color == "blue":
        return "This outfit looks casual. You can add white shoes to complete the look."
    elif top_color == "white" and bottom_color == "black":
        return "Classic and clean. A red accessory could make it stand out."
    elif top_color == "blue" and bottom_color == "black":
        return "You look formal. Consider adding white sneakers to match."
    elif top_color == "red" and bottom_color == "black":
        return "Bold choice! You can soften the look with neutral shoes."
    elif "unknown" in (top_color, bottom_color):
        return "I couldn't clearly detect the outfit colors."
    else:
        return "You look great! Keep up the stylish vibe."

def check_outfit():
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return "Failed to capture image from the camera."

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    boxes, _ = hog.detectMultiScale(gray, winStride=(8, 8))

    if len(boxes) == 0:
        return "No person detected in the image."

    for (x, y, w, h) in boxes:
        top_roi = frame[y:y + h // 3, x:x + w]
        bottom_roi = frame[y + h // 2:y + h, x:x + w]

        top_color = get_dominant_color(top_roi)
        bottom_color = get_dominant_color(bottom_roi)
        suggestion = generate_suggestion(top_color, bottom_color)

        result = f"You are wearing a {top_color} top and a {bottom_color} bottom. {suggestion}"
        print(result)
        return result

    return "Could not analyze outfit properly."
