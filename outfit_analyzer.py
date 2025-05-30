import cv2
import numpy as np
from sklearn.cluster import KMeans

# Load MobileNet SSD model
prototxt = "deploy.prototxt"
model = "mobilenet_iter_73000.caffemodel"

print("🔄 Loading model...")
try:
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Failed to load model:", e)
    exit()

# Class labels
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

def extract_top_bottom_regions(frame):
    h, w = frame.shape[:2]
    top = frame[int(h * 0.15):int(h * 0.40), int(w * 0.30):int(w * 0.70)]
    bottom = frame[int(h * 0.60):int(h * 0.85), int(w * 0.30):int(w * 0.70)]
    return top, bottom

def preprocess_image(region):
    region = cv2.resize(region, (100, 100))
    blurred = cv2.GaussianBlur(region, (7, 7), 0)
    return blurred

def detect_dominant_color(image, k=4):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(pixels)
    counts = np.bincount(labels)
    dominant_index = np.argmax(counts)
    dominant_color = kmeans.cluster_centers_[dominant_index]
    return dominant_color.astype(int)

def rgb_to_hsv(rgb):
    r, g, b = [x / 255.0 for x in rgb]
    mx = max(r, g, b)
    mn = min(r, g, b)
    diff = mx - mn

    if diff == 0:
        h = 0
    elif mx == r:
        h = (60 * ((g - b) / diff) + 360) % 360
    elif mx == g:
        h = (60 * ((b - r) / diff) + 120) % 360
    else:
        h = (60 * ((r - g) / diff) + 240) % 360

    s = 0 if mx == 0 else diff / mx
    v = mx
    return h, s, v

def get_closest_color(h, s, v):
    color_ranges = [
        ("red",     [(0, 10), (350, 360)], 0.5, 0.2, 1.0, 1.0),
        ("orange",  [(11, 40)], 0.5, 0.2, 1.0, 1.0),
        ("yellow",  [(41, 65)], 0.4, 0.4, 1.0, 1.0),
        ("green",   [(66, 170)], 0.4, 0.2, 1.0, 1.0),
        ("blue",    [(171, 260)], 0.4, 0.2, 1.0, 1.0),
        ("purple",  [(261, 320)], 0.3, 0.2, 1.0, 1.0),
        ("brown",   [(10, 40)], 0.3, 0.1, 0.7, 0.6),
    ]

    if v <= 0.15:
        return "black"
    if s <= 0.15 and v >= 0.85:
        return "white"
    if s <= 0.15 and 0.15 < v < 0.85:
        return "gray"

    for name, hue_ranges, sat_min, val_min, sat_max, val_max in color_ranges:
        for (h_min, h_max) in hue_ranges:
            if h_min <= h <= h_max and sat_min <= s <= sat_max and val_min <= v <= val_max:
                return name

    main_hues = {
        "red": 0,
        "orange": 25,
        "yellow": 55,
        "green": 120,
        "blue": 215,
        "purple": 290,
        "brown": 25,
    }

    def hue_distance(a, b):
        d = abs(a - b)
        return min(d, 360 - d)

    closest_color = None
    min_dist = 360
    for color, color_hue in main_hues.items():
        dist = hue_distance(h, color_hue)
        if dist < min_dist:
            min_dist = dist
            closest_color = color

    return closest_color

def suggest_outfit(top_color, bottom_color):
    matching = {
        "red": ["black", "white", "blue", "gray"],
        "orange": ["blue", "white", "black"],
        "yellow": ["blue", "gray", "black"],
        "green": ["black", "white", "brown"],
        "blue": ["white", "gray", "khaki"],
        "purple": ["black", "white", "gray"],
        "brown": ["white", "blue", "green"],
        "black": ["red", "yellow", "white"],
        "white": ["black", "blue", "red"],
        "gray": ["blue", "white", "black"],
    }
    if top_color in matching and bottom_color in matching[top_color]:
        return "✅ Great combo!"
    elif bottom_color in matching and top_color in matching[bottom_color]:
        return "✅ Looks good together!"
    else:
        return f"👕 Try pairing {top_color} with {', '.join(matching.get(top_color, []))}"

def analyze_outfit_colors(frame):
    top, bottom = extract_top_bottom_regions(frame)
    if top is None or bottom is None or top.size == 0 or bottom.size == 0:
        return None, None, None, None
    top = preprocess_image(top)
    bottom = preprocess_image(bottom)

    top_rgb = detect_dominant_color(top)
    bottom_rgb = detect_dominant_color(bottom)

    top_hsv = rgb_to_hsv(top_rgb)
    bottom_hsv = rgb_to_hsv(bottom_rgb)

    top_color = get_closest_color(*top_hsv)
    bottom_color = get_closest_color(*bottom_hsv)

    return top_color, bottom_color, top_rgb, bottom_rgb

def show_color_feedback(frame, top_rgb, bottom_rgb, top_color, bottom_color):
    suggestion = suggest_outfit(top_color, bottom_color)

    cv2.rectangle(frame, (10, 10), (100, 60), top_rgb[::-1].tolist(), -1)
    cv2.putText(frame, f"Top: {top_color}", (110, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.rectangle(frame, (10, 70), (100, 120), bottom_rgb[::-1].tolist(), -1)
    cv2.putText(frame, f"Bottom: {bottom_color}", (110, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.putText(frame, suggestion, (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

# Start camera
print("🎥 Opening camera...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()
print("✅ Camera started")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read frame")
        break

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.4:
            idx = int(detections[0, 0, i, 1])
            if idx >= len(CLASSES) or CLASSES[idx] != "person":
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            person_roi = frame[startY:endY, startX:endX]
            top_color, bottom_color, top_rgb, bottom_rgb = analyze_outfit_colors(person_roi)
            if top_color is None or bottom_color is None:
                continue

            show_color_feedback(frame, top_rgb, bottom_rgb, top_color, bottom_color)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            break  # only process first detected person

    cv2.imshow("Outfit Analyzer - Press 'q' to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import numpy as np
from sklearn.cluster import KMeans

# Load MobileNet SSD model
prototxt = "deploy.prototxt"
model = "mobilenet_iter_73000.caffemodel"

print("🔄 Loading model...")
try:
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Failed to load model:", e)
    exit()

# Class labels
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

def extract_top_bottom_regions(frame):
    h, w = frame.shape[:2]
    top = frame[int(h * 0.15):int(h * 0.40), int(w * 0.30):int(w * 0.70)]
    bottom = frame[int(h * 0.60):int(h * 0.85), int(w * 0.30):int(w * 0.70)]
    return top, bottom

def preprocess_image(region):
    region = cv2.resize(region, (100, 100))
    blurred = cv2.GaussianBlur(region, (7, 7), 0)
    return blurred

def detect_dominant_color(image, k=4):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(pixels)
    counts = np.bincount(labels)
    dominant_index = np.argmax(counts)
    dominant_color = kmeans.cluster_centers_[dominant_index]
    return dominant_color.astype(int)

def rgb_to_hsv(rgb):
    r, g, b = [x / 255.0 for x in rgb]
    mx = max(r, g, b)
    mn = min(r, g, b)
    diff = mx - mn

    if diff == 0:
        h = 0
    elif mx == r:
        h = (60 * ((g - b) / diff) + 360) % 360
    elif mx == g:
        h = (60 * ((b - r) / diff) + 120) % 360
    else:
        h = (60 * ((r - g) / diff) + 240) % 360

    s = 0 if mx == 0 else diff / mx
    v = mx
    return h, s, v

def get_closest_color(h, s, v):
    color_ranges = [
        ("red",     [(0, 10), (350, 360)], 0.5, 0.2, 1.0, 1.0),
        ("orange",  [(11, 40)], 0.5, 0.2, 1.0, 1.0),
        ("yellow",  [(41, 65)], 0.4, 0.4, 1.0, 1.0),
        ("green",   [(66, 170)], 0.4, 0.2, 1.0, 1.0),
        ("blue",    [(171, 260)], 0.4, 0.2, 1.0, 1.0),
        ("purple",  [(261, 320)], 0.3, 0.2, 1.0, 1.0),
        ("brown",   [(10, 40)], 0.3, 0.1, 0.7, 0.6),
    ]

    if v <= 0.15:
        return "black"
    if s <= 0.15 and v >= 0.85:
        return "white"
    if s <= 0.15 and 0.15 < v < 0.85:
        return "gray"

    for name, hue_ranges, sat_min, val_min, sat_max, val_max in color_ranges:
        for (h_min, h_max) in hue_ranges:
            if h_min <= h <= h_max and sat_min <= s <= sat_max and val_min <= v <= val_max:
                return name

    main_hues = {
        "red": 0,
        "orange": 25,
        "yellow": 55,
        "green": 120,
        "blue": 215,
        "purple": 290,
        "brown": 25,
    }

    def hue_distance(a, b):
        d = abs(a - b)
        return min(d, 360 - d)

    closest_color = None
    min_dist = 360
    for color, color_hue in main_hues.items():
        dist = hue_distance(h, color_hue)
        if dist < min_dist:
            min_dist = dist
            closest_color = color

    return closest_color

def suggest_outfit(top_color, bottom_color):
    matching = {
        "red": ["black", "white", "blue", "gray"],
        "orange": ["blue", "white", "black"],
        "yellow": ["blue", "gray", "black"],
        "green": ["black", "white", "brown"],
        "blue": ["white", "gray", "khaki"],
        "purple": ["black", "white", "gray"],
        "brown": ["white", "blue", "green"],
        "black": ["red", "yellow", "white"],
        "white": ["black", "blue", "red"],
        "gray": ["blue", "white", "black"],
    }
    if top_color in matching and bottom_color in matching[top_color]:
        return "✅ Great combo!"
    elif bottom_color in matching and top_color in matching[bottom_color]:
        return "✅ Looks good together!"
    else:
        return f"👕 Try pairing {top_color} with {', '.join(matching.get(top_color, []))}"

def analyze_outfit_colors(frame):
    top, bottom = extract_top_bottom_regions(frame)
    if top is None or bottom is None or top.size == 0 or bottom.size == 0:
        return None, None, None, None
    top = preprocess_image(top)
    bottom = preprocess_image(bottom)

    top_rgb = detect_dominant_color(top)
    bottom_rgb = detect_dominant_color(bottom)

    top_hsv = rgb_to_hsv(top_rgb)
    bottom_hsv = rgb_to_hsv(bottom_rgb)

    top_color = get_closest_color(*top_hsv)
    bottom_color = get_closest_color(*bottom_hsv)

    return top_color, bottom_color, top_rgb, bottom_rgb

def show_color_feedback(frame, top_rgb, bottom_rgb, top_color, bottom_color):
    suggestion = suggest_outfit(top_color, bottom_color)

    cv2.rectangle(frame, (10, 10), (100, 60), top_rgb[::-1].tolist(), -1)
    cv2.putText(frame, f"Top: {top_color}", (110, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.rectangle(frame, (10, 70), (100, 120), bottom_rgb[::-1].tolist(), -1)
    cv2.putText(frame, f"Bottom: {bottom_color}", (110, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.putText(frame, suggestion, (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

# Start camera
print("🎥 Opening camera...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()
print("✅ Camera started")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read frame")
        break

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.4:
            idx = int(detections[0, 0, i, 1])
            if idx >= len(CLASSES) or CLASSES[idx] != "person":
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            person_roi = frame[startY:endY, startX:endX]
            top_color, bottom_color, top_rgb, bottom_rgb = analyze_outfit_colors(person_roi)
            if top_color is None or bottom_color is None:
                continue

            show_color_feedback(frame, top_rgb, bottom_rgb, top_color, bottom_color)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            break  # only process first detected person

    cv2.imshow("Outfit Analyzer - Press 'q' to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
