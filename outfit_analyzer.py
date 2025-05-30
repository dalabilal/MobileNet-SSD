import cv2
import numpy as np
from sklearn.cluster import KMeans
import time

def analyze_outfit():
    try:
        prototxt = "deploy.prototxt"
        model = "mobilenet_iter_73000.caffemodel"
        net = cv2.dnn.readNetFromCaffe(prototxt, model)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("I couldn't open  the camera.")
            return

        print("Camera opened. Showing for 10 seconds. Please stand still.")
        start_time = time.time()
        roi = None
        frame = None

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from camera.")
                break

            cv2.imshow("Outfit Analyzer", frame)

            # Exit after 10 seconds
            if time.time() - start_time > 10:
                break

            # Manual exit with 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Camera closed.")
                cap.release()
                cv2.destroyAllWindows()
                return

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                     0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        person_found = False
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.4:
                idx = int(detections[0, 0, i, 1])
                if idx != 15:  # 15 is person class in COCO
                    continue

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                startX, startY = max(0, startX), max(0, startY)
                endX, endY = min(w, endX), min(h, endY)

                # Draw a rectangle around the detected person
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

                # Draw top rectangle (shirt region)
                topY1 = int(startY + (endY - startY) * 0.15)
                topY2 = int(startY + (endY - startY) * 0.40)
                cv2.rectangle(frame, (startX + int((endX - startX) * 0.30), topY1),
                            (startX + int((endX - startX) * 0.70), topY2), (255, 0, 0), 2)

                # Draw bottom rectangle (pants region)
                botY1 = int(startY + (endY - startY) * 0.60)
                botY2 = int(startY + (endY - startY) * 0.85)
                cv2.rectangle(frame, (startX + int((endX - startX) * 0.30), botY1),
                            (startX + int((endX - startX) * 0.70), botY2), (0, 0, 255), 2)

                # Show the final frame with rectangles
                cv2.imshow("Detected Person and Outfit Regions", frame)
                cv2.waitKey(3000)  # Show for 3 seconds
                cv2.destroyAllWindows()

                # Extract ROI for analysis
                roi = frame[startY:endY, startX:endX]
                person_found = True
                break


        cap.release()
        cv2.destroyAllWindows()

        if person_found and roi is not None and roi.size > 0:
            analyze_colors_from_roi(roi)
        else:
            print("I couldn't detect a person to analyze.")

    except Exception as e:
        print(f"Sorry, I couldn't analyze your outfit. Error: {str(e)}")

def preprocess_image_cv(region):
    region = cv2.resize(region, (100, 100))
    blurred = cv2.GaussianBlur(region, (7, 7), 0)
    return blurred

def detect_dominant_color_cv(image, k=4):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
    labels = kmeans.fit_predict(pixels)
    counts = np.bincount(labels)
    dominant_index = np.argmax(counts)
    dominant_color = kmeans.cluster_centers_[dominant_index]
    return dominant_color.astype(int)

def rgb_to_hsv_cv(rgb):
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

def get_closest_color_cv(h, s, v):
    # Handle black, white, gray first
    if v <= 0.15:
        return "black"
    if s <= 0.15 and v >= 0.70:
        return "white"
    if s <= 0.15 and 0.15 < v < 0.70:
        return "gray"

    # Extended color ranges with light/dark variants
    color_ranges = [
        # name, hue ranges, sat min, val min, sat max, val max
        ("red", [(0, 10), (350, 360)], 0.5, 0.2, 1.0, 1.0),
        ("dark red", [(0, 10), (350, 360)], 0.5, 0.0, 1.0, 0.5),
        ("pink", [(320, 350)], 0.2, 0.7, 0.5, 1.0),
        ("orange", [(11, 40)], 0.5, 0.4, 1.0, 1.0),
        ("light orange", [(11, 40)], 0.3, 0.7, 0.5, 1.0),
        ("yellow", [(41, 65)], 0.4, 0.4, 1.0, 1.0),
        ("light yellow", [(41, 65)], 0.2, 0.7, 0.5, 1.0),
        ("green", [(66, 170)], 0.4, 0.2, 1.0, 1.0),
        ("dark green", [(66, 170)], 0.5, 0.0, 1.0, 0.5),
        ("blue", [(171, 260)], 0.4, 0.3, 1.0, 1.0),
        ("light blue", [(171, 260)], 0.2, 0.6, 0.5, 1.0),
        ("dark blue", [(171, 260)], 0.5, 0.0, 1.0, 0.4),
        ("purple", [(261, 320)], 0.3, 0.2, 1.0, 1.0),
        ("brown", [(10, 40)], 0.3, 0.1, 0.7, 0.6),
        ("light brown", [(25, 35)], 0.3, 0.7, 0.5, 0.9),
        ("medium brown", [(25, 35)], 0.4, 0.4, 0.8, 0.6),
        ("dark brown", [(25, 35)], 0.5, 0.2, 1.0, 0.4),
        ("pale beige", [(25, 35)], 0.1, 0.85, 0.3, 1.0),
        ("rosy", [(345, 360)], 0.2, 0.7, 0.4, 0.85),
        ("umber", [(10, 25)], 0.6, 0.3, 1.0, 0.5),
    ]

    for name, hue_ranges, sat_min, val_min, sat_max, val_max in color_ranges:
        for (h_min, h_max) in hue_ranges:
            # Handle hue wrap around
            if h_min > h_max:
                if (h >= h_min or h <= h_max) and sat_min <= s <= sat_max and val_min <= v <= val_max:
                    return name
            else:
                if h_min <= h <= h_max and sat_min <= s <= sat_max and val_min <= v <= val_max:
                    return name

    # Fallback - find closest by hue (ignoring saturation and value)
    main_hues = {
        "red": 0,
        "pink": 335,
        "orange": 25,
        "yellow": 55,
        "green": 120,
        "blue": 215,
        "light blue": 215,
        "purple": 290,
        "brown": 25,
        "light brown": 25,
        "medium brown": 25,
        "dark brown": 25,
        "pale beige": 30,
        "rosy": 350,
        "umber": 20,
        "black": 0,
        "white": 0,
        "gray": 0,
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



def analyze_colors_from_roi(image):
    h, w = image.shape[:2]
    top = image[int(h * 0.15):int(h * 0.40), int(w * 0.30):int(w * 0.70)]
    bottom = image[int(h * 0.60):int(h * 0.85), int(w * 0.30):int(w * 0.70)]

    if top.size == 0 or bottom.size == 0:
        print("I couldn't find your outfit clearly.")
        return

    top_pre = preprocess_image_cv(top)
    bottom_pre = preprocess_image_cv(bottom)

    top_rgb = detect_dominant_color_cv(top_pre)
    bottom_rgb = detect_dominant_color_cv(bottom_pre)

    top_hsv = rgb_to_hsv_cv(top_rgb)
    bottom_hsv = rgb_to_hsv_cv(bottom_rgb)

    top_color = get_closest_color_cv(*top_hsv)
    bottom_color = get_closest_color_cv(*bottom_hsv)

    print(f"Your shirt is {top_color}, and your pants is {bottom_color}.")

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
    
    business_combinations = [
    ("blue", "white"), ("white", "blue"),
    ("black", "white"), ("white", "black"),
    ("gray", "white"), ("white", "gray"),
    ("gray", "black"), ("black", "gray"),
    ("gray", "blue"), ("blue", "gray")
]
    if (top_color, bottom_color) in business_combinations:
        print("This outfit is suitable for a business interview or meeting.")
    elif top_color in matching and bottom_color in matching[top_color]:
        print("That's a great combination!")
    elif bottom_color in matching and top_color in matching[bottom_color]:
        print("Looks good together!")
    else:
        suggestion = ", ".join(matching.get(top_color, []))
        print(f"You can try {top_color} with {suggestion}.")

analyze_outfit()
