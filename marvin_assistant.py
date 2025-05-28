import cv2
import numpy as np
from sklearn.cluster import KMeans
import speech_recognition as sr
from gtts import gTTS
import os
import tempfile
import datetime
import wikipedia
import requests

# === CONFIG ===
OPENWEATHER_API_KEY = "YOUR_OPENWEATHERMAP_API_KEY"  # Replace with your OpenWeatherMap API key
CITY_NAME = "London"  # Change to your city

# === Load MobileNet model files ===
prototxt = "deploy.prototxt"
model = "mobilenet_iter_73000.caffemodel"

print("🔄 Loading model...")
try:
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Failed to load model:", e)
    exit()

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# === SPEECH FUNCTIONS ===

def speak(text):
    print("Assistant says:", text)
    try:
        tts = gTTS(text=text, lang='en')
        with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as fp:
            tts.save(fp.name)
            if os.name == "nt":
                os.system(f'start /min wmplayer "{fp.name}"')
            else:
                os.system(f"mpg123 {fp.name} 2> /dev/null")
    except Exception as e:
        print("Error in speaking:", e)

def listen_command(timeout=5, phrase_time_limit=5):
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        print("🎧 Listening...")
        try:
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            command = recognizer.recognize_google(audio).lower()
            print(f"🗣 You said: {command}")
            return command
        except sr.WaitTimeoutError:
            return ""
        except sr.UnknownValueError:
            print("❌ Could not understand audio.")
            return ""
        except sr.RequestError as e:
            print(f"❌ Speech Recognition error: {e}")
            return ""

# === DATE & TIME ===

def get_date_time():
    now = datetime.datetime.now()
    date_str = now.strftime("%A, %B %d, %Y")
    time_str = now.strftime("%I:%M %p")
    return f"Today is {date_str} and the time is {time_str}."

# === WIKIPEDIA SEARCH ===

def search_wikipedia(query):
    try:
        summary = wikipedia.summary(query, sentences=2)
        return summary
    except Exception as e:
        return "Sorry, I could not find any information on that."

# === WEATHER INFO ===

def get_weather(city):
    try:
        url = (f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric")
        response = requests.get(url)
        data = response.json()
        if data.get("cod") != 200:
            return "Sorry, I could not get the weather information."
        temp = data["main"]["temp"]
        desc = data["weather"][0]["description"]
        return f"The current temperature in {city} is {temp} degrees Celsius with {desc}."
    except Exception as e:
        return "Sorry, I could not get the weather information."

# === OUTFIT ANALYZER FUNCTIONS ===

def extract_top_bottom_regions(frame):
    h, w = frame.shape[:2]
    top = frame[int(h * 0.15):int(h * 0.40), int(w * 0.30):int(w * 0.70)]
    bottom = frame[int(h * 0.60):int(h * 0.85), int(w * 0.30):int(w * 0.70)]
    if top.size == 0 or bottom.size == 0:
        return None, None
    return top, bottom

def preprocess_image(region):
    region = cv2.resize(region, (100, 100))
    blurred = cv2.GaussianBlur(region, (7, 7), 0)
    return blurred

def detect_dominant_color(image, k=4):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
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
        ("red", [(0, 10), (350, 360)], 0.5, 0.2, 1.0, 1.0),
        ("orange", [(11, 40)], 0.5, 0.2, 1.0, 1.0),
        ("yellow", [(41, 65)], 0.4, 0.4, 1.0, 1.0),
        ("green", [(66, 170)], 0.4, 0.2, 1.0, 1.0),
        ("blue", [(171, 260)], 0.4, 0.2, 1.0, 1.0),
        ("purple", [(261, 320)], 0.3, 0.2, 1.0, 1.0),
        ("brown", [(10, 40)], 0.3, 0.1, 0.7, 0.6),
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
    if top is None or bottom is None:
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

    # Draw top color rectangle and label
    cv2.rectangle(frame, (10, 10), (100, 60), top_rgb[::-1].tolist(), -1)
    cv2.putText(frame, f"Top: {top_color}", (110, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

    # Draw bottom color rectangle and label
    cv2.rectangle(frame, (10, 70), (100, 120), bottom_rgb[::-1].tolist(), -1)
    cv2.putText(frame, f"Bottom: {bottom_color}", (110, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

    # Draw suggestion text
    cv2.putText(frame, suggestion, (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)

    return suggestion

def run_outfit_check():
    print("🎥 Opening camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        speak("Cannot open camera.")
        return

    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    print("✅ Camera started. Say 'close' to stop.")
    speak("Camera started. Say close to stop the outfit check.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Can't receive frame. Exiting...")
            break

        # Detect person in frame (optional - skipping for speed)
        # Instead, just analyze center region colors
        top_color, bottom_color, top_rgb, bottom_rgb = analyze_outfit_colors(frame)
        if None in (top_color, bottom_color, top_rgb, bottom_rgb):
            text = "Could not detect outfit colors."
            cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        else:
            suggestion = show_color_feedback(frame, top_rgb, bottom_rgb, top_color, bottom_color)
            text = f"Top: {top_color}, Bottom: {bottom_color}. {suggestion}"
            print(text)
            speak(text)

        cv2.imshow("Outfit Check (Say close to exit)", frame)

        # Keyboard check for 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Check voice command for "close"
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.2)
            try:
                audio = recognizer.listen(source, timeout=1, phrase_time_limit=3)
                command = recognizer.recognize_google(audio).lower()
                if "close" in command:
                    print("🛑 Closing camera by voice command")
                    break
            except sr.WaitTimeoutError:
                pass
            except sr.UnknownValueError:
                pass
            except sr.RequestError as e:
                print("Speech Recognition error:", e)

    cap.release()
    cv2.destroyAllWindows()
    speak("Camera closed. Goodbye!")

# === MAIN LISTENER LOOP ===

def main():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    speak("Hello! I am Marvin, your assistant. Say 'marvin' to wake me up.")

    while True:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.3)
            print("🎧 Listening for wake word...")
            audio = recognizer.listen(source)

        try:
            text = recognizer.recognize_google(audio).lower()
            print(f"🗣 Heard: {text}")

            if "marvin" in text:
                speak("Yes, how can I help you?")
                command = listen_command()
                if not command:
                    continue

                # Process commands
                if "blue" in command:
                    run_outfit_check()
                elif "time" in command or "date" in command:
                    dt = get_date_time()
                    speak(dt)
                elif "wikipedia" in command:
                    # Extract topic after 'wikipedia'
                    topic = command.replace("wikipedia", "").strip()
                    if topic:
                        result = search_wikipedia(topic)
                        speak(result)
                    else:
                        speak("Please tell me what to search on Wikipedia.")
                elif "weather" in command:
                    weather_info = get_weather(CITY_NAME)
                    speak(weather_info)
                elif "exit" in command or "quit" in command:
                    speak("Goodbye!")
                    break
                else:
                    speak("Sorry, I did not understand. Please say check my outfit, time, wikipedia, or weather.")

        except sr.UnknownValueError:
            print("❌ Could not understand audio.")
        except sr.RequestError as e:
            print(f"❌ Speech Recognition error; {e}")

main()
