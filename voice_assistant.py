import speech_recognition as sr
from gtts import gTTS
import datetime
import os
import requests
import wikipedia
import time
import logging
from PIL import Image
import random
import cv2
import numpy as np
from sklearn.cluster import KMeans
import RPi.GPIO as GPIO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("marvin_assistant.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MarvinAssistant")

# === CONFIG ===
WAKE_WORD = "marvin"
WEATHER_CITY = "Palestine"
WEATHER_API_KEY = "ac1ee19d6a4008e44838765e5e0c81dc"

def update_status(status):
    try:
        with open("/home/smart/MagicMirror/modules/MMM-AssistantFeedback/status.txt", "w") as f:
            f.write(status)
    except:
        pass

# === Text-to-Speech ===
def speak(text):
    update_status("speaking")
    logger.info(f"Assistant: {text}")
    print("Assistant:", text)
    try:
        tts = gTTS(text=text, lang='en')
        tts.save("output.mp3")
        if os.name == 'posix':
            os.system("mpg321 output.mp3 || afplay output.mp3")
        else:
            os.system("start output.mp3")
        time.sleep(0.5)
        update_status("idle")
    except Exception as e:
        logger.error(f"TTS error: {str(e)}")
        print(f"Error with text-to-speech: {str(e)}")

def wait_for_face():
    PIR_PIN = 17
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(PIR_PIN, GPIO.IN)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not accessible.")
        speak("Camera not accessible. Please check the connection.")
        return False

    # print("Waiting for face detection...")
    # speak("I am ready. Please show your face to start.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        command = listen()
        if command and WAKE_WORD in command:
            speak("Yes? How can I assist you?")
            cap.release()
            cv2.destroyAllWindows()
            return True
        
        # If face is detected
        if (len(faces) > 0 and GPIO.input(PIR_PIN)):
                print("Face detected!")
                speak("Hello! How can I assist you?")
                cap.release()
                cv2.destroyAllWindows()
                return True

        # cv2.imshow("Face Detection (Press Q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return False
# === Speech Recognition ===
def listen():
    update_status("listening")
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            print("Listening for input...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            recognizer.energy_threshold = 4000
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")
            logger.info(f"User said: {text}")
            update_status("idle")
            return text.lower()
    except sr.UnknownValueError:
        print("Could not understand audio")
        return None
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        return None
    except Exception as e:
        print(f"Error in speech recognition: {e}")
        return None
def wait_for_wake_word():
    print(f"Waiting for wake word '{WAKE_WORD}'...")
    while True:
        command = listen()
        if command and WAKE_WORD in command:
            speak("Yes? How can I assist you?")
            return True
# === Outfit Analyzer Function ===
def analyze_outfit():
    try:
        prototxt = "deploy.prototxt"
        model = "mobilenet_iter_73000.caffemodel"
        net = cv2.dnn.readNetFromCaffe(prototxt, model)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            speak("I couldn't open the camera.")
            return

        speak("Camera opened. Showing for 10 seconds. Please stand still.")
        start_time = time.time()
        roi = None
        frame = None

        while True:
            ret, frame = cap.read()
            if not ret:
                speak("Failed to read from camera.")
                break

            cv2.imshow("Outfit Analyzer", frame)

            # Exit after 5 seconds
            if time.time() - start_time > 10:
                break

            # Also allow manual exit with 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                speak("Camera closed.")
                cap.release()
                cv2.destroyAllWindows()
                return

        # Run person detection and analysis after camera closes
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
                if idx != 15:  # Not a person
                    continue

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                startX, startY = max(0, startX), max(0, startY)
                endX, endY = min(w, endX), min(h, endY)

                roi = frame[startY:endY, startX:endX]
                person_found = True
                break

        cap.release()
        cv2.destroyAllWindows()

        if person_found and roi is not None and roi.size > 0:
            analyze_colors_from_roi(roi)
        else:
            speak("I couldn't detect a person to analyze.")

    except Exception as e:
        speak("Sorry, I couldn't analyze your outfit.")
        logger.error(f"Outfit analysis error: {str(e)}")

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

def analyze_colors_from_roi(image):
    h, w = image.shape[:2]
    top = image[int(h * 0.15):int(h * 0.40), int(w * 0.30):int(w * 0.70)]
    bottom = image[int(h * 0.60):int(h * 0.85), int(w * 0.30):int(w * 0.70)]

    if top.size == 0 or bottom.size == 0:
        speak("I couldn't find your outfit clearly.")
        return

    top_pre = preprocess_image_cv(top)
    bottom_pre = preprocess_image_cv(bottom)

    top_rgb = detect_dominant_color_cv(top_pre)
    bottom_rgb = detect_dominant_color_cv(bottom_pre)

    top_hsv = rgb_to_hsv_cv(top_rgb)
    bottom_hsv = rgb_to_hsv_cv(bottom_rgb)

    top_color = get_closest_color_cv(*top_hsv)
    bottom_color = get_closest_color_cv(*bottom_hsv)

    speak(f"Your shirt is {top_color}, and your pants is {bottom_color}.")

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
        speak("That's a great combination!")
    elif bottom_color in matching and top_color in matching[bottom_color]:
        speak("Looks good together!")
    else:
        suggestion = ", ".join(matching.get(top_color, []))
        speak(f"You can try {top_color} with {suggestion}.")
# === Weather Info ===
def get_weather(city=WEATHER_CITY):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&units=metric&appid={WEATHER_API_KEY}"
    try:
        response = requests.get(url)
        data = response.json()
        if data["cod"] == 200:
            temp = data["main"]["temp"]
            desc = data["weather"][0]["description"]
            humidity = data["main"]["humidity"]
            wind = data["wind"]["speed"]
            return f"The current temperature in {city} is {temp}°C with {desc}. Humidity is {humidity}% and wind speed is {wind} meters per second."
        else:
            return "Sorry, I couldn't get the weather."
    except Exception as e:
        logger.error(f"Weather API error: {str(e)}")
        return "Error getting weather information."

# === Wikipedia Search ===
def search_wikipedia(query):
    try:
        return wikipedia.summary(query, sentences=2)
    except wikipedia.exceptions.DisambiguationError as e:
        return f"There are multiple options. Try asking about: {e.options[0]} or {e.options[1]}"
    except wikipedia.exceptions.PageError:
        return f"Sorry, I couldn't find information about {query}."
    except Exception as e:
        logger.error(f"Wikipedia search error: {str(e)}")
        return "Sorry, I encountered an error searching for that information."

# === General Knowledge ===
def get_general_answer(query):
    if "your name" in query:
        return "My name is Marvin, your personal voice assistant."
    elif "created you" in query or "made you" in query:
        return "I was created as a helpful voice assistant."
    elif "can you do" in query:
        return "I can tell you the time, check the weather, search for information, and answer general questions."
    elif "thank you" in query or "thanks" in query:
        return "You're welcome! Is there anything else I can help with?"
    return None

# === Input Handler ===
def handle_user_input(user_input):
    if "hello" in user_input or "hi" in user_input:
        speak("Hello! How can I help you today?")
    elif "how are you" in user_input:
        speak("I'm doing great, thank you for asking! How can I assist you?")
    elif "time" in user_input:
        now = datetime.datetime.now().strftime("%I:%M %p")
        speak(f"The time is {now}")
    elif "date" in user_input:
        today = datetime.datetime.now().strftime("%A, %B %d, %Y")
        speak(f"Today is {today}")
    elif "weather" in user_input:
        city = WEATHER_CITY
        words = user_input.split()
        for i, word in enumerate(words):
            if word == "in" and i < len(words) - 1:
                city = words[i + 1].capitalize()
        speak(get_weather(city))
    elif any(word in user_input for word in ["look up", "search", "tell me about", "what is", "who is"]):
        query = user_input
        for phrase in ["tell me about", "look up", "search for", "what is", "who is"]:
            if phrase in user_input:
                query = user_input.split(phrase, 1)[1].strip()
                break
        speak(f"Searching for information about {query}")
        speak(search_wikipedia(query))
    elif "see me" in user_input or "analyze outfit" in user_input:
        speak("Analyzing your outfit now.")
        analyze_outfit()
    elif any(word in user_input for word in ["exit", "stop", "goodbye", "bye"]):
        speak("Goodbye! Have a great day!")
        return False
    else:
        answer = get_general_answer(user_input)
        if answer:
            speak(answer)
        else:
            speak("I'm not sure how to help with that. You can ask me about the weather, time, or search for information.")
    return True

# === Main Loop ===
def voice_assistant():
    speak("Marvin voice assistant is ready. Say 'Marvin' to activate me.")
    try:
        while True:
            if wait_for_face():
                active = True
                while active:
                    user_input = listen()
                    if user_input:
                        active = handle_user_input(user_input)
                    else:
                        speak("I didn't catch that. Could you repeat?")
                speak("Marvin is now in standby mode. Say 'Marvin' to activate me again.")
    except KeyboardInterrupt:
        speak("Voice assistant shutting down. Goodbye!")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        speak("I encountered an unexpected error and need to restart.")

# Start the assistant
voice_assistant()