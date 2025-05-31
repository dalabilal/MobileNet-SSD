import subprocess
from click import command
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
from DHT22sensor import get_room_temperature

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
    GPIO.setup(17, GPIO.IN)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not accessible.")
        speak("Camera not accessible. Please check the connection.")
        return False

    print("Waiting for face detection...")
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
        if (len(faces) > 0 and GPIO.input(17)):
                print("Face detected!")
                speak("Hi there! How can I assist you?")
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
        BASE_DIR = "/home/smart/Desktop/voice/voic_assistant/MobileNet-SSD"
        prototxt = os.path.join(BASE_DIR, "deploy.prototxt")
        model = os.path.join(BASE_DIR, "mobilenet_iter_73000.caffemodel")
        net = cv2.dnn.readNetFromCaffe(prototxt, model)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("I couldn't open the camera.")
            return

        speak("Camera opened. Showing for 10 seconds. Please stand still.")
       
        start_time = time.time()
        roi = None
        frame = None
  
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from camera.")
                break

          
           
            # Exit after 10 seconds
            remaining = int(10 - (time.time() - start_time))
            if remaining < 0:
                break
            cv2.putText(frame, f"{remaining} second", (30, 60),cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)     
            
            cv2.imshow("Outfit Analyzer", frame)
            
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
            speak("I couldn't detect a person to analyze.")

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
        ("pink", [(320, 350)], 0.2, 0.7, 0.5, 1.0),
        ("orange", [(11, 40)], 0.5, 0.4, 1.0, 1.0),
        ("yellow", [(41, 65)], 0.4, 0.4, 1.0, 1.0),
        ("green", [(66, 170)], 0.4, 0.2, 1.0, 1.0),
        ("blue", [(171, 260)], 0.4, 0.3, 1.0, 1.0),
        ("purple", [(261, 320)], 0.3, 0.2, 1.0, 1.0),
        ("brown", [(10, 40)], 0.3, 0.1, 0.7, 0.6),
        ("pale beige", [(25, 35)], 0.1, 0.85, 0.3, 1.0),
        ("rosy", [(345, 360)], 0.2, 0.7, 0.4, 0.85),
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
        "purple": 290,
        "brown": 25,
        "pale beige": 30,
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

    speak(f"Your shirt is {top_color}, and your pants is {bottom_color}.")

    matching = {
        "red": ["black", "white", "gray"],
        "orange": ["white", "black"],
        "yellow": ["white" ,"gray", "black"],
        "green": ["black", "white", "brown"],
        "blue": ["white", "gray", "khaki"],
        "purple": ["black", "white", "gray"],
        "brown": ["white", "black", "green"],
        "black": ["red", "blue", "white"],
        "white": ["black", "blue", "red"],
        "gray": ["red", "white", "black"],
        "pink": ["gray", "white", "black"],
        "pale beige": ["brown", "white", "black"],
    }
    
    business_combinations = [
    ("blue", "white"), ("white", "blue"),
    ("black", "white"), ("white", "black"),
    ("gray", "white"), ("white", "gray"),
    ("gray", "black"), ("black", "gray"),
    ("gray", "blue"), ("blue", "gray")
]
    if (top_color, bottom_color) in business_combinations:
        speak("This outfit is suitable for a business interview or meeting.")
    elif top_color in matching and bottom_color in matching[top_color]:
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
    elif "take picture" in user_input:
        subprocess.call(["/home/smart/Desktop/voice/rhasspy/bin/python","/home/smart/Desktop/voice/voic_assistant/MobileNet-SSD/take_picture.py"])
        speak("QR code has expired. Enjoy!")
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
    elif "room temperature" in user_input or "temperature" in user_input:
        speak(get_room_temperature())
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
    # speak("Marvin voice assistant is ready. Say 'Marvin' to activate me.")
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
