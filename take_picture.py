import cv2
import qrcode
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
import os
import socket
import time

# === SETTINGS ===
image_filename = "image.jpg"
qr_filename = "qr.png"
delay_before_capture = 10  # seconds
display_duration = 30      # seconds
ngrok_url = "https://2080-24-42-76-134.ngrok-free.app"  # Your ngrok HTTPS URL

# === Optional speak() function (remove if you're using your voice assistant)
def speak(text):
    print(f"[Assistant]: {text}")

# === FUNCTIONS ===

def capture_image(filename):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera could not be opened.")
        return False

    speak(f"Please stand in front of the camera. Capturing in {delay_before_capture} seconds.")
    start_time = time.time()
    while time.time() - start_time < delay_before_capture:
        ret, frame = cap.read()
        if not ret:
            continue

        # Draw countdown number
        remaining = int(delay_before_capture - (time.time() - start_time)) + 1
        cv2.putText(frame, f"Capturing in {remaining}s", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.imshow("Camera Preview", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return False

    # Capture final frame
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(filename, frame)
        print(f"✅ Image saved as {filename}")
    cap.release()
    cv2.destroyAllWindows()
    return ret

def start_server(port=8000):
    handler = SimpleHTTPRequestHandler
    httpd = HTTPServer(("", port), handler)
    print(f"🌐 Server running at http://{get_local_ip()}:{port}")
    httpd.serve_forever()

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    finally:
        s.close()

def generate_qr(url, qr_filename):
    qr = qrcode.make(url)
    qr.save(qr_filename)
    print(f"📷 QR Code saved as {qr_filename}")
    img = cv2.imread(qr_filename)
    speak("Scan the QR code. You have 30 seconds.")
    cv2.imshow("Scan this QR Code", img)
    print(f"⏳ QR code will be visible for {display_duration} seconds...")
    cv2.waitKey(display_duration * 1000)
    cv2.destroyAllWindows()

def delete_files(*files):
    for file in files:
        if os.path.exists(file):
            os.remove(file)
            print(f"🗑️ {file} deleted.")
    print("✅ Cleanup complete.")

# === MAIN PROCESS ===
def take_picture_qr_process():
    # Start local server first
    threading.Thread(target=start_server, daemon=True).start()
    time.sleep(1)  # Give server time to start

    if capture_image(image_filename):
        public_url = f"{ngrok_url}/{image_filename}"
        generate_qr(public_url, qr_filename)
        delete_files(image_filename, qr_filename)

# === Entry point ===
if __name__ == "__main__":
    take_picture_qr_process()
