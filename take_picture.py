
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
delay_before_capture = 3  # seconds
auto_delete_after = 30  # seconds
ngrok_url = "https://a5a5-24-42-76-134.ngrok-free.app"  # REPLACE WITH YOUR ngrok link

# === FUNCTIONS ===

def capture_image(filename):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera could not be opened.")
        return False
    print(f"⏳ Waiting {delay_before_capture} seconds before capturing...")
    time.sleep(delay_before_capture)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(filename, frame)
        print(f"✅ Image saved as {filename}")
    cap.release()
    return ret

def start_server(port=8000):
    handler = SimpleHTTPRequestHandler
    httpd = HTTPServer(("", port), handler)
    print(f"🌐 Serving at http://{get_local_ip()}:{port}")
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
    cv2.imshow("Scan this QR Code", img)
    cv2.waitKey(1)

def delete_after_delay(image_file, qr_file, delay):
    time.sleep(delay)
    for file in [image_file, qr_file]:
        if os.path.exists(file):
            os.remove(file)
            print(f"🗑️ {file} deleted.")
    cv2.destroyAllWindows()

# === MAIN ===

if capture_image(image_filename):
    threading.Thread(target=start_server, daemon=True).start()
    public_url = f"{ngrok_url}/{image_filename}"
    generate_qr(public_url, qr_filename)
    threading.Thread(
        target=delete_after_delay,
        args=(image_filename, qr_filename, auto_delete_after),
        daemon=True
    ).start()
