import subprocess

def get_room_temperature():
    try:
        dht22_python = "/home/smart/MagicMirror/modules/MMM-DHT22-Py/venv/bin/python"
        dht22_script = "/home/smart/MagicMirror/modules/MMM-DHT22-Py/dht22_reader.py"

        output = subprocess.check_output([dht22_python, dht22_script])
        output = output.decode("utf-8").strip()

        return f"The current {output.lower()}."
    except Exception as e:
        print(f"[DHT22 ERROR] {e}")
        return "Sorry, I couldn't get the room temperature right now."
