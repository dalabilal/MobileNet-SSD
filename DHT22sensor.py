# room_temperature.py

import board
import adafruit_dht

# Initialize DHT22 sensor on GPIO4
dht_sensor = adafruit_dht.DHT22(board.D4)

def get_room_temperature():
    try:
        temperature = dht_sensor.temperature
        humidity = dht_sensor.humidity
        if temperature is not None and humidity is not None:
            return f"The room temperature is {temperature:.1f}°C and the humidity is {humidity:.1f}%."
        else:
            return "Sorry, I couldn't read the temperature and humidity right now."
    except RuntimeError as error:
        return f"Sensor read error: {error.args[0]}"
    except Exception as e:
        return f"An error occurred while reading the temperature: {str(e)}"
