from onvif import ONVIFCamera

camera_ip = "192.168.18.114"  # Replace with your camera's IP
port = 80  # ONVIF port, typically 80
username = "admin"
password = "admin"  # Try default values

try:
    camera = ONVIFCamera(camera_ip, port, username, password)
    print("Connected to the camera successfully!")

except Exception as e:
    print("Error:", e)