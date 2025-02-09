import keyboard
import time
from camera import Camera
import threading


if __name__ == '__main__':
    camera1 = Camera("NW", "192.168.18.114", 8899, "", "")
    camera2 = Camera("SE", "192.168.18.154", 8899, "", "")
    
    threading.Thread(target=camera1.start).start()
    threading.Thread(target=camera2.start).start()
    
    
    