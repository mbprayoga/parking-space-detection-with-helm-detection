import keyboard
import time
from camera import Camera
import threading

def control_camera(camera1, camera2):
    """Controls the cameras based on keyboard input."""
    while True:
        if keyboard.is_pressed('up'):
            camera1.move_up()
            camera2.move_up()
        elif keyboard.is_pressed('down'):
            camera1.move_down()
            camera2.move_down()
        elif keyboard.is_pressed('left'):
            camera1.move_left()
            camera2.move_left()
        elif keyboard.is_pressed('right'):
            camera1.move_right()
            camera2.move_right()
        elif keyboard.is_pressed('a'):
            camera1.move_to_position_a()
            camera2.move_to_position_a()
        elif keyboard.is_pressed('b'):
            camera1.move_to_position_b()
            camera2.move_to_position_b()
        elif keyboard.is_pressed('q'):
            camera1.stop_movement()
            camera2.stop_movement()
            print("Exiting...")
            break
        elif keyboard.is_pressed('s'):
            camera1.stop_movement()
            camera2.stop_movement()
        # Add a small delay to prevent high CPU usage

if __name__ == '__main__':
    camera1 = Camera("192.168.18.114", 8899, "", "")
    camera2 = Camera("192.168.18.154", 8899, "", "")
    
    # threading.Thread(target=camera1.start_position).start()
    # threading.Thread(target=camera2.start_position).start()
    
    control_camera(camera1, camera2)
    