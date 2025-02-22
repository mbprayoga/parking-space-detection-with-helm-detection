from onvif import ONVIFCamera
import time
import cv2
import os

class Camera:
    def __init__(self, name, ip, port, user, password, move_start, move_distance):
        self.name = name
        self.ip = ip
        self.port = port
        self.user = user
        self.password = password
        self.ptz = None
        self.moverequest = None
        self.active = False
        self.setup_move()
        self.position = "A"
        self.move_start = move_start
        self.move_distance = move_distance

    def capture_image(self, filename):
        filepath = os.path.join("park-image", filename)
        cap = cv2.VideoCapture(f"rtsp://{self.ip}/live/ch00_0")

        if not cap.isOpened():
            print("Error: Could not open video stream.")
            return

        ret, frame = cap.read()
        if ret:
            cv2.imwrite(filepath, frame)
            print(f"Image saved as {filepath}")
        else:
            print("Error: Could not capture image.")

        cap.release()
    
    def setup_move(self):
        """Initializes PTZ services and configuration."""
        try:
            mycam = ONVIFCamera(self.ip, self.port, self.user, self.password, "wsdl")
            media = mycam.create_media_service()
            self.ptz = mycam.create_ptz_service()

            media_profile = media.GetProfiles()[0]

            request = self.ptz.create_type('GetConfigurationOptions')
            request.ConfigurationToken = media_profile.PTZConfiguration.token
            ptz_configuration_options = self.ptz.GetConfigurationOptions(request)

            self.moverequest = self.ptz.create_type('ContinuousMove')
            self.moverequest.ProfileToken = media_profile.token
            self.moverequest.Velocity = self.ptz.GetStatus({'ProfileToken': media_profile.token}).Position
        except Exception as e:
            print(f"Error in setup_move: {str(e)}")

    def do_move(self):
        """Executes continuous move with error handling."""
        try:
            if self.active:
                self.ptz.Stop({'ProfileToken': self.moverequest.ProfileToken})
            self.active = True
            self.ptz.ContinuousMove(self.moverequest)
        except Exception as e:
            print(f"Movement error: {str(e)}")
            self.active = False

    def set_movement(self, x, y, speed=0.3):
        """Sets movement with specified x, y velocities."""
        self.moverequest.Velocity.PanTilt.x = x * speed
        self.moverequest.Velocity.PanTilt.y = y * speed
        self.do_move()

    def stop_movement(self):
        """Stops the camera movement."""
        try:
            self.ptz.Stop({'ProfileToken': self.moverequest.ProfileToken})
            self.active = False
        except Exception as e:
            print(f"Stop movement error: {str(e)}")
    
    def move_to_position_b(self):
        if self.position != "B":
            time.sleep(5)
            for i in range(self.move_distance):
                self.set_movement(x=-1, y=0) 
            self.stop_movement()
            time.sleep(5)  
            self.position = "B"
            self.capture_image(f"{self.name}-B.png")
        
    def move_to_position_a(self):
        if self.position != "A":
            time.sleep(5)
            for i in range(self.move_distance):
                self.set_movement(x=1, y=0)  
            self.stop_movement()
            time.sleep(5)  
            self.position = "A"
            self.capture_image(f"{self.name}-A.png")
                
    def startup(self):
        time.sleep(5)
        # Move up
        for i in range(50):
            self.set_movement(x=0, y=1)
        
        # Move right
        for i in range(60):
            self.set_movement(x=1, y=0)
        
        # Move left
        for i in range(self.move_start):
            self.set_movement(x=-1, y=0)
        
        # Move down
        for i in range(12):
            self.set_movement(x=0, y=-1)
        
        time.sleep(5)    
        self.position = "A"
        self.capture_image(f"{self.name}-A.png")
        
        self.move_to_position_b()
        
        self.stop_movement()
        time.sleep(0.5) 
        
    def run(self, update_callback1, update_callback2, update_parking):
        try:
            self.startup()
            
            time.sleep(10)
            
            while True:
                self.move_to_position_a()
                update_callback2(update_parking)
                time.sleep(3)
                
                self.move_to_position_b()
                update_callback1(update_parking)
                time.sleep(10)
                
        except Exception as e:
            print(f"Run error: {str(e)}")
            self.stop_movement()