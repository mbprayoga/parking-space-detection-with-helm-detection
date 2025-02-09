from onvif import ONVIFCamera
import time
import cv2
import os

class Camera:
    def __init__(self, name, ip, port, user, password):
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

    def capture_image(self, filename):
        # Open the video stream
        filepath = os.path.join("park-image", filename)
        
        cap = cv2.VideoCapture(f"rtsp://{self.ip}/live/ch00_0")  # Replace 0 with the actual video source for this camera

        if not cap.isOpened():
            print("Error: Could not open video stream.")
            return

        # Capture a single frame
        ret, frame = cap.read()

        if ret:
            # Save the frame as a PNG file
            cv2.imwrite(filepath, frame)
            print(f"Image saved as {filepath}")
        else:
            print("Error: Could not capture image.")

        # Release the video stream
        cap.release()
    
    def setup_move(self):
        """Initializes PTZ services and configuration."""
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

    def do_move(self):
        """Executes continuous move."""
        if self.active:
            self.ptz.Stop({'ProfileToken': self.moverequest.ProfileToken})
        self.active = True
        self.ptz.ContinuousMove(self.moverequest)

    def move_up(self):
        """Moves the camera up."""
        self.moverequest.Velocity.PanTilt.x = 0
        self.moverequest.Velocity.PanTilt.y = 0.5
        self.do_move()

    def move_down(self):
        """Moves the camera down."""
        self.moverequest.Velocity.PanTilt.x = 0
        self.moverequest.Velocity.PanTilt.y = -0.5
        self.do_move()

    def move_left(self):
        """Moves the camera left."""
        self.moverequest.Velocity.PanTilt.x = -0.5
        self.moverequest.Velocity.PanTilt.y = 0
        self.do_move()

    def move_right(self):
        """Moves the camera right."""
        self.moverequest.Velocity.PanTilt.x = 0.5
        self.moverequest.Velocity.PanTilt.y = 0
        self.do_move()

    def stop_movement(self):
        """Stops the camera movement."""
        self.ptz.Stop({'ProfileToken': self.moverequest.ProfileToken})
        self.active = False
    
    def move_to_position_b(self):
        if self.position != "B":
            for i in range(20):
                self.move_left()
            time.sleep(2)
            self.position = "B"
            self.capture_image(f"{self.name}-B.png")
        
    def move_to_position_a(self):
        if self.position != "A":
            for i in range(20):
                self.move_right()
            time.sleep(2)
            self.position = "A"
            self.capture_image(f"{self.name}-A.png")
                
    def startup(self):
        for i in range(40):
            self.move_up()
            time.sleep(0.1)
        
        for i in range(50):
            self.move_right()
            time.sleep(0.1)
            
        for i in range(15):
            self.move_left()
            time.sleep(0.1)
            
        for i in range(15):
            self.move_down()
            time.sleep(0.1)
        
    def run(self, update_callback1, update_callback2, update_parking):
        self.startup()
        
        self.move_to_position_b()
        update_callback2(update_parking)
        time.sleep(15)
        self.move_to_position_a()
        update_callback1(update_parking)
        time.sleep(60)
        
        while True:
            self.move_to_position_b()
            update_callback2(update_parking)
            time.sleep(60)  
            self.move_to_position_a()
            update_callback1(update_parking)
            time.sleep(60) 
