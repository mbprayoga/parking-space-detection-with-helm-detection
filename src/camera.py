from onvif import ONVIFCamera
import time
import cv2
import os

class Camera:
    def __init__(self, name, ip, port, user, password, move_start=32, move_distance=26, move_vertical=16):
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
        self.move_vertical = move_vertical

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
            self.mycam = ONVIFCamera(self.ip, self.port, self.user, self.password, "wsdl")
            
            time.sleep(2)
            
            media = self.mycam.create_media_service()
            self.ptz = self.mycam.create_ptz_service()

            self.profiles = media.GetProfiles()
            if not self.profiles:
                raise Exception("No media profiles found")
                
            self.media_profile = self.profiles[0]

            request = self.ptz.create_type('GetConfigurationOptions')
            request.ConfigurationToken = self.media_profile.PTZConfiguration.token
            ptz_configuration_options = self.ptz.GetConfigurationOptions(request)

            self.moverequest = self.ptz.create_type('ContinuousMove')
            self.moverequest.ProfileToken = self.media_profile.token
            self.moverequest.Velocity = self.ptz.GetStatus({'ProfileToken': self.media_profile.token}).Position
            
            print("PTZ initialization complete")
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
            for i in range(self.move_distance):
                self.set_movement(x=-1, y=0) 
            self.stop_movement()
            time.sleep(2)  
            self.position = "B"
            self.capture_image(f"{self.name}-B.png")
            
    def move_to_position_c(self):
        if self.position != "C":
            for i in range(self.move_vertical):
                self.set_movement(x=0, y=-1)
            for i in range(self.move_distance//2):
                self.set_movement(x=1, y=0) 
            self.stop_movement()
            time.sleep(2)  
            self.position = "C"
            self.capture_image(f"{self.name}-C.png")
        
    def move_to_position_a(self):
        if self.position != "A":
            for i in range(self.move_vertical):
                self.set_movement(x=0, y=1)
            for i in range(self.move_distance//2):
                self.set_movement(x=1, y=0)  
            self.stop_movement()
            time.sleep(2)  
            self.position = "A"
            self.capture_image(f"{self.name}-A.png")
                
    def startup(self, update_callback1=None, update_callback2=None, update_callback3=None, update_parking=None):
        # Move up
        for i in range(50):
            self.set_movement(x=0, y=1)
        
        # Move right
        for i in range(75):
            self.set_movement(x=1, y=0)
        
        # Move left
        for i in range(self.move_start):
            self.set_movement(x=-1, y=0)
        
        # Move down
        for i in range(12):
            self.set_movement(x=0, y=-1)
        
        time.sleep(5)    
        self.position = "A"
        
        while True:
            self.move_to_position_b()
            if update_callback2:
                update_callback2(update_parking)
            
            self.move_to_position_c()
            if update_callback3:
                update_callback3(update_parking)
            
            self.move_to_position_a()
            if update_callback1:
                update_callback1(update_parking)
            
            
        
     
    
    def get_presets(self):
        """Get all available presets."""
        try:
            if not self.ptz or not self.media_profile:
                print("PTZ service not initialized")
                return []
                
            request = self.ptz.create_type('GetPresets')
            request.ProfileToken = self.media_profile.token
            presets = self.ptz.GetPresets(request)
            
            if presets:
                print("\nAvailable presets:")
                for preset in presets:
                    print(f"- Name: {preset.Name}, Token: {preset.token}")
            else:
                print("No presets found")
                
            return presets
            
        except Exception as e:
            print(f"Error getting presets: {str(e)}")
            return []

    def goto_preset(self, preset_token):
        """Move to a specific preset position."""
        try:
            request = self.ptz.create_type('GotoPreset')
            request.ProfileToken = self.media_profile.token
            request.PresetToken = preset_token
            
            print(f"Moving to preset with token: {preset_token}")
            self.ptz.GotoPreset(request)
            
            time.sleep(2)
            
        except Exception as e:
            print(f"Error moving to preset: {str(e)}")

    def set_preset(self, preset_name):
        """Set current position as a new preset."""
        try:
            if not self.ptz or not self.media_profile:
                print("PTZ service not initialized")
                return None
                
            self.stop_movement()
            time.sleep(0.5)  
                
            request = self.ptz.create_type('SetPreset')
            request.ProfileToken = self.media_profile.token
            request.PresetName = preset_name
            
            status = self.ptz.GetStatus({'ProfileToken': self.media_profile.token})
            print(f"Current position: {status.Position.PanTilt.x}, {status.Position.PanTilt.y}")
            
            preset_token = self.ptz.SetPreset(request)
            if preset_token:
                print(f"Successfully set preset '{preset_name}' with token {preset_token}")
                return preset_token
            else:
                print("Failed to set preset - no token received")
                return None
        
        except Exception as e:
            print(f"Error setting preset: {str(e)}")
            return None

    def handle_preset_movement(self):
        """Handle keyboard input for preset movement and setting."""
        try:
            import keyboard
            
            time.sleep(3)
            
            print("\nGetting presets...")
            presets = self.get_presets()
            
            print("\nControls:")
            print("1-9: Move to preset")
            print("CTRL+1-9: Set new preset")
            print("Q: Quit preset control")
            
            while True:
                try:
                    if keyboard.is_pressed('q'):
                        print("Exiting preset control")
                        break
                        
                    if keyboard.is_pressed('ctrl'):
                        for i in range(9):
                            if keyboard.is_pressed(str(i + 1)):
                                preset_name = f"Preset_{i + 1}"
                                print(f"\nSetting new preset: {preset_name}")
                                
                                self.stop_movement()
                                time.sleep(1)  
                                token = self.set_preset(preset_name)
                                time.sleep(1)  
                                if token:
                                    presets = self.get_presets()  
                                    
                    
                    else:
                        if presets:  
                            for i, preset in enumerate(presets):
                                if keyboard.is_pressed(str(i + 1)):
                                    print(f"Moving to preset {preset.Name}")
                                    
                                    self.stop_movement()
                                    time.sleep(0.5)
                                    self.goto_preset(preset.token)
                                    time.sleep(1)  
                                    
                except Exception as e:
                    print(f"Error in keyboard handling: {str(e)}")
                    self.stop_movement()  
                    break
                        
        except Exception as e:
            print(f"Error in preset movement: {str(e)}")
            self.stop_movement()  