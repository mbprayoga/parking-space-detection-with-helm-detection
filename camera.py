from onvif import ONVIFCamera
import time

class Camera:
    def __init__(self, ip, port, user, password):
        self.ip = ip
        self.port = port
        self.user = user
        self.password = password
        self.ptz = None
        self.moverequest = None
        self.active = False
        self.setup_move()

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
        self.moverequest.Velocity.PanTilt.y = 1
        self.do_move()

    def move_down(self):
        """Moves the camera down."""
        self.moverequest.Velocity.PanTilt.x = 0
        self.moverequest.Velocity.PanTilt.y = -1
        self.do_move()

    def move_left(self):
        """Moves the camera left."""
        self.moverequest.Velocity.PanTilt.x = -1
        self.moverequest.Velocity.PanTilt.y = 0
        self.do_move()

    def move_right(self):
        """Moves the camera right."""
        self.moverequest.Velocity.PanTilt.x = 1
        self.moverequest.Velocity.PanTilt.y = 0
        self.do_move()

    def stop_movement(self):
        """Stops the camera movement."""
        self.ptz.Stop({'ProfileToken': self.moverequest.ProfileToken})
        self.active = False
        
    def start_position(self):
        for i in range(30):
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
            
    def move_to_position_b(self):
        for i in range(20):
            self.move_left()
            time.sleep(0.1)
            
    def move_to_position_a(self):
        for i in range(20):
            self.move_right()
            time.sleep(0.1)