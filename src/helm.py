import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import hailo
import threading
import serial
import time

from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from hailo_apps_infra.detection_pipeline import GStreamerDetectionApp

class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()

class HelmetDetection:
    def __init__(self,update_callback=None):
        print("cek-helm3")
        self.user_data = user_app_callback_class()
        self.user_data.use_frame = True
        self.update_callback = update_callback
        self.app = GStreamerDetectionApp(self.app_callback, self.user_data)

        self.serial_connection = None
        self.serial_port = '/dev/ttyACM1'
        self.serial_baudrate = 115200

        self.lock = threading.Lock()
        self.latest_detections = None

        try:
            self.serial_connection = serial.Serial(self.serial_port, self.serial_baudrate)
        except:
            print("Serial connection failed")
            exit(1)

    def send_serial(self, initial_detections):
        try:     
            if self.serial_connection:
                # First check if helmet is detected
                if initial_detections['helm']:
                
                    time.sleep(3)  # Wait for 3 seconds
                    
                    # Get fresh detection results after delay
                    with self.lock:
                        if (self.latest_detections and self.latest_detections['helm'] and not self.latest_detections['tanpa-helm']):
                            self.serial_connection.write(b'1')
                            self.serial_connection.flush()       
            time.sleep(0.1)    

        except Exception as e:
            print(f"Error: {e}")


    def app_callback(self, pad, info, user_data):
        buffer = info.get_buffer()
        if buffer is None:
            return Gst.PadProbeReturn.OK

        format, width, height = get_caps_from_pad(pad)

        frame = None
        if user_data.use_frame and format is not None and width is not None and height is not None:
            frame = get_numpy_from_buffer(buffer, format, width, height)

        roi = hailo.get_roi_from_buffer(buffer)
        detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

        valid_labels = ['helm', 'pejalan', 'pemotor', 'tanpa-helm']
        confidence_threshold = 0.7

        detection_results = {
            'helm': False,
            'pejalan': False,
            'pemotor': False,
            'tanpa-helm': False
        }

        if frame is not None:
            for detection in detections:
                label = detection.get_label()
                confidence = detection.get_confidence()
                
                if label and label in valid_labels and confidence > confidence_threshold:
                    detection_results[label] = True
            
            
            with self.lock:
                self.latest_detections = detection_results.copy()
            
            serial_thread = threading.Thread(target=self.send_serial, args=(detection_results,))
            serial_thread.daemon = True
            serial_thread.start()

            if self.update_callback:
                self.update_callback(detection_results, frame)

            # with self.lock:
            #     self.detection_results = detection_results
        
        return Gst.PadProbeReturn.OK
