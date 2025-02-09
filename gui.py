import customtkinter as ctk
from PIL import Image
import cv2
import threading
from natsort import natsorted

from helm import HelmetDetection
from park import ParkingDetectionModel
from camera import Camera

class ParkingDetectionGUI:
    def __init__(self):
        self.parking_model1 = ParkingDetectionModel('park-image/NW-A', 'segment/segment-1')
        self.parking_model2 = ParkingDetectionModel('park-image/NW-B', 'segment/segment-2')
        self.parking_model3 = ParkingDetectionModel('park-image/SE-A', 'segment/segment-3')
        self.parking_model4 = ParkingDetectionModel('park-image/SE-B', 'segment/segment-4')
        
        self.helmet_model = HelmetDetection('models/nano/best.onnx', 0.5, 0.5)

        self.image_source1 = "test-image1.png"
        self.image_source2 = "test-image2.png"
        
        self.camera1 = Camera("NW", "192.168.18.114", 8899, "", "")
        self.camera2 = Camera("SE", "192.168.18.154", 8899, "", "")
        
        ctk.set_appearance_mode("light")

        self.app = ctk.CTk()
        self.app.title("Parking and Helmet Detection System")
        self.app.state('zoomed')
        
        self.app.geometry("1024x600")

        self.left_frame = ctk.CTkFrame(self.app, width=360, height=360, fg_color="transparent")
        self.left_frame.pack(side="left", padx=0, pady=0)

        self.right_frame = ctk.CTkFrame(self.app, width=400, fg_color="transparent")
        self.right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        self.area_labels_frame = ctk.CTkFrame(self.right_frame, width=400, fg_color="transparent")
        self.area_labels_frame.pack(side="right", expand=True, anchor="center")

        self.video_frame = ctk.CTkFrame(self.left_frame, fg_color="transparent")
        self.video_frame.pack(padx=10, pady=10)
        
        self.video_label_helmet = ctk.CTkLabel(self.video_frame, text="", anchor="center", fg_color="transparent")
        self.video_label_helmet.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.message_label = ctk.CTkLabel(self.video_frame, text="", anchor="center", fg_color="transparent", corner_radius=8)
        self.message_label.pack(fill="both", expand=True, padx=5, pady=5)

        self.image_frame = ctk.CTkFrame(self.left_frame, fg_color="transparent")
        self.image_frame.pack(fill="both", expand=True, padx=0, pady=0)
        
        self.image_frame1 = ctk.CTkFrame(self.image_frame, fg_color="transparent")
        self.image_frame1.pack(side="left", fill="both", expand=True,padx=10, pady=10)
        
        self.image_frame2 = ctk.CTkFrame(self.image_frame, fg_color="transparent")
        self.image_frame2.pack(side="left", fill="both", expand=True,padx=10, pady=10)

        self.image_label1 = ctk.CTkLabel(self.image_frame1, text="", anchor="center")
        self.image_label1.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        self.image_label2 = ctk.CTkLabel(self.image_frame2, text="", anchor="center")
        self.image_label2.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        self.area_labels = {}
        
        self.initialize_area_labels(self.parking_model2.area_names)
        
        self.app.protocol("WM_DELETE_WINDOW", self.on_closing)

    def initialize_area_labels(self, area_names):
        """Initialize labels for each parking area."""
        row = 0
        col = 0
        last_area = "A"
        
        sorted_area_names = natsorted(area_names)
        
        for area_name in sorted_area_names:
            vpad = (0, 5)        
            label = ctk.CTkLabel(
                self.area_labels_frame, 
                text=f"{area_name}", 
                fg_color="#457b9d", 
                width=40, 
                height=40, 
                corner_radius=8, 
                font=("Poppins", 14, "bold")
            )
            if (area_name[0] != last_area[0]):
                row += 1
                col = 0
            if (row % 2 == 0):
                vpad = (0, 40)
                
            label.grid(row=row, column=col, padx=5, pady=vpad)
            self.area_labels[area_name] = label
            col += 1
            last_area = area_name

    def update_area_status(self, area_name, status):
        """Update the status of a specific parking area."""
        label = self.area_labels[area_name]
        if status == "Filled":
            label.configure(fg_color="#e63946")
        else:
            label.configure(fg_color="#457b9d")

    def update_video_frame(self, frame, results, label, message_label):
        """Display a video frame on the GUI."""
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        
        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(360, 360))
        
        for result in results:
            if result['class_name'] == 'helmeted':
                message_label.configure(text="", fg_color="transparent")
            if result['class_name'] == 'unhelmeted':
                message_label.configure(text="Gunakan Kelengkapan Berkendara!", text_color="#ffffff", font=("Poppins", 14, "bold"), fg_color="#ff0f0f")
        
        label.configure(image=ctk_img)
        label.pack(padx=0, pady=0)
        label.imgtk = ctk_img  

    def update_image_frame(self, image_path, label):
        """Display an image on the GUI."""
        img = Image.open(image_path)
        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(160, 100))
        
        label.configure(image=ctk_img)
        label.pack(side="left", padx=0, pady=0)
        label.imgtk = ctk_img 

    def on_closing(self):
        """Handle cleanup and proper termination of threads when the application is closed."""
        
        if self.helmet_thread is not None:
            self.helmet_thread.join(timeout=2)
            
        if self.camera_thread1 is not None:
            self.camera_thread1.join(timeout=2)
            
        if self.camera_thread2 is not None:
            self.camera_thread2.join(timeout=2)
        
        self.app.destroy()
        
    def run(self):
        """Run the GUI application."""
        def update_parking(parking_status):
            for area_name, status in parking_status.items():
                self.update_area_status(area_name, status)

        def update_helmet(frame, results):
            self.update_video_frame(frame, results, self.video_label_helmet, self.message_label)

        self.camera_thread1 = threading.Thread(target=self.camera1.run, args=(self.parking_model1.run, self.parking_model2.run, update_parking,), name="Camera1")
        self.camera_thread2 = threading.Thread(target=self.camera3.run, args=(self.parking_model4.run, self.parking_model4.run, update_parking,), name="Camera2")
    
        self.helmet_thread = threading.Thread(target=self.helmet_model.run, args=(update_helmet,), name="HelmetDetection")
        # self.parking_model1 = threading.Thread(target=self.parking_model1.run, args=(update_parking,), name="ParkingDetection1")
        
        self.helmet_thread.daemon = True
        self.camera_thread1.daemon = True
        self.camera_thread2.daemon = True
        
        self.helmet_thread.start()
        self.camera_thread1.start()
        self.camera_thread2.start()
        
        self.update_image_frame(self.image_source1, self.image_label1)
        self.update_image_frame(self.image_source2, self.image_label2)

        print("test")
        
        self.app.mainloop()
        
if __name__ == "__main__":
    gui = ParkingDetectionGUI()
    gui.run()
        