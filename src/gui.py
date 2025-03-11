import customtkinter as ctk
from PIL import Image
import threading
from natsort import natsorted
import time
import os

from helm import HelmetDetection
from park import ParkingDetectionModel
from camera import Camera

class ParkingDetectionGUI:
    def __init__(self):
        self.parking_model1 = ParkingDetectionModel('park-image/image-1.png', 'segment/segment-1')
        self.parking_model2 = ParkingDetectionModel('park-image/image-2.png', 'segment/segment-2')
        self.parking_model3 = ParkingDetectionModel('park-image/image-3.png', 'segment/segment-3')
        self.parking_model4 = ParkingDetectionModel('park-image/image-4.png', 'segment/segment-4')
        self.parking_model5 = ParkingDetectionModel('park-image/image-5.png', 'segment/segment-5')
        self.parking_model6 = ParkingDetectionModel('park-image/image-6.png', 'segment/segment-6')
        
        self.helmet_model = HelmetDetection(self.update_video_frame)

        self.camera1 = Camera("NW", "192.168.183.200", 8899, "", "", 19, 26) 
        self.camera2 = Camera("SE", "192.168.183.241", 8899, "", "", 19, 24)
        
        ctk.set_appearance_mode("light")

        self.app = ctk.CTk()
        self.app.title("Parking and Helmet Detection System")
        
        self.app.geometry("1024x600")

        self.left_frame = ctk.CTkFrame(self.app, width=360, fg_color="transparent")
        self.left_frame.pack(side="left", padx=0, pady=0)

        self.right_frame = ctk.CTkFrame(self.app, width=400, fg_color="transparent")
        self.right_frame.pack(side="right", fill="both", expand=True, padx=0, pady=0)
        
        self.area_labels_frame = ctk.CTkFrame(self.right_frame, width=400, fg_color="transparent")
        self.area_labels_frame.pack(side="right", expand=True, anchor="center", padx=0, pady=0)

        self.video_frame = ctk.CTkFrame(self.left_frame, fg_color="transparent")
        self.video_frame.pack(padx=10, pady=10)
        
        self.video_label_helmet = ctk.CTkLabel(self.video_frame, text="", anchor="center", fg_color="transparent")
        self.video_label_helmet.pack(fill="both", expand=True, padx=5, pady=5)

        self.image_frame = ctk.CTkFrame(self.left_frame, fg_color="transparent")
        self.image_frame.pack(fill="both", expand=True, padx=0, pady=0)
        
        self.message_label = ctk.CTkLabel(self.image_frame, text="", anchor="center", fg_color="transparent", width=340, height=50, corner_radius=8)
        self.message_label.pack(padx=10, pady=10)  
        self.message_label.pack_forget()

        self.warning_active = False
        self.warning_duration = 2
        self.last_warning_time = time.time()

        self.area_labels = {}

        self.initialize_area_labels(ParkingDetectionModel.areas.keys())
        
        self.app.protocol("WM_DELETE_WINDOW", self.on_closing)

    def initialize_area_labels(self, area_names):
        """Initialize labels for each parking area."""
        row = 0
        col = 0
        last_area = "A"
        
        sorted_area_names = natsorted(area_names)
        
        for area_name in sorted_area_names:
            vpad = (1, 5)        
            label = ctk.CTkLabel(
                self.area_labels_frame, 
                text=f"{area_name}", 
                fg_color="#457b9d",
                bg_color="darkgray", 
                width=30, 
                height=30, 
                font=("Poppins", 9, "bold"),
                padx=0,
                pady=0,
            )
            if (area_name[0] != last_area[0]):
                row += 1
                col = 0
            if (row % 2 != 0 and row > 0):
                vpad = (1, 50)
                
            label.grid(row=row, column=col, padx=0, pady=vpad)
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

    def update_video_frame(self, results, frame):
        """Display a video frame on the GUI."""
        img = Image.fromarray(frame)
        
        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(350, 350))
        
        current_time = time.time()
    
        if (results['tanpa-helm'] == True and  results['helm'] == False) or (results['tanpa-helm'] == True and  results['helm'] == True):
            if not self.warning_active:
                self.message_label.configure(text="Lengkapi Helm!", text_color="#ffffff", font=("Poppins", 24, "bold"), fg_color="#e63946") 
                self.message_label.pack(padx=10, pady=10)
                self.last_warning_time = current_time
                self.warning_active = True
        if results['helm'] == True and results['tanpa-helm'] == False:
            if not self.warning_active:
                self.message_label.configure(text="Silahkan Masuk!", text_color="#ffffff", font=("Poppins", 24, "bold"), fg_color="#457b9d") 
                self.message_label.pack(padx=10, pady=10)
                self.last_warning_time = current_time
                self.warning_active = True
        elif current_time - self.last_warning_time >= self.warning_duration:
            if self.warning_active:
                self.message_label.pack_forget()
                self.warning_active = False
        
        
        self.video_label_helmet.configure(image=ctk_img)
        self.video_label_helmet.pack(padx=0, pady=0)
        self.video_label_helmet.imgtk = ctk_img  


    def on_closing(self):
        """Handle cleanup and proper termination of threads when the application is closed."""

        if self.helmet_thread is not None:
            self.helmet_thread.join(timeout=2)

        self.app.destroy()
        self.app.quit()

        os._exit(0)
        
    def run(self):
        """Run the GUI application."""
        def update_parking(parking_status):
            for area_name, status in ParkingDetectionModel.areas.items():
                self.update_area_status(area_name, status)

        # self.parking_model1.run(update_parking)
        # self.parking_model2.run(update_parking)
        # self.parking_model3.run(update_parking)
        # self.parking_model4.run(update_parking)
        # self.parking_model5.run(update_parking)
        # self.parking_model6.run(update_parking)
            
        self.camera_thread1 = threading.Thread(
            target=self.camera1.startup, 
            args=(self.parking_model1.run, self.parking_model2.run, self.parking_model3.run, update_parking,), 
            daemon=True,
            name="Camera1") 
        
        self.camera_thread2 = threading.Thread(
            target=self.camera2.startup, 
            args=(self.parking_model4.run, self.parking_model5.run, self.parking_model6.run, update_parking,), 
            daemon=True,
            name="Camera2") 

        self.helmet_thread = threading.Thread(
            target=self.helmet_model.app.run, 
            daemon=True,
            name="HelmetDetection")
        
    
        self.camera_thread1.start() 
        self.camera_thread2.start() 
        self.helmet_thread.start()

        self.app.mainloop()
        
if __name__ == "__main__":
    gui = ParkingDetectionGUI()
    gui.run()
        