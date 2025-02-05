import customtkinter as ctk
from PIL import Image
import cv2
import threading

class ParkingDetectionGUI:
    def __init__(self, model_interface, helmet_model):
        self.model_interface = model_interface
        self.helmet_model = helmet_model  # Pass the initialized helmet model to the GUI

        self.image_source1 = "test-image1.png"
        self.image_source2 = "test-image2.png"
        
        # Set the appearance mode to light
        ctk.set_appearance_mode("light")

        # Initialize CustomTkinter GUI
        self.app = ctk.CTk()
        self.app.title("Parking and Helmet Detection System")
        self.app.state('zoomed')
        

        # Set the geometry to cover the whole screen
        self.app.geometry("1024x600")

        # Frames
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

        # Frame for additional image previews
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

        # Parking area labels
        self.area_labels = {}
        
        self.parking_thread = None
        self.helmet_thread = None

        # Bind the close event to the cleanup method
        self.app.protocol("WM_DELETE_WINDOW", self.on_closing)

    def initialize_area_labels(self, area_names):
        """Initialize labels for each parking area."""
        row = 0
        col = 0
        last_area = "A"
        for area_name in area_names:
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
        
        # Convert to CTkImage (instead of ImageTk.PhotoImage)
        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(360, 240))
        
        for result in results:
            if result['class_name'] == 'helmeted':
                message_label.configure(text="", fg_color="transparent")
            if result['class_name'] == 'unhelmeted':
                message_label.configure(text="Gunakan Kelengkapan Berkendara!", text_color="#ffffff", font=("Poppins", 14, "bold"), fg_color="#ff0f0f")
        
        # Update the label with the new CTkImage object
        label.configure(image=ctk_img)
        label.pack(padx=0, pady=0)
        label.imgtk = ctk_img  # Store a reference to avoid garbage collection

    def update_image_frame(self, image_path, label):
        """Display an image on the GUI."""
        img = Image.open(image_path)
        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(160, 100))
        
        # Update the label with the new CTkImage object
        label.configure(image=ctk_img)
        label.pack(side="left", padx=0, pady=0)
        label.imgtk = ctk_img  # Store a reference to avoid garbage collection
    
    def run(self):
        """Run the GUI application."""
        def update_parking(frame, parking_status):
            for area_name, status in parking_status.items():
                self.update_area_status(area_name, status)

        def update_helmet(frame, results):
            self.update_video_frame(frame, results, self.video_label_helmet, self.message_label)

        # Display images in the labels
        self.update_image_frame(self.image_source1, self.image_label1)
        self.update_image_frame(self.image_source2, self.image_label2)
        
        # Start the parking detection and helmet detection threads
        self.parking_thread = threading.Thread(target=self.model_interface.run, args=(update_parking,), name="ParkingDetectionModel")
        self.helmet_thread = threading.Thread(target=self.helmet_model.run, args=(update_helmet,), name="HelmetDetection")
        
        self.helmet_thread.setDaemon(True)

        self.parking_thread.start()
        self.helmet_thread.start()

        self.app.mainloop()

    def on_closing(self):
        """Handle cleanup and proper termination of threads when the application is closed."""
        
        if self.helmet_thread is not None:
            self.helmet_thread.join(timeout=2)
            
        if self.parking_thread is not None:
            self.parking_thread.join(timeout=2)
        
        self.app.destroy()
        