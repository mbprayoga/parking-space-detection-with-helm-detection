import customtkinter as ctk
from PIL import Image
import cv2

class ParkingDetectionGUI:
    def __init__(self, model_interface, helmet_model):
        self.model_interface = model_interface
        self.helmet_model = helmet_model  # Pass the initialized helmet model to the GUI

        # Initialize CustomTkinter GUI
        self.app = ctk.CTk()
        self.app.title("Parking and Helmet Detection System")
        self.app.state('zoomed')

        # Frames
        self.left_frame = ctk.CTkFrame(self.app, width=800, corner_radius=10)
        self.left_frame.pack(side="left", fill="both", expand=True)

        self.right_frame = ctk.CTkFrame(self.app, width=400, corner_radius=10)
        self.right_frame.pack(side="right", fill="both", expand=True)

        # Video display labels for parking and helmet detection
        self.video_label_parking = ctk.CTkLabel(self.left_frame, text="", anchor="center")
        self.video_label_parking.pack(fill="both", expand=True)

        self.video_label_helmet = ctk.CTkLabel(self.left_frame, text="", anchor="center")
        self.video_label_helmet.pack(fill="both", expand=True)

        # Parking area labels
        self.area_labels = {}

    def initialize_area_labels(self, area_names):
        """Initialize labels for each parking area."""
        for area_name in area_names:
            label = ctk.CTkLabel(
                self.right_frame, 
                text=f"{area_name}: Empty", 
                fg_color="green", 
                width=200, 
                height=30, 
                corner_radius=8
            )
            label.pack(pady=5, padx=10)
            self.area_labels[area_name] = label

    def update_area_status(self, area_name, status):
        """Update the status of a specific parking area."""
        label = self.area_labels[area_name]
        if status == "Filled":
            label.configure(text=f"{area_name}: Filled", fg_color="red")
        else:
            label.configure(text=f"{area_name}: Empty", fg_color="green")

    def update_video_frame(self, frame, label):
        """Display a video frame on the GUI."""
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        
        # Convert to CTkImage (instead of ImageTk.PhotoImage)
        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(640, 480))
        
        # Update the label with the new CTkImage object
        label.configure(image=ctk_img)
        label.imgtk = ctk_img  # Store a reference to avoid garbage collection

    def run(self):
        """Run the GUI application."""
        def video_loop():
            while True:
                # Get the parking detection frame and status
                frame_parking, parking_status = self.model_interface.run()
                if frame_parking is None:
                    break

                self.update_video_frame(frame_parking, self.video_label_parking)

                # Update parking area statuses
                for area_name, status in parking_status.items():
                    self.update_area_status(area_name, status)

                # Get the helmet detection frame from the passed helmet model
                frame_helmet = self.helmet_model.run()
                if frame_helmet is None:
                    break

                self.update_video_frame(frame_helmet, self.video_label_helmet)

                self.app.update_idletasks()
                self.app.update()

        video_loop()
