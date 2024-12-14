from park import ParkingDetectionModel
from gui import ParkingDetectionGUI
from helm import HelmetDetection

def main():
    # Initialize the parking detection model
    model_interface = ParkingDetectionModel()

    # Initialize the helmet detection model
    helmet_model = HelmetDetection('best.onnx', 0.5, 0.5)  # Load helmet detection model with threshold values

    # Initialize the GUI with the model interface and helmet model
    gui = ParkingDetectionGUI(model_interface, helmet_model)
    
    # Initialize parking area labels in the GUI
    gui.initialize_area_labels(model_interface.area_names)

    # Run the GUI application
    gui.run()

if __name__ == "__main__":
    main()
