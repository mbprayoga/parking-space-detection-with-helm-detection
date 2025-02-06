from park import ParkingDetectionModel
from gui import ParkingDetectionGUI
from helm import HelmetDetection

def main():
    model_interface = ParkingDetectionModel()

    helmet_model = HelmetDetection('models/yolov11/small/best.onnx', 0.5, 0.5)  

    gui = ParkingDetectionGUI(model_interface, helmet_model)
    
    gui.initialize_area_labels(model_interface.area_names)

    gui.run()

if __name__ == "__main__":
    main()
