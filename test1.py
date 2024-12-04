import cv2
import numpy as np
import cvzone
import pickle

cap = cv2.VideoCapture('easy1.mp4')

drawing = False
area_names = []

# Load existing parking segments if available
try:
    with open("parkingsegment", "rb") as f:
        data = pickle.load(f)
        polylines, area_names = data['polylines'], data['area_names']
except:
    polylines = []

points = []  # Stores points of the currently drawn polygon

def draw(event, x, y, flags, param):
    """
    Mouse callback function for drawing polygons.
    """
    global points, drawing

    if event == cv2.EVENT_LBUTTONDOWN:  # Start or add a point
        points.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:  # Undo the last point
        if points:
            points.pop()


while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    frame = cv2.resize(frame, (1020, 500))
    temp_frame = frame.copy()

    # Draw existing polygons
    for i, polyline in enumerate(polylines):
        cv2.polylines(temp_frame, [polyline], True, (0, 0, 255), 2)
        cvzone.putTextRect(temp_frame, f'{area_names[i]}', tuple(polyline[0]), 1, 1)

    # Draw the currently drawn polygon (incomplete)
    if len(points) > 1:
        cv2.polylines(temp_frame, [np.array(points, np.int32)], False, (0, 255, 0), 2)
    elif len(points) == 1:
        cv2.circle(temp_frame, points[0], 5, (0, 255, 0), -1)

    cv2.imshow('FRAME', temp_frame)
    cv2.setMouseCallback('FRAME', draw)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('e') and len(points) > 2:  # End and close the polygon
        points.append(points[0])  # Connect the last point to the first
        current_name = input('Area name: ')  # Input area name
        if current_name:
            area_names.append(current_name)
            polylines.append(np.array(points, np.int32))
        points = []  # Reset for the next polygon

    if key == ord('s'):  # Save polygons to file
        with open("parkingsegment", "wb") as f:
            data = {'polylines': polylines, 'area_names': area_names}
            pickle.dump(data, f)
        print("Segments saved!")

    if key == ord('q'):  # Quit
        break

cap.release()
cv2.destroyAllWindows()
