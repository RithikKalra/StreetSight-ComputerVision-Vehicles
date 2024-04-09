import numpy as np
from ultralytics import YOLO
import cv2
import math
from sort import *
from sklearn.cluster import KMeans

cap = cv2.VideoCapture("../Videos/cars.mp4")

if not cap.isOpened():
    print("Error opening video file")
    exit()

model = YOLO("../Yolo-Weights/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

mask = cv2.imread("mask.png")
if mask is None:
    print("Error loading mask image")
    exit()

print("Video and mask loaded successfully")

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limits = [400, 297, 673, 297]
totalCount = []  # List to store counted vehicles
car_positions = []  # List to store the positions of cars crossing the line
vehicle_widths = []  # List to store the widths of detected vehicles
previous_centers = {}  # Dictionary to store the previous centers of tracked vehicles

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read frame from video")
        break

    # Resize mask to match the dimensions of the img
    mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
    imgRegion = cv2.bitwise_and(img, mask_resized)

    # Run YOLO model on the image
    results = model(imgRegion, stream=True)
    detections = np.empty((0, 5))

    # Process detections
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # Draw bounding boxes for detected vehicles
            if currentClass in ["car", "truck", "bus", "motorbike"] and conf > 0.3:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f'{currentClass} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))
                vehicle_widths.append(w)  

    # Update the tracker
    resultsTracker = tracker.update(detections)

    # Count vehicles and estimate lanes
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cx, cy = x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2

        # Draw the estimated trajectory of the detected vehicle
        if id in previous_centers:
            prev_cx, prev_cy = previous_centers[id]
            cv2.line(img, (prev_cx, prev_cy), (cx, cy), (0, 255, 255), 2)
        previous_centers[id] = (cx, cy)  # Update the previous center for the vehicle

        # Check if the vehicle crosses the line
        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            if totalCount.count(id) == 0:
                totalCount.append(id)
                car_positions.append([cx])

    # Draw the counting line
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    # Estimate and cluster car positions to estimate the number of lanes
    if len(car_positions) > 0:
        min_position = min(car_positions)[0]
        max_position = max(car_positions)[0]
        position_range = max_position - min_position

        # Use the average width of detected vehicles as the estimated lane width
        if vehicle_widths:
            estimated_lane_width = sum(vehicle_widths) / len(vehicle_widths)
        else:
            estimated_lane_width = 50 

        # Estimate the number of lanes
        estimated_num_lanes = max(1, position_range // estimated_lane_width)

        # Ensure the number of clusters is not greater than the number of samples
        num_clusters = min(len(car_positions), int(estimated_num_lanes))

        # Use KMeans to cluster car positions with the estimated number of lanes
        if num_clusters > 1: 
            kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(car_positions)
            num_lanes = len(set(kmeans.labels_))

            # Draw lines for the estimated lanes
            for i in range(num_lanes):
                lane_center = int(kmeans.cluster_centers_[i][0])
                cv2.line(img, (lane_center, 0), (lane_center, img.shape[0]), (255, 0, 0), 2)
        else:
            num_lanes = num_clusters 

        cv2.putText(img, f'Lanes: {num_lanes}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.putText(img, f'Vehicles: {len(totalCount)}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
