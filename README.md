# Vehicle Detection and Lane Estimation YOLOv8

This Python script performs vehicle detection in a video stream and estimates the number of lanes on a road. It uses the Ultralytics YOLO model for object detection and the SORT algorithm for object tracking.

Demo: https://www.loom.com/share/fa555ad3f0d34bd99861a0755f192457

## Features

- Vehicle Detection: Identifies vehicles in each video frame using the YOLOv8 model.
- Object Tracking: Employs the SORT algorithm to track the movement of each detected vehicle.
- Lane Estimation: Uses KMeans clustering on vehicle positions to estimate the number of lanes on the road.
- Visualization: Draws bounding boxes around vehicles, displays class labels with confidence scores, and illustrates estimated lanes and vehicle trajectories.

## Requirements

- Python 3
- OpenCV library (`cv2`)
- Ultralytics YOLO (`ultralytics`)
- SORT algorithm (`sort`)
- Scikit-learn (`sklearn`)

Make sure to install the necessary packages using `pip` before running the script.

pip install opencv-python-headless scikit-learn sort

Note: You may need to adjust the installation command based on your environment.

YOLO Model: The pre-trained YOLOv8 model can be downloaded from Ultralytics or trained on custom data. Place the weights file (e.g., yolov8n.pt) in the project directory.

Video Source: The traffic video (e.g., cars.mp4) should be located in the ../Videos/ directory relative to the script.

Mask Image: An optional mask image (mask.png) can be used to focus detection on a specific region within the video frames. This is useful for ignoring areas that are not relevant to the analysis.

## Usage

Place the script in a directory with the following files:

- `../Videos/cars.mp4`: The video file of traffic where vehicles are to be detected.
- `../Yolo-Weights/yolov8n.pt`: The file containing the pre-trained YOLO weights.
- `mask.png`: An image file used to mask the region of interest in the video frames.

Run the script using the command line with python vehicle_lane_detection.py. Ensure that the current working directory is set to where the script is located. Press 'q' while the output window is active to exit the program.


Press 'q' to quit the video window.

## Detailed Breakdown

**Initialization**: The script starts by setting up the video capture, loading the YOLO model, defining class names, and initializing the SORT tracker.

**Main Loop**: For each frame in the video:
- The region of interest is applied if a mask is provided.
- The YOLO model identifies vehicles in the frame, and detected vehicles are passed to the SORT tracker.
- Vehicle positions are collected as they cross a pre-defined counting line.
- The script waits until it has accumulated a certain number of positions to ensure accurate trajectory estimation.
- The vehicle's width is also stored to calculate the lane width.

**Lane Estimation**:
- Collected vehicle positions are used to estimate lane boundaries.
- If a sufficient number of positions are available, the KMeans algorithm clusters these positions, with the   number of clusters representing the number of lanes.
- The average width of the detected vehicles helps to fine-tune the number of lanes estimated.
- For visual feedback, the script draws lines representing estimated lane centers on the video frames.

**Output**:
- The script displays the processed video frames in real-time, with bounding boxes around detected vehicles, trajectory lines, and the estimated number of lanes.
- A count of detected vehicles is also displayed.



## Notes
- The accuracy of vehicle detection and lane estimation can vary based on video quality, camera angle, and environmental conditions.
- It's essential to have a proper mask image and adjust the limits for the counting line based on the specific traffic scene being analyzed.
- The script's parameters, such as the YOLO model weights and mask image, may need adjustment to match the specifics of the deployment environment.

##FinalProjectUnet.ipynb Requirements
- gtFine_trainvaltest.zip dataset
- leftImg8bit_trainvaltest.zip dataset
- NVDIA GPU with CUDA toolkit
