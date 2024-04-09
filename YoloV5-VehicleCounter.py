!git clone https://github.com/ultralytics/yolov5.git
%cd yolov5
!pip install -r requirements.txt

!python detect.py --weights yolov5s.pt --img 640 --conf 0.25 --source 0

from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression
from torchvision.transforms.functional import resize
from PIL import Image, ImageDraw
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

# Load YOLOv5 model
model = attempt_load('yolov5s.pt')

image_path = 'content/example_road.jpg'
image = Image.open(image_path).convert('RGB')

resized_image = resize(image, (416, 416))
image_np = np.array(resized_image)

image_tensor = torch.from_numpy(image_np)
image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
image_tensor = image_tensor.float() / 255.0

results = model(image_tensor)
results = non_max_suppression(results)
print(results)

print("Image shape:", image_np.shape)
print("Number of results tensors:", len(results))

num_cars = 0
# Iterate through the results and draw bounding boxes
for box in results[0]:\
    num_cars+=1
    x1, y1, x2, y2 = box[0:4].cpu().numpy().astype(int) 
    confidence = box[4].cpu().numpy()
    class_label = int(box[5].cpu().numpy())  

    print("Bounding box coordinates:", x1, y1, x2, y2)
    cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Save image with bounded boxes
save_img = Image.fromarray(image_np)
save_img.save("content/result_img.jpg")

plt.figure(figsize=(10, 10))
plt.imshow(image_np)
plt.axis('off')
plt.show()

# Determine busyness
print(f"There are {num_cars} vehicles on this street!")
