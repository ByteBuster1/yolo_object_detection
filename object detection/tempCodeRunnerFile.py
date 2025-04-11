import cv2
import numpy as np

# Load YOLO
yolo = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load COCO class labels
classes = []
with open("coco.names", "r") as file:
    classes = [line.strip() for line in file.readlines()]

# Get the output layer names
layer_names = yolo.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo.getUnconnectedOutLayers()]

# Define colors for bounding boxes
colorRed = (0, 0, 255)
colorGreen = (0, 255, 0)

# Load image
name = "image2.jpg"
img = cv2.imread(name)
height, width, channels = img.shape

# Detecting objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
yolo.setInput(blob)
outputs = yolo.forward(output_layers)

class_ids = []
confidences = []
boxes = []

# Process the outputs
for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:  # Confidence threshold
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Calculate the coordinates for the bounding box
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply Non-Maximum Suppression
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Draw bounding boxes and labels on the image
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        
        # Increase font scale and thickness
        font_scale = 5.5  # Increase this value for larger text
        thickness = 3     # Increase this value for bolder text
        
        cv2.rectangle(img, (x, y), (x + w, y + h), colorGreen, 3)
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, font_scale, colorRed, thickness)

# Resize the image for display
desired_width = 800
aspect_ratio = height / width
new_height = int(desired_width * aspect_ratio)
resized_img = cv2.resize(img, (desired_width, new_height))

# Make the window resizable
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)  # Allow resizing

# Display the resized image
cv2.imshow("Image", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the output image
cv2.imwrite("output.jpg", img)
