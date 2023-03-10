from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap = cv2.VideoCapture("../Videos/cars.mp4")
cap.set(3, 640)
cap.set(4, 480)

model = YOLO("../Yolo-weights/yolov8l.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
mask = cv2.imread("mask.png")

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)  # takes in a 1x5 matrix as input
limits = [400, 297, 673, 297]
total_counts = []  # Here this list stores whatever car is counted
while True:
    success, img = cap.read()
    img_region = cv2.bitwise_and(img, mask)
    img_graphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, img_graphics, (0, 0))
    results = model(img_region, stream=True)
    detections = np.empty((0, 5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            conf = (math.ceil(box.conf[0] * 100)) / 100
            if currentClass == "car" and conf > 0.3:
                '''cvzone.putTextRect(img, f"{currentClass} {conf}", (max(0, x1), max(35, y1 - 20)), scale=0.6,
                                   thickness=1, offset=3)'''
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))
    results_Tracker = tracker.update(detections)
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
    for results in results_Tracker:
        x1, y1, x2, y2, Id = results
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        bbox = x1, y1, x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        cvzone.cornerRect(img, bbox, l=9)
        cvzone.putTextRect(img, f"{Id}", (max(0, x1), max(35, y1 - 20)), scale=2,
                           thickness=1, offset=3)
        if limits[1] + 20 > cy > limits[1] - 20 and 400 < cx < 673:
            if total_counts.count(Id) == 0:
                total_counts.append(Id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    # cvzone.putTextRect(img, f"Counts:{len(total_counts)}", (50, 50))
    cv2.putText(img, str(len(total_counts)), (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)
    cv2.waitKey(1)
    cv2.imshow("People Counter", img)
