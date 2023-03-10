from ultralytics import YOLO
import cv2
import cvzone
import math

cap = cv2.VideoCapture("../Videos/motorbikes.mp4")
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
while True:
    success, img = cap.read()
    img_crop = cv2.bitwise_and(img, mask)
    results = model(img_crop, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # print(x1, y1, x2, y2)
            bbox = x1, y1, x2 - x1, y2 - y1
            cvzone.cornerRect(img, bbox)
            conf = (math.ceil(box.conf[0] * 100)) / 100
            cls = box.cls[0]
            cvzone.putTextRect(img, f"{classNames[int(cls)]} {conf}", (max(0, x1), max(35, y1 - 20)))
            # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 1), 1)
    cv2.imshow("Webcam", img)
    cv2.waitKey(1)
