from ultralytics import YOLO
import cv2

model = YOLO("Orchid_label.v5i.yolov8/runs/detect/train/weights/best.pt")
img_path = "/Users/kuangsin/FedAvg/image_processing/client1/Orchid_label.v5i.yolov8/valid/images/58_JPG.rf.5c0cafc88c295f86f1f987dd0ae8b546.jpg"
img = cv2.imread(img_path)

results = model(img)

# Iterate over the detected orchid leaves
for box in results[0].boxes.data:
    # Get bounding box coordinates and calculate dimensions
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    
    width = x2 - x1
    height = y2 - y1
    
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(img, f"W: {width}px H: {height}px", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


cv2.imwrite("output_with_dimensions.jpg", img)
