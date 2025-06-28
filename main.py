from ultralytics import YOLO
import cv2

model = YOLO("models/best_v1.pt")  

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True)

    for r in results:
        annotated_frame = r.plot()

    cv2.imshow("Casco Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
