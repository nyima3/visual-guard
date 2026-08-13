"""Real-time webcam detection using threading and queue to avoid blocking."""
import cv2
import threading
import queue
from ultralytics import YOLO
from .utils import draw_boxes


def webcam_demo(weights='models/best.pt', device=None, conf=0.25):
    model = YOLO(weights)
    if device:
        model.to(device)
    cap = cv2.VideoCapture(0)
    q = queue.Queue(maxsize=2)

    def reader():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if not q.full():
                q.put(frame)
    th = threading.Thread(target=reader, daemon=True)
    th.start()

    while True:
        if q.empty():
            continue
        frame = q.get()
        res = model(frame, conf=conf)[0]
        boxes=[]; labels=[]; scores=[]
        for box in res.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
            boxes.append((x1,y1,x2,y2))
            scores.append(float(box.conf[0]))
            labels.append(model.names[int(box.cls[0])])
        out = draw_boxes(frame, boxes, labels, scores)
        cv2.imshow('webcam', out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release(); cv2.destroyAllWindows()


if __name__ == '__main__':
    webcam_demo()
