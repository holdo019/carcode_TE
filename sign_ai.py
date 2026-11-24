"""
sign_ai.py - best.pt(YOLO) 로 표지판을 검출하는 모듈

main.py 에서:
    from sign_ai import SignDetector

    detector = SignDetector("best.pt 경로")
    boxes = detector.detect(img)   # [(x1, y1, x2, y2, label), ...]

필요: pip install ultralytics
"""
from ultralytics import YOLO


class SignDetector:
    def __init__(self, model_path, conf=0.5, imgsz=640, device=None):
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device   # None 이면 자동 선택 (GPU 있으면 GPU)

    def detect(self, img):
        """BGR 영상 -> [(x1, y1, x2, y2, label), ...] (픽셀 좌표, 정수)"""
        results = self.model.predict(
            img, conf=self.conf, imgsz=self.imgsz,
            device=self.device, verbose=False)[0]

        names = results.names
        out = []
        for b in results.boxes:
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            label = names[int(b.cls[0])]
            out.append((x1, y1, x2, y2, label))
        return out