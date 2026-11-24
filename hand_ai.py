"""
hand_ai.py - 손 찾기 (MediaPipe)

    detector = HandDetector()
    hand = detector.detect(img)      # BGR 영상 -> 손이 있으면 dict, 없으면 None
        hand = {"cx": 손바닥 중심 x(픽셀), "cy": ..., "size": 손바닥 크기(픽셀),
                "bbox": (x1, y1, x2, y2), "points": [(x, y), ...21개]}

size = 손목(0번 점)에서 가운데 손가락 뿌리(9번 점)까지의 픽셀 거리.
손이 카메라에 가까우면 크고, 멀면 작다 -> 거리를 재는 데 쓴다. (손바닥이 카메라를 향할 때 가장 안정적)

설치:  pip install mediapipe
MediaPipe 버전에 따라 두 방식을 모두 지원한다.
  - 옛 방식(mp.solutions)이 있으면 그대로 쓴다 (모델 파일 필요 없음)
  - 없으면 새 방식(Tasks)을 쓰고, 모델 파일 hand_landmarker.task 를 처음 한 번 자동으로 내려받는다
"""
import math
import os
import time
import urllib.request

import cv2

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")


class HandDetector:
    def __init__(self, conf=0.5):
        import mediapipe as mp
        self.mp = mp
        self.legacy = hasattr(mp, "solutions")
        if self.legacy:
            self.hands = mp.solutions.hands.Hands(
                static_image_mode=False, max_num_hands=1,
                min_detection_confidence=conf, min_tracking_confidence=conf)
        else:
            self._make_task_detector(conf)
        self.t0 = time.monotonic()

    def _make_task_detector(self, conf):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        if not os.path.exists(MODEL_PATH):
            print(f"[손] 모델 파일을 내려받는 중 (처음 한 번만): {MODEL_URL}")
            try:
                urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            except Exception as e:
                raise RuntimeError(
                    f"모델 내려받기 실패: {e}\n브라우저로 위 주소를 열어 파일을 받아 "
                    f"{MODEL_PATH} 로 저장하세요.") from e
        # 경로가 아니라 파일 내용을 넘긴다 (MediaPipe 는 한글 등이 든 윈도우 경로의 파일을 못 여는 경우가 있다)
        with open(MODEL_PATH, "rb") as f:
            model_bytes = f.read()
        if len(model_bytes) < 1_000_000:
            raise RuntimeError(f"모델 파일이 이상하다({len(model_bytes)} bytes). {MODEL_PATH} 를 지우고 다시 실행하세요.")
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_buffer=model_bytes),
            running_mode=vision.RunningMode.VIDEO, num_hands=1,
            min_hand_detection_confidence=conf, min_hand_presence_confidence=conf,
            min_tracking_confidence=conf)
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(self, img):
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.legacy:
            res = self.hands.process(rgb)
            if not res.multi_hand_landmarks:
                return None
            lms = res.multi_hand_landmarks[0].landmark
        else:
            mp_img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
            ts = int((time.monotonic() - self.t0) * 1000)
            res = self.landmarker.detect_for_video(mp_img, ts)
            if not res.hand_landmarks:
                return None
            lms = res.hand_landmarks[0]
        return hand_from_points([(p.x * w, p.y * h) for p in lms])


def hand_from_points(points):
    """21개 손 점(픽셀) -> 손 정보. (MediaPipe 와 따로 시험할 수 있게 분리)"""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    palm = [points[i] for i in (0, 5, 9, 13, 17)]           # 손목 + 네 손가락 뿌리 = 손바닥
    return {"cx": sum(p[0] for p in palm) / 5, "cy": sum(p[1] for p in palm) / 5,
            "size": math.dist(points[0], points[9]),
            "bbox": (min(xs), min(ys), max(xs), max(ys)), "points": points}