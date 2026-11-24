"""
hand_test.py - MediaPipe 손 인식만 따로 시험 (라파/모터 없이, 노트북 웹캠으로)

준비:  python -m venv venv  ->  venv 활성화  ->  pip install opencv-python numpy mediapipe
같은 폴더에 hand_ai.py 가 있어야 한다.   실행:  python hand_test.py     (종료: q)

1단계 mediapipe 불러오기 / 2단계 모델 준비(처음엔 내려받음) / 3단계 웹캠으로 손 인식
어느 단계에서 멈추는지 보면 된다.
"""
import sys

print("python", sys.version.split()[0])
try:
    import cv2
    import numpy
    print("1) opencv", cv2.__version__, " numpy", numpy.__version__)
    import mediapipe as mp
    print("1) mediapipe", getattr(mp, "__version__", "?"), " (legacy solutions:", hasattr(mp, "solutions"), ")")
except Exception as e:
    print("1) 불러오기 실패:", e)
    sys.exit(1)

try:
    from hand_ai import HandDetector
    det = HandDetector()
    print("2) 손 인식 준비 완료")
except Exception as e:
    print("2) 준비 실패:", e)
    sys.exit(1)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("3) 웹캠을 못 열었다. VideoCapture(0) 을 1 로 바꿔 보세요")
    sys.exit(1)
print("3) 웹캠 시작. 카메라에 손바닥을 보이세요. q 로 종료")
while True:
    ok, img = cap.read()
    if not ok:
        break
    hand = det.detect(img)
    if hand is None:
        txt = "hand: not found"
    else:
        for x, y in hand["points"]:
            cv2.circle(img, (int(x), int(y)), 4, (0, 255, 255), -1)
        x1, y1, x2, y2 = [int(v) for v in hand["bbox"]]
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
        txt = f"size={hand['size']:.0f}px  cx={hand['cx']:.0f}"
    cv2.putText(img, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("hand test", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
