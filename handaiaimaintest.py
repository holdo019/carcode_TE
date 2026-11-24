"""
handmain.py - 손 따라가기 + uturn 표지판(5초) 제자리 회전 (노트북).

필요 파일(같은 폴더): serv.py  hand_ai.py  hand_follow.py  motor_signal.py  sign_ai.py  best.pt
pip install opencv-python numpy mediapipe ultralytics

키: c 기준 거리 정하기   space GO<->STOP   g GO   s STOP(비상 정지)   q 종료
"""
import time

import cv2

import motor_signal
import serv
from hand_follow import HandFollower, HandParams
from sign_ai import SignDetector

# ---------------- 설정 ----------------
SEND_TO_PI = True
START_STOPPED = False
SEND_INTERVAL = 0.1
FRAME_TIMEOUT = 0.5

# --- 손 따라가기 ---
TARGET_SIZE = 0
TOL = 0.12
KP_DIST = 1.5
KP_TURN = 0.8
TURN_DEAD = 0.08
MAX_SPEED = 30
MIN_SPEED = 15

# --- 표지판 ---
MODEL_PATH = "best.pt"
SIGN_NAME = "uturn"          # uturnx 는 무시
CONF = 0.5
IMGSZ = 640
ROI_W, ROI_H = 0.4, 0.5      # 가운데 박스 (화면 대비 비율)
HOLD_SECONDS = 5.0
LOST_GRACE = 0.3
SPIN_SPEED = 30
SPIN_SECONDS = 1           # 180도 돌도록 직접 맞출 것
SPIN_DIR = 1                 # 반대로 돌면 -1
# --------------------------------------


def roi_rect(w, h):
    rw, rh = int(w * ROI_W), int(h * ROI_H)
    x1, y1 = (w - rw) // 2, (h - rh) // 2
    return x1, y1, x1 + rw, y1 + rh


def sign_in_roi(boxes, roi):
    x1, y1, x2, y2 = roi
    for bx1, by1, bx2, by2, label in boxes:
        if label != SIGN_NAME:
            continue
        cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            return True
    return False


def draw(img, HP, hand, boxes, roi, wheels, go, error, state, info):
    view = img.copy()
    h, w = view.shape[:2]
    cv2.line(view, (w // 2, 0), (w // 2, h), (255, 255, 255), 1)

    # 표지판 박스
    for bx1, by1, bx2, by2, label in boxes:
        cv2.rectangle(view, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
        cv2.putText(view, label, (bx1, max(by1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    rc = (0, 200, 0) if state == "SPIN" else (0, 255, 255) if "holding" in info else (255, 255, 255)
    cv2.rectangle(view, (roi[0], roi[1]), (roi[2], roi[3]), rc, 3)

    lines = [f"{'GO' if go else 'STOP'}  {state}  {info}"]
    if error:
        lines.append(error)
    elif hand is None:
        lines.append("hand: not found (robot stops)")
    else:
        for x, y in hand["points"]:
            cv2.circle(view, (int(x), int(y)), 3, (0, 255, 255), -1)
        x1, y1, x2, y2 = [int(v) for v in hand["bbox"]]
        cv2.rectangle(view, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cx, cy = int(hand["cx"]), int(hand["cy"])
        cv2.drawMarker(view, (cx, cy), (255, 0, 255), cv2.MARKER_CROSS, 24, 3)
        if HP.target > 0:
            cv2.circle(view, (cx, cy), int(HP.target / 2), (0, 200, 0), 2)
            cv2.circle(view, (cx, cy), int(hand["size"] / 2), (0, 255, 255), 1)
        st = ("target not set: press c" if HP.target <= 0 else
              "too close -> back" if hand["size"] > HP.target * (1 + HP.tol) else
              "too far -> forward" if hand["size"] < HP.target * (1 - HP.tol) else "distance OK")
        lines.append(f"size={hand['size']:.0f} target={HP.target}  {st}")
    lines.append(f"left={wheels[0]:.0f} right={wheels[1]:.0f}" + ("" if SEND_TO_PI else "  (NOT sent to Pi)"))

    y = 28
    color = (0, 200, 0) if go else (0, 0, 255)
    for i, t in enumerate(lines):
        cv2.putText(view, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7 if i == 0 else 0.6,
                    color if i == 0 else (255, 255, 255), 2)
        y += 28 if i == 0 else 26
    return view


def main():
    H = HandParams(target=TARGET_SIZE, tol=TOL, kp_dist=KP_DIST, kp_turn=KP_TURN,
                   turn_dead=TURN_DEAD, max_speed=MAX_SPEED, min_speed=MIN_SPEED)
    if H.target <= 0:
        print("[손] 기준 거리가 아직 없다. 원하는 거리에 손을 두고 c 키를 누르세요.")

    error = None
    hand_det = None
    try:
        from hand_ai import HandDetector
        hand_det = HandDetector()
        print("[손] 손 인식 준비 완료")
    except Exception as e:
        error = f"hand AI error: {str(e).splitlines()[0][:60]}"
        print(f"[손] 사용할 수 없음: {e}")

    sign_det = SignDetector(MODEL_PATH, conf=CONF, imgsz=IMGSZ)
    names = sign_det.model.names
    print("모델 클래스:", names)
    if SIGN_NAME not in names.values():
        print(f"경고: 모델에 '{SIGN_NAME}' 클래스가 없다. 위 클래스 이름을 확인하세요")

    follower = HandFollower()
    serv.start()
    go = not START_STOPPED
    wheels = (0.0, 0.0)
    hand = None
    boxes = []
    roi = (0, 0, 0, 0)
    last_fid, last_frame_t, last_send = -1, 0.0, 0.0

    state = "FOLLOW"             # FOLLOW(손 따라가기) -> SPIN(회전) -> CLEAR(표지판 치울 때까지)
    hold_start = None
    last_seen = 0.0
    spin_start = 0.0
    info = ""

    try:
        while True:
            now = time.monotonic()
            img, fid = serv.get_frame()

            if img is not None and fid != last_fid:
                last_fid, last_frame_t = fid, now
                h, w = img.shape[:2]
                roi = roi_rect(w, h)

                hand = hand_det.detect(img) if hand_det is not None else None
                boxes = sign_det.detect(img)
                inside = sign_in_roi(boxes, roi)

                if not go:
                    state, hold_start = "FOLLOW", None
                    follower.reset()
                    wheels = (0.0, 0.0)
                    info = "STOP"
                elif state == "FOLLOW":
                    # 5초 카운트 (카운트 중에도 손 따라가기는 계속)
                    if inside:
                        last_seen = now
                        if hold_start is None:
                            hold_start = now
                        held = now - hold_start
                        info = f"holding {held:.1f}/{HOLD_SECONDS:.0f}s"
                        if held >= HOLD_SECONDS:
                            state, spin_start = "SPIN", now
                            follower.reset()
                            print("[유턴] 5초 충족 -> 제자리 회전")
                    else:
                        if hold_start is not None and now - last_seen > LOST_GRACE:
                            hold_start = None
                        info = "" if hold_start is None else "holding (lost?)"
                    if state == "FOLLOW":
                        wheels = follower.step(hand, w, H) if error is None else (0.0, 0.0)
                elif state == "SPIN":
                    info = f"SPIN {now - spin_start:.1f}/{SPIN_SECONDS:.1f}s"
                elif state == "CLEAR":
                    # 표지판이 박스에서 사라지면 다시 손 따라가기
                    info = "remove sign from box"
                    wheels = (0.0, 0.0)
                    if not inside:
                        state, hold_start = "FOLLOW", None
                        follower.reset()

                cv2.imshow("CAMERA", draw(img, H, hand, boxes, roi, wheels, go, error, state, info))
                send_now = True
            else:
                send_now = False
                if now - last_frame_t > FRAME_TIMEOUT:
                    wheels = (0.0, 0.0)
                    hold_start = None

            # 회전은 시간으로 제어
            if state == "SPIN":
                if go and now - spin_start < SPIN_SECONDS and now - last_frame_t <= FRAME_TIMEOUT:
                    wheels = (SPIN_DIR * SPIN_SPEED, -SPIN_DIR * SPIN_SPEED)
                else:
                    wheels = (0.0, 0.0)
                    state = "CLEAR"
                    print("[유턴] 회전 종료")

            if SEND_TO_PI and (send_now or now - last_send >= SEND_INTERVAL):
                if go and wheels != (0.0, 0.0):
                    serv.send_command(motor_signal.to_signal(*wheels))
                else:
                    serv.send_command("STOP,-,-")
                last_send = now

            key = cv2.waitKey(20) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                if hand is not None:
                    H.target = int(hand["size"])
                    print(f"[손] 기준 크기 {H.target}px.  유지하려면 TARGET_SIZE = {H.target}")
                else:
                    print("[손] 손이 보일 때 눌러야 한다")
            elif key == ord(" "):
                go = not go
                print("GO" if go else "STOP")
            elif key == ord("g"):
                go = True
                print("GO")
            elif key == ord("s"):
                go = False
                print("STOP")
    except KeyboardInterrupt:
        pass
    finally:
        if SEND_TO_PI:
            serv.send_command("STOP,-,-")
            time.sleep(0.2)
        serv.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()