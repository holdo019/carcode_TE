"""
handmain.py - 손 따라가기 전용 메인 (노트북).  손을 따라가며 손과의 거리를 일정하게 유지한다.

실행:  python handmain.py        (라파에서 caramain.py 를 실행해 두면 영상이 온다)
같은 폴더에 필요한 파일:  serv.py  hand_ai.py  hand_follow.py  motor_signal.py     그리고  pip install opencv-python numpy mediapipe
(라인 관련 파일 linetrack.py / engine.py / settings_io.py 는 필요 없다)

사용 순서
  1. 원하는 거리에 손바닥을 카메라 쪽으로 향하게 두고 c 키 -> 그 손 크기가 '기준 거리'가 된다
  2. 손이 멀어지면 앞으로, 가까워지면 뒤로, 좌우로 움직이면 방향을 튼다. 손이 안 보이면 멈춘다
  3. 같은 거리를 계속 쓰려면, 콘솔에 나온 기준 크기를 아래 TARGET_SIZE 에 적어 둔다

설정 파일은 없다. 따라가는 값은 아래 숫자만 고치면 된다.

키 (영상 창을 클릭한 상태에서)
  c      지금 손 크기를 기준 거리로 정한다 (이번 실행 동안만)
  space  GO <-> STOP      g  GO      s  STOP (비상 정지)
  q      종료

라파로 보내는 신호:  "M,AIN1,AIN2,PWMA,BIN1,BIN2,PWMB"  또는  "STOP,-,-"   (만드는 곳: motor_signal.py)
신호가 0.5초 넘게 끊기면 라파가 스스로 멈춘다.
"""
import time

import cv2

import motor_signal
import serv
from hand_follow import HandFollower, HandParams

# ---------------- 설정 ----------------
SEND_TO_PI = True            # False: 라파로 바퀴 신호를 보내지 않는다 / True: 신호 전송 (바퀴가 움직인다)
START_STOPPED = False        # False: 시작부터 GO / True: 시작은 STOP (g 또는 space 로 출발)
SEND_INTERVAL = 0.1          # 라파로 신호를 다시 보내는 최소 간격(초)
FRAME_TIMEOUT = 0.5          # 영상이 이 시간 넘게 안 오면 정지 신호

TARGET_SIZE = 0              # 기준 손 크기(픽셀) = 유지할 거리. 0 이면 c 키로 정한다 (정하기 전에는 안 움직임)
TOL = 0.12                   # 기준과 이 비율 이내로 비슷하면 '적당한 거리' -> 앞뒤로 안 움직임
KP_DIST = 1.5                # 거리 오차 -> 앞뒤 속도 세기 (올리면 빨리 반응)
KP_TURN = 0.8                # 좌우 오차 -> 방향 틀기 세기
TURN_DEAD = 0.08             # 화면 가운데에서 이 비율 이내면 방향을 틀지 않음
MAX_SPEED = 30               # 최대 속도 (PWM 0~100)
MIN_SPEED = 15               # 움직일 때의 최소 속도 (너무 약하면 모터가 안 돈다)
# --------------------------------------


def draw(img, HP, hand, wheels, go, error):
    view = img.copy()
    h, w = view.shape[:2]
    cv2.line(view, (w // 2, 0), (w // 2, h), (255, 255, 255), 1)               # 화면 중앙
    lines = [f"HAND  {'GO' if go else 'STOP'}"]
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
            cv2.circle(view, (cx, cy), int(HP.target / 2), (0, 200, 0), 2)      # 기준 크기 (초록 원)
            cv2.circle(view, (cx, cy), int(hand["size"] / 2), (0, 255, 255), 1) # 지금 크기 (노랑 원)
        state = ("target not set: press c" if HP.target <= 0 else
                 "too close -> back" if hand["size"] > HP.target * (1 + HP.tol) else
                 "too far -> forward" if hand["size"] < HP.target * (1 - HP.tol) else "distance OK")
        lines.append(f"size={hand['size']:.0f} target={HP.target}  {state}")
    lines.append(f"left={wheels[0]:.0f} right={wheels[1]:.0f}" + ("" if SEND_TO_PI else "  (NOT sent to Pi)"))
    y = 28
    color = (0, 200, 0) if go else (0, 0, 255)
    for i, t in enumerate(lines):
        cv2.putText(view, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8 if i == 0 else 0.6,
                    color if i == 0 else (255, 255, 255), 2)
        y += 28 if i == 0 else 26
    return view


def main():
    H = HandParams(target=TARGET_SIZE, tol=TOL, kp_dist=KP_DIST, kp_turn=KP_TURN,
                   turn_dead=TURN_DEAD, max_speed=MAX_SPEED, min_speed=MIN_SPEED)
    if H.target <= 0:
        print("[손] 기준 거리가 아직 없다. 원하는 거리에 손을 두고 c 키를 누르세요.")

    error = None
    detector = None
    try:
        from hand_ai import HandDetector
        detector = HandDetector()
        print("[손] 손 인식 준비 완료")
    except Exception as e:                          # mediapipe 없음, 모델 내려받기 실패 등
        error = f"hand AI error: {str(e).splitlines()[0][:60]}"
        print(f"[손] 사용할 수 없음: {e}")

    follower = HandFollower()
    serv.start()
    go = not START_STOPPED
    wheels = (0.0, 0.0)
    hand = None
    last_fid = -1
    last_frame_t = 0.0
    last_send = 0.0
    try:
        while True:
            now = time.monotonic()
            img, fid = serv.get_frame()

            if img is not None and fid != last_fid:
                last_fid, last_frame_t = fid, now
                hand = detector.detect(img) if detector is not None else None
                if go and error is None:
                    wheels = follower.step(hand, img.shape[1], H)
                else:
                    follower.reset()
                    wheels = (0.0, 0.0)
                cv2.imshow("CAMERA", draw(img, H, hand, wheels, go, error))
                send_now = True
            else:
                send_now = False
                if now - last_frame_t > FRAME_TIMEOUT:      # 영상이 끊기면 정지
                    wheels = (0.0, 0.0)

            # 새 영상마다 보내고, 영상이 없어도 SEND_INTERVAL 마다 다시 보낸다 (라파의 안전 정지를 막기 위해)
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
                    print(f"[손] 기준 크기 {H.target}px.  이 거리를 계속 쓰려면 위쪽을  TARGET_SIZE = {H.target}  로 바꾸세요")
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
