"""
main.py - 노트북 메인: 로봇을 실제로 움직이는 프로그램

실행:  python3 main.py            (라파에서 caramain.py 를 실행해 두면 영상이 온다)
먼저 camain.py 로 값을 맞춰 line_settings.json 을 만들어 둔다. 이 프로그램은 그 파일을 불러와 쓴다 (읽기만 한다).
(파일이 없으면 기본값으로 시작하고 콘솔에 알려 준다)
손 따라가기는 handmain.py 가 따로 한다 (이 프로그램은 라인 전용).

하는 일: 라파 영상을 받아 판단하고, 라파에 모터 핀 신호를 보낸다.
  "M,AIN1,AIN2,PWMA,BIN1,BIN2,PWMB"  또는  "STOP,-,-"      (만드는 곳: motor_signal.py)
  신호가 0.5초 넘게 끊기면 라파가 스스로 멈춘다.
SEND_TO_PI = False 이면 아무것도 보내지 않는다 (화면에 "NOT sent to Pi" 표시).

라인을 따라간다 (장애물이 있으면 우회).
USE_SIGN_AI = True 로 하면 앞이 막혔을 때 표지판 AI(best.pt)로 확인한다: 표지판이면 돌아가고(스탑 3번째면 정지), 아니면 우회.
  (이때만 sign_gate.py  sign_ai.py  best.pt  와 pip install ultralytics 가 필요)
키 (영상 창을 클릭한 상태에서)
  space  GO <-> STOP      g  GO      s  STOP (비상 정지)
  r      설정 파일을 다시 불러오기 (camain.py 로 값을 바꾼 뒤)
  q      종료

같은 폴더에 필요한 파일:  serv.py  engine.py  settings_io.py  linetrack.py  motor_signal.py  (손 관련 파일은 필요 없다)
"""
import os
import time

import cv2

import motor_signal
import serv
from engine import Engine
from settings_io import SETTINGS_FILE, load_settings, new_settings

SEND_TO_PI = True           # False: 라파로 바퀴 신호를 보내지 않는다 / True: 신호 전송 (바퀴가 움직인다)
START_STOPPED = False        # False: 시작부터 GO / True: 시작은 STOP (g 또는 space 로 출발)
USE_SIGN_AI = False          # True: 막히면 표지판 AI 로 확인 (표지판이면 돌아가기, 아니면 우회). 3번째 스탑 표지판이면 정지
SIGN_MODEL = "best.pt"       # 표지판 모델 경로 (상대 경로면 이 파일과 같은 폴더)
SIGN_LABELS = ("stop1", "stop2", "stop3")   # 스탑 표지판 클래스 이름 (모델 클래스 이름과 같아야 함. None = 모든 클래스)
SIGN_CONF = 0.5              # 표지판 검출 신뢰도 기준
STOP_LIMIT = 3               # 스탑 표지판을 이 번째로 만나면 정지
SEND_INTERVAL = 0.1          # 라파로 신호를 다시 보내는 최소 간격(초)
FRAME_TIMEOUT = 0.5          # 영상이 이 시간 넘게 안 오면 정지 신호


def main():
    S = new_settings()
    if not load_settings(S):
        print(f"[설정] {SETTINGS_FILE} 이 없어서 기본값으로 시작한다. camain.py 로 먼저 값을 맞추고 저장하세요.")
    S.p.stop_limit = STOP_LIMIT
    gate = None
    if USE_SIGN_AI:
        try:
            from sign_gate import SignGate
            path = SIGN_MODEL if os.path.isabs(SIGN_MODEL) else os.path.join(os.path.dirname(os.path.abspath(__file__)), SIGN_MODEL)
            gate = SignGate(path, labels=SIGN_LABELS, conf=SIGN_CONF)
            print("[표지판] AI 준비 완료")
        except Exception as e:                    # ultralytics 없음, 모델 파일 없음 등
            print(f"[표지판] AI 를 쓸 수 없어서 우회만 한다: {e}")
    serv.start()
    eng = Engine(S, sent=SEND_TO_PI, gate=gate)

    go = not START_STOPPED
    last_fid = -1
    last_frame_t = 0.0
    last_send = 0.0
    prev = time.monotonic()
    try:
        while True:
            now = time.monotonic()
            dt, prev = now - prev, now
            img, fid = serv.get_frame()

            if img is not None and fid != last_fid:
                last_fid, last_frame_t = fid, now
                view, _ = eng.update(img, now, dt, go)
                cv2.imshow("CAMERA", view)
                send_now = True
            else:
                send_now = False
                if now - last_frame_t > FRAME_TIMEOUT:      # 영상이 끊기면 정지 (우회 타이머도 같이 멈춤)
                    eng.hold(dt, stop=True)
                elif not go:                                # 정지 중이면 타이머만 멈춘다 (GO 중에는 시간이 그대로 흐른다)
                    eng.hold(dt, stop=True)

            # 새 영상마다 보내고, 영상이 없어도 SEND_INTERVAL 마다 다시 보낸다 (라파의 안전 정지를 막기 위해)
            if SEND_TO_PI and (send_now or now - last_send >= SEND_INTERVAL):
                if go and eng.wheels != (0.0, 0.0):
                    serv.send_command(motor_signal.to_signal(*eng.wheels))
                else:
                    serv.send_command("STOP,-,-")
                last_send = now

            key = cv2.waitKey(20) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                go = not go
                print("GO" if go else "STOP")
            elif key == ord("g"):
                go = True
                print("GO")
            elif key == ord("s"):
                go = False
                print("STOP")
            elif key == ord("r"):
                load_settings(S)
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