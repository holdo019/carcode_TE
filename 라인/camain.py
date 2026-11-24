"""
camain.py - 노트북 메인: 라인 찾기를 노트북에서 하고, 라파에는 바퀴 속도만 보낸다

실행:  python3 camain.py        (serv.py, linetrack.py 와 같은 폴더)

창
  CAMERA    라파 영상 + 자른 범위(노란 선) + 라인 위치 + 막힘 확인 창(초록=비어 있음, 빨강=막힘)
  MASK      라인으로 본 부분(초록 = 따라가는 덩어리, 빨강 = 두께가 안 맞아 버린 덩어리, 흰색 = 그 밖)
            맨 위 글자: 잰 선 두께(화면 맨 아래 기준 px). 이 숫자를 보고 WIDTH 를 맞춘다
  SETTINGS  슬라이더. 움직이면 바로 적용된다

키 (영상 창을 클릭한 상태에서)
  space  GO <-> STOP        g  GO        s  STOP (비상 정지)
  p      지금 슬라이더 값을 line_settings.json 에 저장  (다음 실행 때 자동으로 불러옴)
  q      종료 (종료할 때도 저장)

라파로 보내는 신호:  "W,왼쪽,오른쪽"  (바퀴 속도 0~100)  또는  "STOP,-,-"
신호가 0.5초 넘게 끊기면 라파가 스스로 멈춘다.
SEND_TO_PI = False 이면 신호를 보내지 않는다. 화면에는 "보냈다면 이 속도" 가 그대로 표시되므로
모터 없이 라인 인식과 슬라이더만 맞춰 볼 수 있다. 바퀴를 움직일 준비가 되면 True 로 바꾼다.
"""
import json
import os
import time

import cv2
import numpy as np

import serv
from junction import RULE, classify, clean_mask, decide, find_bar, kind_name
from linetrack import Driver, FOLLOW, Params, blocked_info, detect

SEND_TO_PI = True           # False: 라파로 바퀴 신호를 보내지 않는다 (영상과 이미지 설정만 확인) / True: 바퀴 신호 전송
START_STOPPED = True        # False: 시작부터 GO / True: 시작은 STOP (g 또는 space 로 출발)
SEND_INTERVAL = 0.1          # 라파로 신호를 다시 보내는 최소 간격(초)
FRAME_TIMEOUT = 0.5          # 영상이 이 시간 넘게 안 오면 정지 신호
SETTINGS_FILE = "line_settings.json"

# 슬라이더: (이름, 최대값, 기본값 가져오는 함수, 값 -> Params 에 넣는 함수)
SLIDERS = [
    ("SPEED",      100, lambda P: P.speed,                  lambda P, v: setattr(P, "speed", v)),
    ("KP x10",      40, lambda P: int(P.kp * 10),           lambda P, v: setattr(P, "kp", v / 10)),
    ("ROI top %",   95, lambda P: int(P.roi_top * 100),     lambda P, v: setattr(P, "roi_top", v / 100)),
    ("FAR bottom %", 95, lambda P: int(P.far_bottom * 100), lambda P, v: setattr(P, "far_bottom", v / 100)),
    ("THRESH(0=auto)", 255, lambda P: P.thresh,             lambda P, v: setattr(P, "thresh", v)),
    ("MIN AREA",  3000, lambda P: P.min_area,               lambda P, v: setattr(P, "min_area", v)),
    ("DARK LINE",    1, lambda P: int(P.dark_line),         lambda P, v: setattr(P, "dark_line", bool(v))),
    ("COLOR DIFF", 255, lambda P: P.color_diff,             lambda P, v: setattr(P, "color_diff", v)),
    ("BLOCK %",    100, lambda P: int(P.block_ratio * 100), lambda P, v: setattr(P, "block_ratio", v / 100)),
    ("AVOID",        1, lambda P: int(P.avoid),             lambda P, v: setattr(P, "avoid", bool(v))),
    ("T_OUT x0.1s", 50, lambda P: int(P.t_out * 10),        lambda P, v: setattr(P, "t_out", v / 10)),
    ("T_PASS x0.1s", 50, lambda P: int(P.t_pass * 10),      lambda P, v: setattr(P, "t_pass", v / 10)),
    ("T_BACK x0.1s", 50, lambda P: int(P.t_back * 10),      lambda P, v: setattr(P, "t_back", v / 10)),
    ("WIDTH px(0=off)", 300, lambda P: P.width,             lambda P, v: setattr(P, "width", v)),
    ("WIDTH TOL %",  90, lambda P: int(P.width_tol * 100),  lambda P, v: setattr(P, "width_tol", v / 100)),
    ("TAPER %",     100, lambda P: int(P.taper * 100),      lambda P, v: setattr(P, "taper", v / 100)),
]


def load_settings(P):
    """저장된 슬라이더 값이 있으면 불러온다"""
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE) as f:
                saved = json.load(f)
            for name, _, _, setter in SLIDERS:
                if name in saved:
                    setter(P, int(saved[name]))
            print(f"[설정] {SETTINGS_FILE} 불러옴")
        except (OSError, ValueError) as e:
            print(f"[설정] 불러오기 실패: {e}")


def save_settings(P):
    values = {name: getter(P) for name, _, getter, _ in SLIDERS}
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(values, f, indent=2, ensure_ascii=False)
        print(f"[설정] {SETTINGS_FILE} 저장: {values}")
    except OSError as e:
        print(f"[설정] 저장 실패: {e}")


def setup_windows(P):
    cv2.namedWindow("SETTINGS", cv2.WINDOW_NORMAL)
    for name, maximum, getter, _ in SLIDERS:
        cv2.createTrackbar(name, "SETTINGS", getter(P), maximum, lambda v: None)
    cv2.resizeWindow("SETTINGS", 480, 60 + 40 * len(SLIDERS))


def read_sliders(P):
    for name, _, _, setter in SLIDERS:
        setter(P, cv2.getTrackbarPos(name, "SETTINGS"))


def draw_camera(img, P, res, ratio, wheels, state, go):
    """영상 위에 자른 범위, 라인 위치, 막힘 창, 상태를 그린다"""
    view = img.copy()
    h, w = view.shape[:2]
    cv2.line(view, (0, res["y0"]), (w, res["y0"]), (0, 255, 255), 1)         # near 영역 위쪽 경계 (노랑)
    if res.get("far_y1"):
        cv2.line(view, (0, res["far_y1"]), (w, res["far_y1"]), (0, 140, 255), 1)   # far 영역 아래쪽 경계 (주황)
    x0, x1, y0, y1 = P.win
    blocked = ratio >= P.block_ratio
    cv2.rectangle(view, (int(w * x0), int(h * y0)), (int(w * x1), int(h * y1)),
                  (0, 0, 255) if blocked else (0, 200, 0), 2)
    cv2.line(view, (w // 2, res["y0"]), (w // 2, h), (255, 255, 255), 1)    # 화면 중앙
    if res["found"]:
        cx = int(res["cx"])
        cv2.line(view, (cx, res["y0"]), (cx, h), (255, 0, 255), 3)         # 라인 중심
    color = (0, 200, 0) if go else (0, 0, 255)
    cv2.putText(view, f"{'GO' if go else 'STOP'}  {state}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(view, f"line={'yes' if res['found'] else 'LOST'} err={res['err']:+.2f} blocked={ratio:.2f}",
                (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(view, f"left={wheels[0]:.0f} right={wheels[1]:.0f}" + ("" if SEND_TO_PI else "  (NOT sent to Pi)"), (10, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return view


def far_info(res, P, stem_x):
    """far 영역(위쪽 갈림길 인식 범위)의 가로선과 종류 -> (가로선 또는 None, 종류 글자, far 마스크). 화면 표시용"""
    full, fy = res["mask_full"], res["far_y1"]
    if not fy:
        return None, "far: off (FAR bottom 너무 작음)", None
    h, w = full.shape[:2]
    area = max(1, int(P.min_area * (fy / h) / 0.30))
    far = full[:fy]
    bar = find_bar(far, area, w)
    if bar is None:
        return None, "far: no bar", far
    ways = classify(clean_mask(far, area), bar, stem_x if stem_x is not None else w / 2, w)
    return bar, f"far: {kind_name(ways)} -> {decide(ways, RULE)}", far


def draw_far(res, bar, text, far):
    """FAR 창: far 영역 마스크 + 가로선 네모 + 종류"""
    if far is None:
        return None
    vis = cv2.cvtColor(far, cv2.COLOR_GRAY2BGR)
    if bar is not None:
        cv2.rectangle(vis, (bar[2], bar[0]), (bar[3], bar[1]), (0, 140, 255), 2)
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 32), (0, 0, 0), -1)
    cv2.putText(vis, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return vis


def draw_mask(res, P):
    mask = res["mask"]
    vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    if res["rejected"] is not None:
        vis[res["rejected"]] = (0, 0, 255)                  # 빨강 = 두께가 안 맞아 버린 덩어리
    if res["chosen"] is not None:
        vis[res["chosen"]] = (0, 255, 0)
    text = "width: -" if res["width"] is None else f"width={res['width']:.0f}px (at bottom)"
    color = (0, 255, 255)
    if P.width > 0:
        text += f"  target={P.width} tol={P.width_tol * 100:.0f}%  " + ("OK" if res["width_ok"] else "REJECT")
        color = (0, 255, 0) if res["width_ok"] else (0, 0, 255)
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 32), (0, 0, 0), -1)         # 글자가 잘 보이게 검은 띠
    cv2.putText(vis, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return vis


def main():
    P = Params()
    load_settings(P)
    setup_windows(P)
    serv.start()

    driver = Driver()
    go = not START_STOPPED
    last_fid = -1
    last_frame_t = 0.0
    last_send = 0.0
    prev = time.monotonic()
    wheels = (0.0, 0.0)
    stem_x = None
    try:
        while True:
            now = time.monotonic()
            dt, prev = now - prev, now
            read_sliders(P)
            img, fid = serv.get_frame()

            if img is not None and fid != last_fid:
                last_fid, last_frame_t = fid, now
                res = detect(img, P)
                ratio, ratio_l, ratio_r = blocked_info(img, P)
                if go:
                    left, right = driver.step(now, res["found"], res["err"], ratio, ratio_l, ratio_r, P)
                    wheels = (max(0, min(100, left)), max(0, min(100, right)))
                else:
                    driver.hold(dt)
                    wheels = (0.0, 0.0)
                if res["found"]:
                    stem_x = res["cx"]
                bar, ftext, far = far_info(res, P, stem_x)
                view = draw_camera(img, P, res, ratio, wheels, driver.state, go)
                if bar is not None:
                    cv2.rectangle(view, (bar[2], bar[0]), (bar[3], bar[1]), (0, 140, 255), 2)
                cv2.putText(view, ftext, (10, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
                cv2.imshow("CAMERA", view)
                fv = draw_far(res, bar, ftext, far)
                if fv is not None:
                    cv2.imshow("FAR", fv)
                cv2.imshow("MASK", draw_mask(res, P))
                send_now = True
            else:
                send_now = False
                if now - last_frame_t > FRAME_TIMEOUT:      # 영상이 끊기면 정지
                    wheels = (0.0, 0.0)
                    driver.hold(dt)
                elif not go:
                    driver.hold(dt)

            # 새 영상마다 보내고, 영상이 없어도 SEND_INTERVAL 마다 다시 보낸다 (라파의 안전 정지를 막기 위해)
            if SEND_TO_PI and (send_now or now - last_send >= SEND_INTERVAL):
                if go and wheels != (0.0, 0.0):
                    serv.send_command(f"W,{wheels[0]:.0f},{wheels[1]:.0f}")
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
            elif key == ord("p"):
                save_settings(P)
    except KeyboardInterrupt:
        pass
    finally:
        save_settings(P)
        if SEND_TO_PI:
            serv.send_command("STOP,-,-")
            time.sleep(0.2)
        serv.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()