"""
fork_view.py - 갈림길 확인용 도구 (바퀴는 절대 안 움직인다. 화면 표시와 콘솔 출력만 한다)

실행:  python fork_view.py        (라파에서 caramain.py 를 실행해 두면 영상이 온다)
필요한 파일:  serv.py  linetrack.py  settings_io.py  line_settings.json (camain.py 로 만든 것. 없으면 기본값)

라인 마스크의 '먼 쪽 띠'를 왼쪽 / 가운데 / 오른쪽 3칸으로 나눠 각 칸에 라인이 있는지 0/1 로 보여 준다.
    0,1,0  직선      1,0,1  양갈래      1,1,0  직진+왼쪽 길      0,1,1  직진+오른쪽 길
    1,1,1  가로로 가로지름      1,0,0 / 0,0,1  옆으로만      0,0,0  길 끝(막다른 길 / 라인 놓침)
값이 바뀔 때마다 콘솔에 순서를 찍는다.  로봇을 갈림길과 고리 입구 위로 손으로 천천히 밀어 지나가며
순서가 어떻게 나오는지 본다.

키:  q 종료      r 설정 파일 다시 불러오기
"""
import time

import cv2
import numpy as np

import serv
from settings_io import load_settings, new_settings

# ---------------- 설정 ----------------
BAND = (0.0, 0.45)       # 라인 마스크(자른 영역)에서 볼 띠: 위(먼 쪽)에서 0% ~ 45%.  (0,0.5) 처럼 바꿔 본다
MIN_FILL = 0.06          # 한 칸에서 라인 픽셀이 이 비율 이상이면 1
MIN_BLOB = 0.25          # P.min_area 의 이 비율보다 작은 점은 무시 (노이즈)
STABLE_FRAMES = 3        # 같은 값이 이 프레임 수 연속이어야 바뀐 것으로 인정
HISTORY = 8              # 화면에 보여 줄 최근 변화 개수
# --------------------------------------

NAMES = {
    (0, 1, 0): "straight", (1, 0, 1): "FORK (left+right)", (1, 1, 0): "straight + left",
    (0, 1, 1): "straight + right", (1, 1, 1): "cross (all)", (1, 0, 0): "left only",
    (0, 0, 1): "right only", (0, 0, 0): "end / lost",
}


def bits_from_mask(mask, min_area, band=BAND, min_fill=MIN_FILL, min_blob=MIN_BLOB):
    """마스크 -> ((왼쪽, 가운데, 오른쪽) 0/1, (띠 시작 y, 끝 y), 칸별 채움 비율)"""
    h, w = mask.shape[:2]
    clean = np.zeros_like(mask)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area * min_blob:
            clean[labels == i] = 255
    y1, y2 = int(h * band[0]), max(int(h * band[0]) + 1, int(h * band[1]))
    bits, fills = [], []
    for k in range(3):
        cell = clean[y1:y2, k * w // 3:(k + 1) * w // 3]
        fill = float((cell > 0).mean()) if cell.size else 0.0
        fills.append(fill)
        bits.append(1 if fill >= min_fill else 0)
    return tuple(bits), (y1, y2), fills


class Stable:
    """같은 값이 STABLE_FRAMES 번 연속이어야 바뀐 것으로 인정"""
    def __init__(self):
        self.cur = None
        self.cand = None
        self.count = 0

    def update(self, bits):
        if bits == self.cur:
            self.cand, self.count = None, 0
            return False
        if bits == self.cand:
            self.count += 1
        else:
            self.cand, self.count = bits, 1
        if self.count >= STABLE_FRAMES:
            self.cur, self.cand, self.count = bits, None, 0
            return True
        return False


def draw(img, y0, band_y, bits, fills, history):
    view = img.copy()
    h, w = view.shape[:2]
    for k in range(3):
        x1, x2 = k * w // 3, (k + 1) * w // 3
        color = (0, 200, 0) if bits[k] else (0, 0, 200)
        cv2.rectangle(view, (x1 + 2, y0 + band_y[0]), (x2 - 2, y0 + band_y[1]), color, 2)
        cv2.putText(view, f"{fills[k]:.2f}", (x1 + 8, y0 + band_y[0] + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(view, f"{bits[0]},{bits[1]},{bits[2]}  {NAMES.get(bits, '?')}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    if history:
        cv2.putText(view, " > ".join("".join(map(str, b)) for b in history[-HISTORY:]), (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return view


def main():
    from linetrack import detect
    S = new_settings()
    if not load_settings(S):
        print("[설정] line_settings.json 이 없어서 기본값으로 시작한다")
    serv.start()
    stable = Stable()
    history = []
    shown = (0, 0, 0)
    last_fid = -1
    try:
        while True:
            img, fid = serv.get_frame()
            if img is not None and fid != last_fid:
                last_fid = fid
                res = detect(img, S.p)
                bits, band_y, fills = bits_from_mask(res["mask"], S.p.min_area)
                if stable.update(bits):
                    shown = bits
                    history.append(bits)
                    print(f"{bits[0]},{bits[1]},{bits[2]}  {NAMES.get(bits, '?')}    (순서: "
                          + " > ".join("".join(map(str, b)) for b in history[-HISTORY:]) + ")")
                cv2.imshow("FORK", draw(img, res["y0"], band_y, shown, fills, history))
                cv2.imshow("MASK", res["mask"])
            key = cv2.waitKey(20) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                load_settings(S)
    except KeyboardInterrupt:
        pass
    finally:
        serv.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()