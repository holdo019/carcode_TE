"""
linetrack.py - 노트북에서 돌아가는 라인 찾기 + 앞이 막혔는지 판단 + 우회 (라파 카메라 1대)

main.py 가 가져다 쓴다. 영상은 라파가 보낸 BGR 이미지(cv2.imdecode 결과).
  detect(img, P)        -> dict(found, err, cx, mask, chosen, y0, width, width_ok, rejected)   라인 찾기
  blocked_info(img, P)  -> (전체, 왼쪽, 오른쪽) 비율                   앞이 막혔는지
  Driver.step(...)      -> (왼쪽 바퀴, 오른쪽 바퀴) 0~100            주행 / 우회 상태 기계

바퀴는 앞으로만 돈다 (회전 = 한쪽 바퀴만 느리게 또는 정지).
"""
from dataclasses import dataclass

import cv2
import numpy as np

FOLLOW, OUT, PASS, BACK, FIND = "FOLLOW", "OUT", "PASS", "BACK", "FIND"
UTURN, DONE, TURN = "UTURN", "DONE", "TURN"

NEAR_REF = 0.30      # MIN AREA 는 가까운 영역 높이가 화면의 이 비율일 때의 값. 영역 높이에 비례해 늘리고 줄인다
MIN_REGION_PX = 24   # 영역 높이가 이보다 작으면 그 영역은 쓰지 않는다


@dataclass
class Params:
    speed: int = 25            # 직진 속도 (PWM 0~100)
    kp: float = 1.0            # 조향 세기
    roi_top: float = 0.70      # near(라인 따라가기) 영역: 이 높이 아래만 본다 (0=맨 위, 1=맨 아래). 0.70 = 아래 30%
    far_bottom: float = 0.35   # far(갈림길/코너 감지) 영역: 이 높이 위만 본다. 0.35 = 위 35%. near 와 겹쳐도 된다 (0 = far 안 씀)
    approach_scale: float = 0.6   # 코너가 예고되면 속도를 이 배수로 줄인다
    t_turn_min: float = 0.6    # 코너 회전: 최소 이 시간은 돈다
    t_turn_max: float = 4.0    # 코너 회전: 라인이 다시 가운데 안 와도 이 시간이면 그만 돈다
    thresh: int = 0            # 흑백 기준값. 0 = 자동(Otsu), 1~255 = 직접 지정
    min_area: int = 300        # 라인 덩어리가 이보다 작으면 못 찾은 것
    dark_line: bool = True     # True: 밝은 바닥의 어두운 선 / False: 어두운 바닥의 밝은 선
    color_diff: int = 60       # 바닥색과 이만큼 다르면 '바닥이 아님'
    block_ratio: float = 0.35  # 창의 이 비율이 바닥이 아니면 막힌 것 (라인만 있으면 약 0.15)
    block_frames: int = 3      # 연속 이 프레임 수 막혀야 확정
    avoid: bool = True         # True: 우회 / False: 멈추고 기다림
    t_out: float = 0.7         # 우회: 옆으로 도는 시간 (한 바퀴만)
    t_pass: float = 1.2        # 우회: 장애물 옆을 직진하는 시간
    t_back: float = 0.7        # 우회: 라인 쪽으로 되돌아 도는 시간
    t_find_max: float = 5.0    # 우회: 라인 찾기를 포기하는 시간
    cooldown: float = 1.5      # 우회가 끝난 뒤 이 시간 동안은 '막힘'을 무시
    win: tuple = (0.30, 0.70, 0.60, 0.95)   # 막힘 확인 창 (x0, x1, y0, y1), 화면 대비 비율
    width: int = 0             # 선 두께 검사: 화면 맨 아래에서의 목표 두께(픽셀). 0 = 검사 안 함 (기존 동작)
    width_tol: float = 0.35    # 두께 허용 오차 (0.35 = 목표의 ±35%)
    taper: float = 0.6         # 원근: 화면 맨 위의 선 두께 / 맨 아래의 선 두께 (1.0 = 원근 무시)
    min_wheel: int = 15        # 모터가 도는 최소 PWM. 이보다 낮게 계산된 속도는 올리거나 0 으로 (0 = 이 보정을 끔)
    lost_timeout: float = 4.0  # 라인을 이 시간(초) 넘게 못 찾으면 제자리에서 돌지 않고 정지 (라인이 보이면 다시 출발)
    stop_limit: int = 3        # 표지판 AI: 스탑 표지판을 이 번째로 만나면 정지 (그 전에는 돌아간다)
    t_uturn_min: float = 1.5   # 돌아가기: 최소 이 시간은 돈다
    t_uturn_max: float = 6.0   # 돌아가기: 라인을 못 찾아도 이 시간이면 그만 돈다
    uturn_dir: str = "left"    # 돌아가기: 도는 방향 (한쪽 바퀴만)


# ------------------------------------------------------------
# 라인 찾기
# ------------------------------------------------------------
def line_width(labels, stats, i, P, roi_h):
    """덩어리 i 의 두께를 여러 높이(최대 8곳)에서 가로로 재서 중앙값을 낸다.
    원근 때문에 위쪽 선은 좁으므로, 높이마다 taper 비율로 나눠 '화면 맨 아래에서의 두께(픽셀)' 로 환산한다.
    중앙값이라 갈림길의 가로 라인처럼 몇 줄만 넓은 것은 견딘다."""
    x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
    w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
    counts = (labels[y:y + h, x:x + w] == i).sum(axis=1)
    taper = min(1.0, max(0.1, P.taper))
    vals = []
    for r in np.linspace(0, h - 1, min(8, h)).astype(int):
        factor = taper + (1 - taper) * ((y + r) / max(1, roi_h - 1))     # 맨 아래 1.0 ... 맨 위 taper
        vals.append(counts[r] / factor)
    return float(np.median(vals))


def regions(h, P):
    """화면 높이 h(픽셀)에서 두 영역의 픽셀 범위. 모든 비율은 전체 화면 높이 기준이고 여기서만 픽셀로 바꾼다.
    반환: (near_y0, far_y1).  near = [near_y0, h), far = [0, far_y1).  far 가 쓸 수 없으면 far_y1 = 0"""
    y0 = max(0, min(int(h * P.roi_top), h - MIN_REGION_PX))
    y1 = int(h * P.far_bottom)
    return y0, (y1 if y1 >= MIN_REGION_PX else 0)


def scale_area(area, region_h, h):
    """MIN AREA 같은 면적 기준을 영역 높이에 맞춘다 (기준: 영역 높이가 화면의 NEAR_REF 일 때)"""
    return max(1, int(area * (region_h / h) / NEAR_REF))


def detect(img, P):
    """화면 전체를 한 번만 흑백으로 나눈다 (기준값은 near 영역에서 구해 전체에 쓴다).
    near 영역: 덩어리 중 라인을 고른다. P.width = 0 이면 가장 큰 덩어리, > 0 이면 큰 덩어리부터 두께를 재서 맞는 첫 덩어리.
    far 영역은 갈림길/코너 판단용 마스크만 돌려 준다 (junction.py).
    err: -1(맨 왼쪽) ~ +1(맨 오른쪽).   width = 잰 두께(맨 아래 기준 픽셀), width_ok = 두께 통과 여부,
    rejected = 두께가 안 맞아 버린 덩어리 (화면 표시용).   mask = near 영역, mask_full = 전체, far_y1 = far 영역 끝 높이"""
    h, w = img.shape[:2]
    y0, far_y1 = regions(h, P)
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (7, 7), 0)
    mode = cv2.THRESH_BINARY_INV if P.dark_line else cv2.THRESH_BINARY
    if P.thresh > 0:
        t = P.thresh
    else:
        t, _ = cv2.threshold(gray[y0:], 0, 255, mode + cv2.THRESH_OTSU)     # 기준값은 near 에서
    _, full = cv2.threshold(gray, t, 255, mode)
    full = cv2.morphologyEx(full, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))   # 작은 점 제거
    mask = full[y0:]
    min_area = scale_area(P.min_area, h - y0, h)

    res = {"found": False, "err": 0.0, "cx": None, "mask": mask, "chosen": None, "y0": y0,
           "width": None, "width_ok": None, "rejected": None,
           "mask_full": full, "far_y1": far_y1}
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    if n > 1:
        for i in 1 + np.argsort(-stats[1:, cv2.CC_STAT_AREA]):           # 큰 덩어리부터
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                break
            wd = line_width(labels, stats, i, P, mask.shape[0])
            ok = True
            if P.width > 0:
                ok = abs(wd / P.width - 1) <= max(P.width_tol, 0.05)
            if ok:
                cx = float(centroids[i][0])
                res.update(found=True, err=(cx - w / 2) / (w / 2), cx=cx, chosen=(labels == i),
                           width=wd, width_ok=True if P.width > 0 else None)
                break
            if res["width"] is None:                                     # 버린 것 중 가장 큰 덩어리의 두께를 보여 준다
                res["width"], res["width_ok"] = wd, False
            rej = labels == i
            res["rejected"] = rej if res["rejected"] is None else (res["rejected"] | rej)
    return res


# ------------------------------------------------------------
# 앞이 막혔는지: 화면 아래 가운데 창에서 '바닥색과 다른' 픽셀의 비율
# ------------------------------------------------------------
def blocked_info(img, P):
    h, w = img.shape[:2]
    y0 = regions(h, P)[0]
    edge = int(w * 0.15)
    zone = img[y0:, :]
    side = np.concatenate([zone[:, :edge].reshape(-1, 3), zone[:, w - edge:].reshape(-1, 3)])
    floor = np.median(side, axis=0)                      # 바닥색 = 같은 화면 좌우 가장자리의 중앙값

    x0, x1, wy0, wy1 = P.win
    win = img[int(h * wy0):int(h * wy1), int(w * x0):int(w * x1)].astype(np.float32)
    diff = np.linalg.norm(win - floor.astype(np.float32), axis=2) > P.color_diff
    half = diff.shape[1] // 2
    return float(diff.mean()), float(diff[:, :half].mean()), float(diff[:, half:].mean())


# ------------------------------------------------------------
# 주행 / 우회 상태 기계. (왼쪽, 오른쪽) 바퀴 속도 0~100 을 돌려준다
# ------------------------------------------------------------
class Driver:
    def __init__(self):
        self.state = FOLLOW
        self.t_state = 0.0
        self.avoid_dir = "left"
        self.block_count = 0
        self.block_until = 0.0
        self.last_err = 0.0
        self.stop_count = 0           # 지금까지 만난 스탑 표지판 수
        self.lost_since = None        # 라인을 놓치기 시작한 때 (보이면 None)
        self.warned_lost = False
        self.warned = False
        self.turn_dir = "right"

    def _set(self, state, now):
        if state != self.state:
            print(f"[상태] {self.state} -> {state}")
        self.state = state
        self.t_state = now
        self.lost_since = None

    def hold(self, dt):
        """멈춰 있는 동안 호출: 타이머가 같이 멈춰서 우회가 하던 자리에서 이어진다"""
        self.t_state += dt
        self.block_until += dt
        if self.lost_since is not None:
            self.lost_since += dt

    @staticmethod
    def _pivot(direction, speed):
        """한쪽 바퀴만: 오른쪽으로 = 왼쪽 바퀴만 돈다, 왼쪽으로 = 오른쪽 바퀴만 돈다"""
        return (speed, 0) if direction == "right" else (0, speed)

    @staticmethod
    def _lift(l, r, P):
        """모터가 안 도는 낮은 PWM 을 보정한다 (P.min_wheel).
        빠른 쪽 바퀴가 min_wheel 보다 낮으면 두 바퀴를 같은 비율로 올려 조향 비율은 유지한다.
        느린 쪽 바퀴는 min_wheel 의 절반 이상이면 min_wheel 로 올리고, 절반 미만이면 0 (한쪽 바퀴만)"""
        m = P.min_wheel
        l, r = max(0.0, l), max(0.0, r)
        if m <= 0 or max(l, r) <= 0:
            return l, r
        hi = max(l, r)
        if hi < m:
            k = m / hi
            l, r = l * k, r * k
        lo = min(l, r)
        if 0 < lo < m:
            lo = m if lo >= m / 2 else 0.0
            l, r = (lo, r) if l <= r else (l, lo)
        return l, r

    def step(self, now, found, err, ratio, ratio_l, ratio_r, P, ai=False, verdict=None, corner=None):
        """(왼쪽, 오른쪽) 바퀴 속도 0~100.  계산은 _step, 여기서는 모터가 안 도는 낮은 속도만 보정한다"""
        if P.min_wheel > 0 and 0 < P.speed < P.min_wheel and not self.warned:
            print(f"[주의] SPEED {P.speed} 가 MIN_WHEEL {P.min_wheel} 보다 낮아 모터가 안 돈다 -> {P.min_wheel} 이상으로 올려 보낸다")
            self.warned = True
        l, r = self._step(now, found, err, ratio, ratio_l, ratio_r, P, ai, verdict, corner)
        if self.state == DONE:
            return 0, 0
        return self._lift(l, r, P)

    def _step(self, now, found, err, ratio, ratio_l, ratio_r, P, ai=False, verdict=None, corner=None):
        """corner = junction.py 의 판단 (None 이면 평소).  dict(phase="approach"|"bar"|"turn", dir="left"|"right"|"straight")
        approach = 코너가 멀리 보임: 천천히.  bar = 가로선이 가까이 옴: 조향을 끄고 천천히 직진.  turn = 지금 돌 때"""
        s = P.speed
        el = now - self.t_state

        if self.state == FOLLOW:
            blocked = ratio >= P.block_ratio and now >= self.block_until
            self.block_count = self.block_count + 1 if blocked else 0
            if self.block_count >= P.block_frames:
                if ai:                                    # 표지판 AI 를 쓰는 중: 판단이 나올 때까지 선다
                    if verdict is None:
                        return 0, 0
                    if verdict == "sign":                 # 스탑 표지판이면 돌아간다 (3번째면 정지)
                        self.block_count = 0
                        self.stop_count += 1
                        print(f"[표지판] 스탑 {self.stop_count}/{P.stop_limit}")
                        if self.stop_count >= P.stop_limit:
                            self._set(DONE, now)
                            return 0, 0
                        self._set(UTURN, now)
                        return self._pivot(P.uturn_dir, s)
                    # verdict == "obstacle": 아래의 우회로 이어진다
                if not P.avoid:
                    return 0, 0                           # 길이 빌 때까지 정지
                self.avoid_dir = "left" if ratio_l <= ratio_r else "right"    # 덜 막힌 쪽
                print(f"[막힘] 비율 {ratio:.2f} -> {self.avoid_dir} 쪽으로 우회")
                self.block_count = 0
                self._set(OUT, now)
                return self._pivot(self.avoid_dir, s)
            if corner is not None:
                if corner["phase"] == "turn" and corner["dir"] in ("left", "right"):
                    print(f"[코너] {corner['dir']} 으로 돈다")
                    self.turn_dir = corner["dir"]
                    self.block_count = 0
                    self._set(TURN, now)
                    return self._pivot(self.turn_dir, s * P.approach_scale)
                sc = s * P.approach_scale
                if corner["phase"] == "bar":              # 가로선이 보인다: 조향 끔 (라인 중심이 가로선에 끌려가므로)
                    return sc, sc
                if found:                                 # approach: 느리게, 조향은 그대로
                    self.last_err = err
                    self.lost_since = None
                    return sc * (1 + P.kp * err), sc * (1 - P.kp * err)
            if found:
                self.last_err = err
                self.lost_since = None
                self.warned_lost = False
                steer = P.kp * err                        # err > 0: 라인이 오른쪽 -> 오른쪽으로
                return s * (1 + steer), s * (1 - steer)
            # 라인을 놓침: 마지막에 본 쪽으로 한쪽 바퀴만 돌려 찾는다. lost_timeout 이 지나면 정지 (보이면 다시 출발)
            if self.lost_since is None:
                self.lost_since = now
            if P.lost_timeout > 0 and now - self.lost_since > P.lost_timeout:
                if not self.warned_lost:
                    print(f"[라인 못 찾음] {P.lost_timeout:.0f}초 넘게 라인이 안 보여 정지한다 (라인이 보이면 다시 출발)")
                    self.warned_lost = True
                return 0, 0
            return self._pivot("right" if self.last_err >= 0 else "left", s)

        if self.state == TURN:                            # 직각 코너: 한쪽 바퀴만 돌아 near 라인이 다시 가운데 오면 끝
            bar = corner is not None and corner.get("bar", False)
            if (el >= P.t_turn_min and found and abs(err) < 0.5 and not bar) or el >= P.t_turn_max:
                self.block_until = now + P.cooldown
                self.block_count = 0
                self._set(FOLLOW, now)
            return self._pivot(self.turn_dir, s * P.approach_scale)

        if self.state == DONE:                            # 3번째 스탑 표지판: 끝
            return 0, 0

        if self.state == UTURN:                           # 한쪽 바퀴만 돌아 뒤로 돈다. 라인이 다시 가운데에 보이면 끝
            if (el >= P.t_uturn_min and found and abs(err) < 0.5) or el >= P.t_uturn_max:
                self.block_until = now + P.cooldown
                self.block_count = 0
                self._set(FOLLOW, now)
            return self._pivot(P.uturn_dir, s)

        if self.state == OUT:                             # 우회할 쪽으로 돈다
            if el >= P.t_out:
                self._set(PASS, now)
            return self._pivot(self.avoid_dir, s)

        if self.state == PASS:                            # 장애물 옆을 직진
            if el >= P.t_pass:
                self._set(BACK, now)
            return s, s

        if self.state == BACK:                            # 라인 쪽으로 되돌아 돈다
            if el >= P.t_back:
                self._set(FIND, now)
            return self._pivot("right" if self.avoid_dir == "left" else "left", s)

        # FIND: 라인 쪽으로 휘면서 천천히 전진, 라인이 보이면 FOLLOW
        if (found and abs(err) < 0.5) or el >= P.t_find_max:
            self.block_until = now + P.cooldown
            self.block_count = 0
            self._set(FOLLOW, now)
        if self.avoid_dir == "left":                      # 라인은 오른쪽에 있다
            return s * 0.8, s * 0.2
        return s * 0.2, s * 0.8