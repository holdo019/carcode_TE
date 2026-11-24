"""
junction.py - 직각 코너 / 갈림길 처리 (오른손 규칙).  화면 위쪽 far 영역으로 미리 보고, 가까이 오면 돌라고 신호를 준다

    jn = Junction()
    corner = jn.update(res, now, P)     # 새 영상마다 (FOLLOW / TURN 중).  res = linetrack.detect() 결과
                                        # None = 평소.  아니면 dict(phase, dir, bar)  -> Driver.step(..., corner=corner)
    jn.active                           # True 면 코너/갈림길을 지나는 중 ('막힘' 판단을 무시한다)
    jn.idle(now, cooldown=False)        # 우회/유턴 중이거나 정지 중일 때

흐름
  1. far 영역(화면 위쪽)에 '가로선'(가로로 길게 이어진 라인)이 보이면 코너/갈림길이 예고된 것 -> phase "approach" (속도를 줄인다)
     가로선이 라인(세로 줄기)의 왼쪽/오른쪽으로 뻗었는지, 줄기가 가로선 너머로 계속되는지 보고 종류를 안다:
       오른쪽 코너  왼쪽 코너  T자(양쪽)  +자(양쪽+직진)  직진+옆길
  2. 오른손 규칙: 갈 수 있는 길 중 오른쪽 > 직진 > 왼쪽 순서로 고른다 (RULE).
  3. 가로선이 near 영역(가까운 쪽)에 들어오면 phase "bar" (조향을 끄고 직진: 라인 중심이 가로선에 끌려가므로).
  4. 가로선이 near 영역의 TRIGGER 높이에 닿으면 신호: phase "turn" (고른 방향이 left/right 일 때) -> Driver 가 TURN 상태로 돈다.
     직진을 골랐으면 돌지 않고 가로선이 지나갈 때까지 조향만 끈다.
  5. TURN 은 near 라인이 다시 가운데에 보이면 끝난다 (Driver).  끝난 뒤 COOLDOWN 초는 새로 세지 않는다.
모든 높이는 화면 높이에 대한 비율이고 (Params.roi_top = near 시작, far_bottom = far 끝), 면적 기준은 영역 높이에 맞춰 scale 된다.
"""
import cv2
import numpy as np

# ---------------- 설정 ----------------
RULE = "right"           # 오른손 규칙: "right" 이면 오른쪽 > 직진 > 왼쪽.  "left" 면 왼쪽 > 직진 > 오른쪽
BAR_FRAC = 0.30          # 가로로 이만큼(화면 폭 비율) 이어진 라인이 있어야 '가로선'
BAR_ROWS = 3             # 가로선은 최소 이만큼의 줄 수 (픽셀) 두께
SIDE_MARGIN = 0.12       # 가로선 끝이 줄기에서 화면 폭의 이 비율 이상 벗어나야 그쪽 '길이 있다'
STRAIGHT_BAND = 0.06     # 가로선 위쪽에 줄기가 이어지는지 볼 때 줄기 좌우 폭 (화면 폭 비율)
STRAIGHT_FILL = 0.6      # 가로선 위쪽 줄 중 이 비율 이상에 라인이 있으면 '직진 길이 있다' (선 굵기와 무관)
MIN_BLOB = 0.25          # 영역 min_area 의 이 비율보다 작은 점은 무시 (노이즈)
STABLE_FRAMES = 3        # 가로선이 이 프레임 수 연속 보여야 인정
TRIGGER = 0.5            # 가로선 중심이 near 영역 높이의 이 비율(0=위, 1=아래)에 닿으면 돌라는 신호
LOST_FAR_TIME = 1.0      # 예고 뒤 가로선이 이 시간 사라지고 near 에도 없으면 오인식으로 보고 취소
MAX_TIME = 8.0           # 이 시간(초)이 지나면 무조건 끝
COOLDOWN = 1.5           # 끝난 뒤 이 시간(초) 동안은 새로 세지 않음
# --------------------------------------

NAMES = {
    (0, 1, 0): "straight", (1, 0, 1): "FORK (left+right)", (1, 1, 0): "straight + left",
    (0, 1, 1): "straight + right", (1, 1, 1): "cross (all)", (1, 0, 0): "left only",
    (0, 0, 1): "right only", (0, 0, 0): "end / lost",
}


def clean_mask(mask, min_area, min_blob=MIN_BLOB):
    """작은 점(잡음)을 지운 마스크"""
    clean = np.zeros_like(mask)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area * min_blob:
            clean[labels == i] = 255
    return clean


def bits_from_mask(mask, min_area, band=(0.0, 0.45), min_fill=0.06, min_blob=MIN_BLOB, cleaned=False):
    """마스크 -> ((왼쪽, 가운데, 오른쪽) 0/1, (띠 시작 y, 끝 y), 칸별 채움 비율)"""
    clean = mask if cleaned else clean_mask(mask, min_area, min_blob)
    h, w = clean.shape[:2]
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


def choose(bits, rule="left"):
    """있는 길 중 하나를 고른다: "left" / "center" / "right" """
    order = (0, 1, 2) if rule == "left" else (2, 1, 0)
    names = ("left", "center", "right")
    for k in order:
        if bits[k]:
            return names[k]
    return "center"


def mass_ratio(clean, band, taper):
    """먼 쪽 띠의 줄당 선 픽셀 수 / (화면 아래쪽 줄의 선 픽셀 수 x 원근으로 줄어든 비율).  1 근처 = 선 하나, 크면 옆으로 길이 붙음"""
    h = clean.shape[0]
    counts = (clean > 0).sum(axis=1)
    y1, y2 = int(h * band[0]), max(int(h * band[0]) + 1, int(h * band[1]))
    top, bottom = counts[y1:y2], counts[int(h * 0.8):]
    top, bottom = top[top > 0], bottom[bottom > 0]
    if len(top) == 0 or len(bottom) == 0:
        return 0.0
    taper = min(1.0, max(0.1, taper))
    factor = taper + (1 - taper) * (((y1 + y2) / 2) / max(1, h - 1))       # 띠 가운데 높이에서 기대하는 두께 비율
    return float(top.mean()) / (float(bottom.mean()) * factor)


def arm_err(clean, choice, min_pixels):
    """고른 길이 있는 쪽 마스크만으로 라인 중심을 다시 계산한다 -> (찾았나, err)"""
    h, w = clean.shape[:2]
    x0, x1 = {"left": (0, w // 2), "center": (w // 4, 3 * w // 4), "right": (w // 2, w)}[choice]
    ys, xs = np.nonzero(clean[:, x0:x1])
    if len(xs) < min_pixels:
        return False, 0.0
    cx = float(np.median(xs)) + x0                    # 평균 대신 중앙값: 옆으로 붙은 다른 길에 덜 끌린다
    return True, (cx - w / 2) / (w / 2)



def find_bar(mask, min_area, w):
    """마스크에서 가로선을 찾는다 -> None 또는 (위 y, 아래 y, 왼쪽 끝 x, 오른쪽 끝 x).  (마스크 안의 좌표)"""
    if mask is None or mask.shape[0] < BAR_ROWS:
        return None
    clean = clean_mask(mask, min_area)
    k = max(3, int(w * BAR_FRAC))
    bar = cv2.morphologyEx(clean, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
    rows = np.nonzero(bar.any(axis=1))[0]
    if len(rows) < BAR_ROWS:
        return None
    cols = np.nonzero(bar.any(axis=0))[0]
    return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())


def classify(far_clean, bar, stem_x, w):
    """가로선이 줄기(stem_x) 기준으로 어느 쪽으로 뻗었나 + 줄기가 위로 이어지나 -> 가능한 길 집합 {"left","right","straight"}"""
    top, bot, x0, x1 = bar
    m = SIDE_MARGIN * w
    ways = set()
    if x0 < stem_x - m:
        ways.add("left")
    if x1 > stem_x + m:
        ways.add("right")
    above = far_clean[:max(0, top - 2), int(max(0, stem_x - STRAIGHT_BAND * w)):int(min(w, stem_x + STRAIGHT_BAND * w))]
    if above.shape[0] >= 6 and float((above > 0).any(axis=1).mean()) >= STRAIGHT_FILL:
        ways.add("straight")
    return ways


def decide(ways, rule=RULE):
    """오른손 규칙으로 하나 고른다 (없으면 None)"""
    order = ("right", "straight", "left") if rule == "right" else ("left", "straight", "right")
    for d in order:
        if d in ways:
            return d
    return None


def kind_name(ways):
    w = set(ways)
    return {frozenset({"right"}): "right corner", frozenset({"left"}): "left corner",
            frozenset({"left", "right"}): "T", frozenset({"left", "right", "straight"}): "cross",
            frozenset({"right", "straight"}): "straight + right", frozenset({"left", "straight"}): "straight + left",
            frozenset({"straight"}): "straight"}.get(frozenset(w), "-")


class Junction:
    def __init__(self, rule=RULE):
        self.rule = rule
        self.phase = None             # None / "approach" / "bar" / "turn"(신호 보낸 뒤 TURN 중)
        self.plan = None              # 고른 방향 "left" / "right" / "straight" / None
        self.ways = set()
        self.t0 = 0.0
        self.cool_until = 0.0
        self.seen = 0                 # 가로선이 연속으로 보인 프레임 수
        self.far_lost_t = None
        self.stem_x = None            # 가로선이 오기 전의 near 라인 중심 x
        self.far_bar = None           # (화면 높이 기준 좌표) 화면 표시용
        self.near_bar = None
        self.history = []             # (시각, 종류, 고른 방향) - 나중에 같은 곳을 도는지 보는 데 쓴다

    @property
    def active(self):
        return self.phase is not None

    @property
    def mode(self):
        return f"{self.phase}:{self.plan}" if self.phase else None

    def idle(self, now, cooldown=False):
        """우회/유턴 중이거나 정지 중: 코너 모드를 끈다. cooldown=True 면 끝난 직후처럼 잠깐 새로 세지 않는다"""
        self.phase = None
        self.plan = None
        self.seen = 0
        self.far_lost_t = None
        if cooldown:
            self.cool_until = max(self.cool_until, now + COOLDOWN)

    def _end(self, now, why):
        print(f"[코너] 끝: {why}")
        self.idle(now, cooldown=True)

    def update(self, res, now, P, state="FOLLOW"):
        """반환: None(평소) / dict(phase, dir, bar).  state = Driver.state"""
        full = res["mask_full"]
        h, w = full.shape[:2]
        y0, far_y1 = res["y0"], res["far_y1"]
        nm_h = h - y0
        near_area = max(1, int(P.min_area * (nm_h / h) / 0.30))
        near_bar = find_bar(res["mask"], near_area, w)
        self.near_bar = None if near_bar is None else (near_bar[0] + y0, near_bar[1] + y0, near_bar[2], near_bar[3])
        if near_bar is None and res["found"]:
            self.stem_x = res["cx"]                                  # 가로선이 오기 전의 줄기 위치를 기억해 둔다
        stem_x = self.stem_x if self.stem_x is not None else w / 2

        # --- TURN 중: 새로 판단하지 않고 near 가로선이 아직 있는지만 알려 준다
        if state == "TURN":
            self.phase = "turn"
            return {"phase": "turn", "dir": self.plan, "bar": near_bar is not None}
        if self.phase == "turn":                                     # TURN 이 끝났다
            self._end(now, "회전 끝")

        # --- far 영역 가로선 보기
        far_bar, far_clean = None, None
        if far_y1:
            far = full[:far_y1]
            far_area = max(1, int(P.min_area * (far_y1 / h) / 0.30))
            far_clean = clean_mask(far, far_area)
            far_bar = find_bar(far, far_area, w)
        self.far_bar = far_bar
        mid_bar = None                                               # far 와 near 사이(겹치지 않을 때 생기는 틈)에 있는 가로선
        if far_y1 and far_y1 < y0:
            gap = full[far_y1:y0]
            mid_bar = find_bar(gap, max(1, int(P.min_area * (gap.shape[0] / h) / 0.30)), w) if gap.shape[0] >= BAR_ROWS else None

        if self.phase is None:
            if far_bar is not None or near_bar is not None or mid_bar is not None:
                self.seen += 1
            else:
                self.seen = 0
            if self.seen < STABLE_FRAMES or now < self.cool_until:
                return None
            self.phase, self.t0, self.far_lost_t = "approach", now, None
            self.plan = None
            print("[코너] 가로선이 보인다 -> 천천히")

        # --- 예고/접근 중
        if now - self.t0 > MAX_TIME:
            self._end(now, "시간 초과")
            return None

        if far_bar is not None and near_bar is None:                 # 아직 멀다: 종류를 계속 다시 본다
            self.ways = classify(far_clean, far_bar, stem_x, w)
            new = decide(self.ways, self.rule)
            if new != self.plan:
                self.plan = new
                print(f"[코너] {kind_name(self.ways)} -> {new} 으로 간다")
        if far_bar is None and near_bar is None and mid_bar is None:
            self.far_lost_t = self.far_lost_t or now
            if now - self.far_lost_t > LOST_FAR_TIME:
                self._end(now, "가로선이 사라짐")
                return None
        else:
            self.far_lost_t = None

        if near_bar is None:
            return {"phase": "approach", "dir": self.plan, "bar": False}
        # near 영역에 들어왔다: 조향 끔. 중심이 TRIGGER 높이에 닿으면 신호
        self.phase = "bar"
        centre = (near_bar[0] + near_bar[1]) / 2
        if far_bar is None and self.plan is None:                    # 멀리서 못 본 채 가까이 옴: 지금 본 가로선으로 판단
            self.ways = classify(np.zeros((1, w), np.uint8), near_bar, stem_x, w)
            self.plan = decide(self.ways, self.rule)
        if self.plan in ("left", "right") and centre >= TRIGGER * nm_h:
            self.history.append((now, kind_name(self.ways), self.plan))
            print(f"[코너] 신호! {self.plan} (#{len(self.history)})")
            return {"phase": "turn", "dir": self.plan, "bar": True}
        return {"phase": "bar", "dir": self.plan, "bar": True}
