"""
sign_gate.py - 앞이 막혔을 때 '표지판인가, 그냥 장애물인가' 를 AI 로 판단한다

    gate = SignGate("best.pt", labels=("stop1", "stop2", "stop3"))
    verdict = gate.update(img, P.win)     # 막힌 동안 새 영상마다 호출
    gate.reset()                          # 막히지 않았을 때 호출

verdict
    None        아직 모른다 (몇 프레임 더 봐야 한다)  -> 로봇은 선 채로 기다린다
    "sign"      가운데 박스(막힘 창) 안에 표지판이 있다  -> 돌아가기.  gate.label = 클래스 이름
    "obstacle"  막혔는데 표지판은 없다                   -> 우회

판단 방법: 막힘 창(가운데 네모 박스)과 겹치는 표지판 검출이 최근 WINDOW 프레임 중 CONFIRM 번 이상이면 sign,
WINDOW 프레임을 다 보도록 한 번도 없으면 obstacle. 한 프레임의 오인식이나 깜빡임에 속지 않게 하려는 것이다.
표지판 모델은 막혔을 때만 돌리므로 평소에는 노트북 부담이 없다.

필요: sign_ai.py (best.pt 를 불러오는 기존 파일)  /  pip install ultralytics
"""
from collections import Counter, deque


def overlap_with_window(box, win_px):
    """검출 박스와 막힘 창이 겹치는 정도 0~1:  겹친 면적 / (검출 박스 면적, 창 면적 중 작은 쪽).
    표지판이 창보다 작으면 '표지판이 창 안에 들어갔는가', 창을 꽉 채울 만큼 크면 '창이 표지판으로 덮였는가'"""
    x1, y1, x2, y2 = box[:4]
    wx1, wy1, wx2, wy2 = win_px
    iw = min(x2, wx2) - max(x1, wx1)
    ih = min(y2, wy2) - max(y1, wy1)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    smaller = min((x2 - x1) * (y2 - y1), (wx2 - wx1) * (wy2 - wy1))
    return inter / smaller if smaller > 0 else 0.0


class SignGate:
    def __init__(self, model_path=None, labels=("stop1", "stop2", "stop3"), conf=0.5, imgsz=640,
                 device=None, confirm=3, window=6, min_overlap=0.3, detector=None):
        if detector is None:
            from sign_ai import SignDetector                  # ultralytics 는 여기서만 불러온다
            detector = SignDetector(model_path, conf=conf, imgsz=imgsz, device=device)
            names = getattr(getattr(detector, "model", None), "names", None)
            if names is not None:
                names = list(names.values()) if isinstance(names, dict) else list(names)
                print(f"[표지판] 모델 클래스: {names}")
                if labels and not set(labels) & set(names):
                    print(f"[표지판] 경고: 찾을 표지판 이름 {tuple(labels)} 이 모델 클래스에 없다. SIGN_LABELS 를 맞추세요")
        self.det = detector
        self.labels = tuple(labels) if labels else None       # None 이면 모든 클래스를 표지판으로 본다
        self.confirm = confirm
        self.window = window
        self.min_overlap = min_overlap
        self.reset()

    def reset(self):
        self.hist = deque(maxlen=self.window)     # 최근 프레임마다 '창 안에 표지판이 있었나'
        self.seen = Counter()                     # 표지판 이름별 본 횟수
        self.boxes = []                           # 화면에 그리기 위한 (x1, y1, x2, y2, label, 창 안인가)
        self.label = None
        self.verdict = None

    def update(self, img, win):
        """win = (x0, x1, y0, y1) 화면 대비 비율 (linetrack.Params.win).  반환: None / "sign" / "obstacle" """
        h, w = img.shape[:2]
        x0, x1, y0, y1 = win
        win_px = (int(w * x0), int(h * y0), int(w * x1), int(h * y1))
        self.boxes = []
        hit = None
        for b in self.det.detect(img):
            label = b[4]
            is_sign = self.labels is None or label in self.labels
            inside = is_sign and overlap_with_window(b, win_px) >= self.min_overlap
            self.boxes.append((*b[:4], label, inside))
            if inside:
                hit = label
        self.hist.append(hit is not None)
        if hit is not None:
            self.seen[hit] += 1

        if sum(self.hist) >= self.confirm:
            self.label = self.seen.most_common(1)[0][0]
            self.verdict = "sign"
        elif len(self.hist) == self.window and sum(self.hist) == 0:
            self.label = None
            self.verdict = "obstacle"
        else:
            self.verdict = None
        return self.verdict

    def status(self):
        return f"hits={sum(self.hist)}/{len(self.hist)} verdict={self.verdict or 'wait'}" + (f"({self.label})" if self.label else "")