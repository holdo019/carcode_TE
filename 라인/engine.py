"""
engine.py - 라인 전용: 영상 한 장을 받아 바퀴 속도를 정하고 화면을 그린다 (main.py 가 쓴다)

    eng = Engine(S, sent=False, gate=None, junction=None)   # S = settings_io.new_settings()  (S.p = 라인 설정)
                                                     # gate = sign_gate.SignGate  (표지판 AI. None 이면 안 쓴다)
                                                     # junction = junction.Junction  (갈림길 처리. None 이면 안 쓴다)
    view, mask_view = eng.update(img, now, dt, go)   # 새 영상마다.  eng.wheels = (왼쪽, 오른쪽) 속도 0~100
    eng.hold(dt, stop=True)                          # 새 영상이 없을 때

라인을 따라가고, 앞이 막히면 우회한다 (판단은 linetrack.py).  손 따라가기는 handmain.py 가 따로 한다.
gate 가 있으면: 앞이 막혔을 때 AI 로 보고, 스탑 표지판이면 돌아가고(3번째 스탑이면 정지), 아니면 우회한다.
junction 이 있으면: 갈림길을 알아보고 갈 길을 하나 골라 그쪽 라인만 따라간다 (판단은 junction.py).
sent=False 이면 화면에 "NOT sent to Pi" 를 표시한다.
"""
import cv2

from junction import kind_name
from linetrack import FOLLOW, TURN, Driver, blocked_info, detect


class Engine:
    def __init__(self, S, sent=False, gate=None, junction=None):
        self.S = S
        self.sent = sent
        self.gate = gate
        self.junction = junction
        self.driver = Driver()
        self.wheels = (0.0, 0.0)

    def hold(self, dt, stop=True):
        """새 영상이 없을 때 호출: 우회 타이머가 같이 멈춘다. stop=True 이면 속도도 0 으로"""
        self.driver.hold(dt)
        if stop:
            self.wheels = (0.0, 0.0)

    def update(self, img, now, dt, go):
        P = self.S.p
        res = detect(img, P)
        ratio, ratio_l, ratio_r = blocked_info(img, P)
        found, err = res["found"], res["err"]
        corner = None
        if self.junction is not None:
            if go and self.driver.state in (FOLLOW, TURN):
                corner = self.junction.update(res, now, P, self.driver.state)   # 코너/갈림길 판단 (오른손 규칙)
                if self.junction.active:
                    ratio = ratio_l = ratio_r = 0.0                      # 가로 라인을 장애물로 착각하지 않게
            elif not go:
                self.junction.idle(now)
            else:
                self.junction.idle(now, cooldown=True)                  # 우회/유턴 중
        verdict = None
        if self.gate is not None:
            if go and self.driver.state == FOLLOW and ratio >= P.block_ratio:
                verdict = self.gate.update(img, P.win, now)       # 막혔을 때만 AI 를 돌린다
            else:
                self.gate.reset()
        if go:
            left, right = self.driver.step(now, found, err, ratio, ratio_l, ratio_r, P,
                                           ai=self.gate is not None, verdict=verdict, corner=corner)
            self.wheels = (max(0, min(100, left)), max(0, min(100, right)))
        else:
            self.driver.hold(dt)
            self.wheels = (0.0, 0.0)
        return (draw_line_view(img, P, res, ratio, self.wheels, self.driver.state, go, self.sent,
                               self.gate, self.driver.stop_count, self.junction),
                draw_mask(res))


def _text(view, lines, color):
    y = 28
    for i, t in enumerate(lines):
        cv2.putText(view, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8 if i == 0 else 0.6,
                    color if i == 0 else (255, 255, 255), 2)
        y += 28 if i == 0 else 26


def draw_line_view(img, P, res, ratio, wheels, state, go, sent, gate=None, stops=0, junction=None):
    view = img.copy()
    h, w = view.shape[:2]
    cv2.line(view, (0, res["y0"]), (w, res["y0"]), (0, 255, 255), 1)         # near 영역 위쪽 경계 (노랑)
    if res.get("far_y1"):
        cv2.line(view, (0, res["far_y1"]), (w, res["far_y1"]), (0, 140, 255), 1)   # far 영역 아래쪽 경계 (주황)
    x0, x1, y0, y1 = P.win
    cv2.rectangle(view, (int(w * x0), int(h * y0)), (int(w * x1), int(h * y1)),
                  (0, 0, 255) if ratio >= P.block_ratio else (0, 200, 0), 2)    # 막힘 창 (빨강 = 막힘)
    cv2.line(view, (w // 2, res["y0"]), (w // 2, h), (255, 255, 255), 1)       # 화면 중앙
    if res["found"]:
        cv2.line(view, (int(res["cx"]), res["y0"]), (int(res["cx"]), h), (255, 0, 255), 3)   # 라인 중심
    lines = [f"LINE  {'GO' if go else 'STOP'}  {state}",
             f"line={'yes' if res['found'] else 'LOST'} err={res['err']:+.2f} blocked={ratio:.2f}",
             f"left={wheels[0]:.0f} right={wheels[1]:.0f}" + ("" if sent else "  (NOT sent to Pi)")]
    if gate is not None:
        for x1, y1, x2, y2, label, inside in gate.boxes:                   # 표지판 AI 검출 (초록 = 가운데 박스 안)
            cv2.rectangle(view, (x1, y1), (x2, y2), (0, 255, 0) if inside else (160, 160, 160), 2)
            cv2.putText(view, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if inside else (160, 160, 160), 2)
        lines.append(f"sign AI: {gate.status()}  stops={stops}/{P.stop_limit}")
    if junction is not None:
        lines.append("corner: " + (f"{junction.mode}  ({kind_name(junction.ways)})" if junction.active else "-"))
        for bar, col in ((junction.far_bar, (0, 140, 255)), (junction.near_bar, (0, 255, 255))):
            if bar is not None:
                cv2.rectangle(view, (bar[2], bar[0]), (bar[3], bar[1]), col, 2)
    _text(view, lines, (0, 200, 0) if go else (0, 0, 255))
    return view


def draw_mask(res):
    vis = cv2.cvtColor(res["mask"], cv2.COLOR_GRAY2BGR)
    if res["chosen"] is not None:
        vis[res["chosen"]] = (0, 255, 0)
    return vis