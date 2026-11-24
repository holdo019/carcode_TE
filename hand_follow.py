"""
hand_follow.py - 손을 따라가며 거리를 일정하게 유지하는 제어

    follower = HandFollower()
    left, right = follower.step(hand, image_width, HP)     # -100 ~ +100  (+ 앞, - 뒤)

  손이 화면 오른쪽에 있으면 오른쪽으로 틀고, 왼쪽이면 왼쪽으로 튼다.
  손이 기준보다 크게 보이면(가까움) 뒤로, 작게 보이면(멂) 앞으로 간다. 기준과 비슷하면 멈춘다.
  손이 안 보이면 멈춘다.
  앞뒤로 안 움직이고 방향만 틀 때는 한쪽 바퀴만 돌린다.
기준 크기(HP.target)는 원하는 거리에 손을 두고 main.py 에서 c 키로 저장한다 (hand_settings.json).
"""
from dataclasses import dataclass


@dataclass
class HandParams:
    target: int = 0          # 기준 손 크기(픽셀). 0 = 아직 저장 안 함 -> 움직이지 않음
    tol: float = 0.12        # 기준과 이 비율 이내로 비슷하면 '적당한 거리' -> 앞뒤로 안 움직임
    kp_dist: float = 1.5     # 거리 오차 -> 앞뒤 속도 세기
    kp_turn: float = 0.8     # 좌우 오차 -> 방향 틀기 세기
    turn_dead: float = 0.08  # 화면 가운데에서 이 비율 이내면 방향을 틀지 않음
    max_speed: int = 30      # 최대 속도 (PWM 0~100)
    min_speed: int = 15      # 움직일 때의 최소 속도 (너무 약하면 모터가 안 돈다)


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


class HandFollower:
    def __init__(self):
        self.cx = None       # 부드럽게 만든 손 위치와 크기
        self.size = None

    def reset(self):
        self.cx = self.size = None

    def step(self, hand, width, HP):
        """hand: HandDetector.detect() 결과 또는 None.  반환: (왼쪽, 오른쪽) -100 ~ +100"""
        if hand is None or HP.target <= 0:
            self.reset()
            return 0, 0

        a = 0.5                                              # 떨림을 줄이는 평균
        self.cx = hand["cx"] if self.cx is None else a * hand["cx"] + (1 - a) * self.cx
        self.size = hand["size"] if self.size is None else a * hand["size"] + (1 - a) * self.size

        err_x = (self.cx - width / 2) / (width / 2)          # + : 손이 오른쪽
        if abs(err_x) < HP.turn_dead:
            err_x = 0.0
        err_d = (HP.target - self.size) / HP.target          # + : 손이 작다 = 멀다 -> 앞으로
        if abs(err_d) < HP.tol:
            err_d = 0.0

        fwd = _clamp(HP.kp_dist * err_d, -1, 1) * HP.max_speed
        turn = _clamp(HP.kp_turn * err_x, -1, 1) * HP.max_speed

        if fwd == 0:
            if turn > 0:                                     # 제자리에서 오른쪽으로: 왼쪽 바퀴만
                left, right = turn, 0
            elif turn < 0:
                left, right = 0, -turn
            else:
                left, right = 0, 0
        else:
            left = _clamp(fwd + turn, -HP.max_speed, HP.max_speed)
            right = _clamp(fwd - turn, -HP.max_speed, HP.max_speed)

        def lift(v):                                         # 약하면 모터가 안 돌아서 최소 속도로 올림
            if v == 0 or abs(v) >= HP.min_speed:
                return v
            return HP.min_speed if v > 0 else -HP.min_speed
        return lift(left), lift(right)