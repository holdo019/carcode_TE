"""
motor_signal.py - 바퀴 속도를 라파로 보낼 핀 신호로 바꾼다

    to_signal(left, right)  ->  "M,AIN1,AIN2,PWMA,BIN1,BIN2,PWMB"
      left, right : -100 ~ +100  (+ 앞, - 뒤, 0 정지)
      예)  to_signal(25, 25)  ->  "M,1,0,25,0,1,25"

모터 A = 오른쪽 바퀴 (AIN1 HIGH = 앞)   <- 확인됨
모터 B = 왼쪽 바퀴  (BIN2 HIGH = 앞)    <- 확인됨
뒤로 가는 핀(AIN2, BIN1)은 아직 확인하지 않았다. 라파에서 motor_test.py 로 확인한 뒤,
다르면 아래 BACKWARD_* 표를 고친다.
"""
# ---------------- 설정 ----------------
SWAP_LR = False         # True: 왼쪽 속도가 오른쪽 바퀴(A)를 움직이게 (좌우가 바뀐 경우)
INVERT_LEFT = False     # True: 왼쪽 바퀴가 "앞"에서 뒤로 돌 때
INVERT_RIGHT = False    # True: 오른쪽 바퀴가 "앞"에서 뒤로 돌 때

# 모터 하나의 (앞 핀, 뒤 핀).  A = (AIN1, AIN2),  B = (BIN2, BIN1)
MOTOR_A = ("AIN1", "AIN2")
MOTOR_B = ("BIN2", "BIN1")
# --------------------------------------


def _one(v, pins, out):
    """v(-100~100) 한 개를 핀 값으로: 앞이면 앞 핀 HIGH, 뒤면 뒤 핀 HIGH, 힘은 |v|. 두 핀 이름을 key 로 out 에 넣는다"""
    fwd, back = pins
    v = max(-100.0, min(100.0, v))
    out[fwd] = 1 if v > 0 else 0
    out[back] = 1 if v < 0 else 0
    return round(abs(v))


def to_signal(left, right):
    if INVERT_LEFT:
        left = -left
    if INVERT_RIGHT:
        right = -right
    a_val, b_val = (left, right) if SWAP_LR else (right, left)    # 기본: A = 오른쪽, B = 왼쪽
    pin = {}
    pwma = _one(a_val, MOTOR_A, pin)
    pwmb = _one(b_val, MOTOR_B, pin)
    return f"M,{pin['AIN1']},{pin['AIN2']},{pwma},{pin['BIN1']},{pin['BIN2']},{pwmb}"