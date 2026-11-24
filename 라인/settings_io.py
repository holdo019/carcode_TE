"""
settings_io.py - main.py 가 line_settings.json 을 불러오는 곳 (라인 전용, 읽기만 한다)

  line_settings.json   camain.py 가 슬라이더로 맞춰 저장한 파일

    S = new_settings()          # S.p = 라인 설정 (linetrack.Params)
    load_settings(S)            # 파일을 읽어 S 에 넣는다. 파일이 있고 읽으면 True
"""
import json
import os
from types import SimpleNamespace

from linetrack import Params

HERE = os.path.dirname(os.path.abspath(__file__))                 # 이 파일과 같은 폴더
SETTINGS_FILE = os.path.join(HERE, "line_settings.json")


def _set(name, conv=lambda v: v):
    return lambda S, v: setattr(S.p, name, conv(v))


# (이름, 최대값, 현재값 읽기, 값 넣기).  JSON 의 키가 이 이름이다.  camain.py 의 슬라이더와 같은 이름
LINE_SLIDERS = [
    ("SPEED",          100, lambda S: S.p.speed,                  _set("speed")),
    ("KP x10",          40, lambda S: int(S.p.kp * 10),           _set("kp", lambda v: v / 10)),
    ("ROI top %",       95, lambda S: int(S.p.roi_top * 100),     _set("roi_top", lambda v: v / 100)),
    ("FAR bottom %",   95, lambda S: int(S.p.far_bottom * 100),  _set("far_bottom", lambda v: v / 100)),
    ("THRESH(0=auto)", 255, lambda S: S.p.thresh,                 _set("thresh")),
    ("MIN AREA",      3000, lambda S: S.p.min_area,               _set("min_area")),
    ("DARK LINE",        1, lambda S: int(S.p.dark_line),         _set("dark_line", bool)),
    ("COLOR DIFF",     255, lambda S: S.p.color_diff,             _set("color_diff")),
    ("BLOCK %",        100, lambda S: int(S.p.block_ratio * 100), _set("block_ratio", lambda v: v / 100)),
    ("AVOID",            1, lambda S: int(S.p.avoid),             _set("avoid", bool)),
    ("T_OUT x0.1s",     50, lambda S: int(S.p.t_out * 10),        _set("t_out", lambda v: v / 10)),
    ("T_PASS x0.1s",    50, lambda S: int(S.p.t_pass * 10),       _set("t_pass", lambda v: v / 10)),
    ("T_BACK x0.1s",    50, lambda S: int(S.p.t_back * 10),       _set("t_back", lambda v: v / 10)),
    ("WIDTH px(0=off)", 300, lambda S: S.p.width,                 _set("width")),
    ("WIDTH TOL %",     90, lambda S: int(S.p.width_tol * 100),   _set("width_tol", lambda v: v / 100)),
    ("TAPER %",        100, lambda S: int(S.p.taper * 100),       _set("taper", lambda v: v / 100)),
]


def new_settings():
    return SimpleNamespace(p=Params())


def load_settings(S, path=SETTINGS_FILE):
    """line_settings.json 을 읽어 S 에 넣는다. 없거나 읽을 수 없으면 False (기본값 그대로)"""
    if not os.path.exists(path):
        return False
    try:
        with open(path, encoding="utf-8") as f:
            saved = json.load(f)
        for name, _, _, setter in LINE_SLIDERS:
            if name in saved:
                setter(S, int(saved[name]))
        print(f"[설정] 불러옴: {path}")
        return True
    except (OSError, ValueError) as e:
        print(f"[설정] {path} 불러오기 실패: {e}")
        return False