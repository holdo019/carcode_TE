"""
drive.py - Raspberry Pi 4: puts the motor signals from the laptop on the TB6612 pins (RPi.GPIO)

The Pi does NOT decide anything. The laptop sends the pin values, this file only sets the pins.
  laptop signal   "M,AIN1,AIN2,PWMA,BIN1,BIN2,PWMB"      example: "M,1,0,40,0,1,40"
                    AIN1 AIN2 BIN1 BIN2 : 1 = HIGH, 0 = LOW
                    PWMA PWMB           : power 0 ~ 100 (PWM duty)
                  "STOP,-,-"                              all pins LOW, power 0
STBY is held HIGH while this program runs.
Safety: if no signal arrives for TIMEOUT_S seconds (or the video connection ends), everything is turned off.
Which pin means forward / left / right is decided on the laptop. The only thing to edit here is the pin numbers.
"""
import threading
import time

import RPi.GPIO as GPIO

# ---------------- settings: pin numbers (GPIO numbers) ----------------
STBY = 22
PWMA, AIN1, AIN2 = 18, 17, 27     # motor A
PWMB, BIN1, BIN2 = 12, 5, 6       # motor B
FREQ = 1000                       # PWM frequency in Hz
TIMEOUT_S = 0.5                   # no signal for this long -> everything off
DEBUG = True                      # print the pin values when they change
# ----------------------------------------------------------------------

DIR_PINS = (AIN1, AIN2, BIN1, BIN2)


class Pins:
    """The TB6612 pins. apply(a1, a2, pa, b1, b2, pb) sets all six; off() turns everything off"""

    def __init__(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        for p in (STBY, PWMA, PWMB) + DIR_PINS:
            GPIO.setup(p, GPIO.OUT, initial=GPIO.LOW)
        self.pwma = GPIO.PWM(PWMA, FREQ)
        self.pwmb = GPIO.PWM(PWMB, FREQ)
        self.pwma.start(0)
        self.pwmb.start(0)
        GPIO.output(STBY, GPIO.HIGH)

    def apply(self, a1, a2, pa, b1, b2, pb):
        for pin, v in zip(DIR_PINS, (a1, a2, b1, b2)):
            GPIO.output(pin, GPIO.HIGH if v else GPIO.LOW)
        self.pwma.ChangeDutyCycle(max(0, min(100, pa)))
        self.pwmb.ChangeDutyCycle(max(0, min(100, pb)))

    def off(self):
        self.apply(0, 0, 0, 0, 0, 0)

    def close(self):
        try:
            self.off()
            GPIO.output(STBY, GPIO.LOW)
            self.pwma.stop()
            self.pwmb.stop()
            del self.pwma, self.pwmb          # delete the PWM objects BEFORE cleanup (no error message at exit)
            GPIO.cleanup()
        except Exception:
            pass


def parse(payload):
    """b"M,1,0,40,0,1,40" -> (1, 0, 40, 0, 1, 40).  b"STOP,-,-" -> all zeros.  Anything else -> ValueError"""
    p = payload.decode(errors="replace").split(",")
    if p[0] == "STOP":
        return (0, 0, 0, 0, 0, 0)
    if p[0] != "M" or len(p) != 7:
        raise ValueError(payload)
    a1, a2, b1, b2 = int(p[1]), int(p[2]), int(p[4]), int(p[5])
    pa, pb = float(p[3]), float(p[6])
    if any(v not in (0, 1) for v in (a1, a2, b1, b2)) or not (0 <= pa <= 100) or not (0 <= pb <= 100):
        raise ValueError(payload)
    return (a1, a2, pa, b1, b2, pb)


class RobotDrive:
    """client.py talks to it: put(image) for every camera image, on_command(bytes) for every laptop signal"""

    def __init__(self, timeout_s=TIMEOUT_S):
        self.pins = Pins()
        self.timeout_s = timeout_s
        self._lock = threading.Lock()
        self._t = 0.0                  # time of the last laptop signal
        self._active = False           # True while something is switched on
        self._shown = None
        self._shown_t = 0.0
        self._run = True
        self._th = threading.Thread(target=self._watchdog, daemon=True)
        self._th.start()
        print(f"[DRIVE] ready: waiting for laptop signals, everything off after {timeout_s}s without one")

    def put(self, img):
        """The Pi does not look at the images (the laptop does). None = the video connection ended -> off"""
        if img is None:
            self._set((0, 0, 0, 0, 0, 0), "video lost")

    def on_command(self, payload):
        try:
            values = parse(payload)
        except (ValueError, IndexError):
            print(f"[DRIVE] ignored unknown signal: {payload!r}")
            return
        self._set(values, "laptop", from_laptop=True)

    def _set(self, values, why, from_laptop=False):
        with self._lock:
            self.pins.apply(*values)
            self._active = bool(values[0] or values[1] or values[3] or values[4]) or values[2] > 0 or values[5] > 0
            if from_laptop:
                self._t = time.monotonic()
        self._show(values, why)

    def _show(self, v, why):
        """Print only when the values change (at most 4 times a second, an all-off is always printed)"""
        key = (v[0], v[1], round(v[2]), v[3], v[4], round(v[5]))
        now = time.monotonic()
        off = not any(key)
        if DEBUG and key != self._shown and (off or now - self._shown_t > 0.25):
            print(f"[DRIVE] AIN1={key[0]} AIN2={key[1]} PWMA={key[2]} | BIN1={key[3]} BIN2={key[4]} PWMB={key[5]}"
                  + (f"   (off: {why})" if off else ""))
            self._shown, self._shown_t = key, now

    def _watchdog(self):
        while self._run:
            time.sleep(0.05)
            with self._lock:
                late = self._active and time.monotonic() - self._t > self.timeout_s
            if late:
                self._set((0, 0, 0, 0, 0, 0), "no signal from the laptop")

    def close(self):
        self._run = False
        self._th.join(timeout=1)
        self.pins.close()
        print("[DRIVE] closed")