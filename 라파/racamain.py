"""
caramain.py - run this on the Raspberry Pi 4:   python3 caramain.py

The camera image goes to the laptop (serv.py / camain.py). The laptop finds the line.
Image processing only (ROBOT = False): files needed = caramain.py  client.py
With the wheels (ROBOT = True): the laptop sends wheel speeds back and drive.py runs them -> also needs drive.py
Wheel directions and pins are at the top of drive.py (test them with motor_test.py).
"""
import client

# ---------------- settings ----------------
SERVER_IP = client.SERVER_IP   # laptop IP (the default is set at the top of client.py)
ROBOT = True                  # False: only send camera images to the laptop (no motors, drive.py not needed)
                               # True: the wheels follow the laptop signals (needs drive.py)
DUMMY = False                  # True: fake images instead of the camera (for testing)
# ------------------------------------------


def main():
    try:
        cam = client.open_camera(DUMMY)
        print("[CAMERA] ready" + (" (dummy)" if DUMMY else ""))
    except Exception as e:
        print(f"[ERROR] cannot open the camera: {e} "
              "(check: rpicam-hello --list-cameras, or set DUMMY = True)")
        return

    robot = None
    try:
        if ROBOT:
            import drive                      # only needed when the wheels are used
            robot = drive.RobotDrive()
        client.run(SERVER_IP, cam, robot)
    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        if robot:
            robot.close()       # stop the wheels + driver off
        client.close_camera(cam)
        print("[MAIN] exited")


if __name__ == "__main__":
    main()