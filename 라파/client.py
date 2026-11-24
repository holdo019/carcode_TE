"""
client.py - Raspberry Pi 4: ONE camera + TCP communication with the laptop (serv.py), as functions

Test run (sends the camera image to the laptop):   python3 client.py
Or import it from your own program:
  cam = open_camera()                 open the camera -> (grab, close)
  run(ip, cam, robot=None)            connect to the laptop, send images, receive commands
                                      (reconnects every 3 s if the connection drops, returns on Ctrl+C)
  close_camera(cam)

If a robot object is given: every image goes to robot.put(img) (robot.put(None) when sending ends)
and every command from the laptop goes to robot.on_command(bytes).
"""
import socket
import struct
import threading
import time

import cv2
import numpy as np

# ---------------- settings ----------------
SERVER_IP = "192.168.50.105"    # laptop IP"172.30.0.201""192.168.50.105"
PORT = 5000
DUMMY = False                   # True: fake images instead of the camera (for testing)
CAM_ID = 0
W, H, QUALITY, FPS = 640, 480, 80, 15
# ------------------------------------------

MAGIC, HDR = b"RC", "!2sBBII"
HDR_SIZE = struct.calcsize(HDR)
IMAGE, COMMAND, CLOSE = 1, 2, 3
LIMITS = {IMAGE: 2 * 1024 * 1024, COMMAND: 32, CLOSE: 0}


# ------------------------------------------------------------
# message send / receive
# ------------------------------------------------------------
def recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("connection closed")
        buf += chunk
    return bytes(buf)


def recv_msg(sock):
    magic, t, cam, frame_id, n = struct.unpack(HDR, recv_exact(sock, HDR_SIZE))
    if magic != MAGIC or t not in LIMITS or n > LIMITS[t] or (n == 0) != (t == CLOSE):
        raise ValueError(f"bad header: magic={magic!r} type={t} len={n}")
    return t, cam, frame_id, recv_exact(sock, n)


def send_msg(sock, lock, t, cam, frame_id, payload=b""):
    with lock:
        sock.sendall(struct.pack(HDR, MAGIC, t, cam, frame_id, len(payload)) + payload)


def abort(sock):
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass


# ------------------------------------------------------------
# camera
# ------------------------------------------------------------
def open_camera(dummy=DUMMY):
    """Open the camera -> (function that grabs one RGB image, function that closes the camera)"""
    if dummy:
        n = [0]

        def grab():
            n[0] += 1
            img = np.full((H, W, 3), (255, 120, 0), np.uint8)
            cv2.putText(img, f"DUMMY #{n[0]}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
            return img
        return grab, lambda: None

    from picamera2 import Picamera2
    cam = Picamera2(CAM_ID)
    cam.configure(cam.create_video_configuration(main={"size": (W, H), "format": "RGB888"}))
    cam.start()

    def close():
        try:
            cam.stop()
            cam.close()
        except Exception:
            pass
    return (lambda: cam.capture_array("main")), close


def close_camera(cam):
    cam[1]()


# ------------------------------------------------------------
# send images / receive commands
# ------------------------------------------------------------
def send_images(cam, sock, lock, stop, robot=None):
    frame_id = 0
    try:
        while not stop.is_set():
            t0 = time.time()
            img = cam[0]()
            if robot:
                robot.put(img)                  # hand the image to the robot (line following etc.)
            ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
            if ok:
                send_msg(sock, lock, IMAGE, 0, frame_id, jpg.tobytes())
                frame_id += 1
            time.sleep(max(0, 1 / FPS - (time.time() - t0)))
    except OSError as e:
        print(f"[SEND] failed: {e}")
    finally:
        abort(sock)


def session(sock, cam, robot=None):
    """Handle one connection. Returns True if the server asked to close, False if the connection was lost"""
    stop = threading.Event()
    lock = threading.Lock()
    th = threading.Thread(target=send_images, args=(cam, sock, lock, stop, robot), daemon=True)
    th.start()

    try:
        while True:
            t, _, _, data = recv_msg(sock)
            if t == CLOSE:
                print("[CONN] server asked to close")
                return True
            if t == COMMAND:
                if robot:
                    robot.on_command(data)
                else:
                    print(f"[CMD] {data.decode(errors='replace')}")
    except (OSError, ValueError) as e:
        print(f"[CONN] lost: {e}")
        return False
    finally:
        stop.set()
        if robot:
            robot.put(None)                     # image sending ended -> stop the robot
        abort(sock)
        th.join(timeout=3)


def run(ip, cam, robot=None):
    """Connect to the laptop and run session(). If dropped, reconnect after 3 s.
    Returns on Ctrl+C or when the server asks to close"""
    try:
        while True:
            try:
                sock = socket.create_connection((ip, PORT), timeout=5)
            except OSError as e:
                print(f"[ERROR] cannot connect to {ip}:{PORT}: {e} - retrying in 3 s")
                time.sleep(3)
                continue

            sock.settimeout(None)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"[CONN] connected to {ip}:{PORT}")

            try:
                quit_requested = session(sock, cam, robot)
            finally:
                sock.close()

            if quit_requested:
                return
            print("[CONN] reconnecting in 3 s")
            time.sleep(3)
    except KeyboardInterrupt:
        pass


# ------------------------------------------------------------
# test run: only sends the camera image
# ------------------------------------------------------------
def main():
    try:
        cam = open_camera()
        print("[CAMERA] ready" + (" (dummy)" if DUMMY else ""))
    except Exception as e:
        print(f"[ERROR] cannot open the camera: {e} (check: rpicam-hello --list-cameras, or set DUMMY = True)")
        return
    try:
        run(SERVER_IP, cam)
    finally:
        close_camera(cam)
        print("[MAIN] exited")


if __name__ == "__main__":
    main()