"""
serv.py - 노트북 TCP 서버 (카메라 1대용)
라파4(client.py)가 보내는 이미지를 받고, 라파로 명령을 보낸다.

단독 실행(영상 확인):  python3 serv.py        종료: 영상 창에서 q
다른 프로그램에서 사용:
    import serv
    serv.start()                  서버를 열고 바로 반환
    img, frame_id = serv.get_frame()   최신 이미지 (아직 없으면 (None, -1))
    serv.send_command("GO,-,-")   라파로 명령 문자열 전송 (32바이트 이하)
    serv.stop()
"""
import socket
import struct
import threading
import time

import cv2
import numpy as np

# ==============================
# 기본 설정
# ==============================
PORT = 5000

MAGIC = b"RC"
HDR = "!2sBBII"
HDR_SIZE = struct.calcsize(HDR)

IMAGE = 1
COMMAND = 2
CLOSE = 3

LIMITS = {IMAGE: 2 * 1024 * 1024, COMMAND: 32, CLOSE: 0}

# ==============================
# 전역 변수
# ==============================
conn = None
running = False
server_sock = None

send_lock = threading.Lock()
frame_lock = threading.Lock()

frame = None        # 최신 이미지
latest_id = -1


# ==============================
# 메시지 수신 / 전송
# ==============================
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
        raise ValueError("잘못된 메시지")
    return t, cam, frame_id, recv_exact(sock, n)


def send_msg(sock, t, payload=b""):
    with send_lock:
        sock.sendall(struct.pack(HDR, MAGIC, t, 0, 0, len(payload)) + payload)


def send_command(text):
    """라파로 명령 문자열을 보낸다. 연결이 없으면 False"""
    c = conn
    if c is None:
        return False
    try:
        send_msg(c, COMMAND, text.encode()[:32])
        return True
    except OSError:
        return False


# ==============================
# 클라이언트 연결 및 이미지 수신
# ==============================
def serve(server):
    global conn, frame, latest_id

    while True:
        try:
            c, addr = server.accept()
        except OSError:
            return

        print(f"[연결] {addr[0]}:{addr[1]}")
        c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        conn = c

        try:
            while True:
                t, cam, frame_id, data = recv_msg(c)

                if t == CLOSE:
                    print("[연결] 클라이언트가 종료를 알림")
                    break
                if t != IMAGE:
                    continue

                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    print("[경고] 이미지 디코딩 실패")
                    continue

                with frame_lock:
                    frame = img
                    latest_id = frame_id

        except (OSError, ValueError) as e:
            if running:
                print(f"[서버 오류] {e}")

        finally:
            conn = None
            with frame_lock:
                frame = None
                latest_id = -1
            c.close()
            if running:
                print("[연결 종료] 다시 접속 대기 중")


# ==============================
# 외부(main.py 등)에서 쓰는 함수
# ==============================
def get_frame():
    """(이미지, frame_id) 반환. 아직 없으면 (None, -1)."""
    with frame_lock:
        return frame, latest_id


def start():
    """서버를 백그라운드 스레드로 열고 바로 반환."""
    global server_sock, running

    running = True
    server_sock = socket.socket()
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(("0.0.0.0", PORT))
    server_sock.listen(1)
    print(f"[서버 열림] 포트 {PORT} 에서 대기 중")

    threading.Thread(target=serve, args=(server_sock,), daemon=True).start()


def stop():
    global running

    running = False
    c = conn
    if c is not None:
        try:
            send_msg(c, CLOSE)
            time.sleep(0.3)
            c.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
    if server_sock is not None:
        server_sock.close()
    print("[서버 닫힘]")


# ==============================
# 단독 실행(테스트용): 영상만 보여줌
# ==============================
def main():
    start()
    try:
        while True:
            img, _ = get_frame()
            if img is not None:
                cv2.imshow("CAMERA", img)
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()