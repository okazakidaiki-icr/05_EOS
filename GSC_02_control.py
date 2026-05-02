#%%
import time
import re
import serial
from serial.tools import list_ports

REFRESH_SERIAL_READ = 1e-1
CRLF = "\r\n"


class GSC02control:
    def __init__(self, port: str, baudrate: int = 9600):
        self.ser = serial.Serial()
        self.ser.baudrate = baudrate
        self.ser.port = port
        self.ser.rtscts = True
        self.ser.dsrdtr = False
        self.ser.timeout = 2 * REFRESH_SERIAL_READ

    # --- connection helpers ---
    @staticmethod
    def list_devices():
        ports = list_ports.comports()
        devices = [info.device for info in ports]
        print(devices)
        if not devices:
            print("エラー: ポートが見つかりません。接続を確認してください。")
        return devices

    def connect(self) -> bool:
        try:
            if not self.ser.is_open:
                self.ser.open()
            return True
        except Exception as e:
            print(f"シリアルポート接続エラー: {e}")
            return False

    def disconnect(self):
        try:
            if self.ser.is_open:
                self.ser.close()
        except Exception:
            pass

    # --- operations ---
    class DeviceOperation:
        def __init__(self, ser: serial.Serial):
            self.ser = ser

        # ---- low-level helpers ----
        def _ensure_open(self):
            if not getattr(self.ser, "is_open", False):
                self.ser.open()

        def _close(self):
            try:
                self.ser.close()
            except Exception:
                pass

        # ---- status ----
        def is_finished(self, *, timeout_s: float = 60.0, poll_s: float = 0.1) -> bool:
            """Poll '!:' until controller returns 'R' (READY)."""
            self._ensure_open()
            self.ser.reset_input_buffer()

            t0 = time.time()
            while True:
                self.ser.write(f"!:{CRLF}".encode())
                resp = self.ser.readline().decode(errors="ignore").strip()

                if resp == "R":
                    return True
                if resp == "B":
                    if (time.time() - t0) >= timeout_s:
                        return False
                    time.sleep(poll_s)
                    continue

                # unexpected -> keep polling but respect timeout
                if (time.time() - t0) >= timeout_s:
                    return False
                time.sleep(poll_s)

        def version_confirmation(self) -> str:
            """Read ROM version with '?:V'."""
            self._ensure_open()
            self.ser.reset_input_buffer()
            self.ser.write(f"?:V{CRLF}".encode())
            resp = self.ser.readline().decode(errors="ignore").strip()
            print(f"GSC-02 ROM Version: {resp}")
            self._close()
            return resp

        # ---- motion ----
        def go_to(self, axis: int, val: int, pol: str = "+", *, timeout_s: float = 120.0):
            """Move and wait until completion (M: ... then G:)."""
            axis_str = str(int(axis))
            val_str = str(int(val))
            pol_str = "+" if pol != "-" else "-"

            self._ensure_open()
            if not self.is_finished(timeout_s=timeout_s):
                raise TimeoutError("Controller stayed Busy before move.")

            # Set relative pulse move (M:) then drive (G:)
            self.ser.write(f"M:{axis_str}{pol_str}P{val_str}{CRLF}".encode())

            if not self.is_finished(timeout_s=timeout_s):
                raise TimeoutError("Controller stayed Busy after M: command.")

            self.ser.write(f"G:{CRLF}".encode())

            # Wait motion complete before returning
            if not self.is_finished(timeout_s=timeout_s):
                raise TimeoutError("Motion did not complete within timeout.")

            self._close()

        # ---- position query (NEW) ----
        def query_positions_raw(self, *, timeout_s: float = 5.0) -> str:
            """Send Q: and return raw response line."""
            self._ensure_open()
            if not self.is_finished(timeout_s=timeout_s):
                raise TimeoutError("Controller stayed Busy; cannot query positions.")
            self.ser.write(f"Q:{CRLF}".encode())
            resp = self.ser.readline().decode(errors="ignore").strip()
            self._close()
            return resp

        def get_positions(self, *, timeout_s: float = 5.0) -> list[int]:
            resp = self.query_positions_raw(timeout_s=timeout_s)
            parts = [p.strip() for p in re.split(r"[,，]", resp) if p.strip()]
            pos: list[int] = []
            for s in parts:
                t = s.replace(" ", "")
                if re.fullmatch(r"[+-]?\d+", t):
                    pos.append(int(t))
                else:
                    break
            return pos

        def get_position(self, axis: int = 1, *, timeout_s: float = 5.0) -> int:
            pos = self.get_positions(timeout_s=timeout_s)
            if axis < 1 or axis > len(pos):
                raise IndexError(f"axis={axis} out of range (available={len(pos)})")
            return pos[axis - 1]

if __name__ == '__main__':
    GSC02_control = GSC02control("COM8", 9600)
    GSC02_control.list_devices()
    if GSC02_control.connect():
        device_op = GSC02_control.DeviceOperation(GSC02_control.ser)
        device_op.version_confirmation()  # Example of moving to 500 nm wavelength
        device_op.set_velocity(vS=1000, vF=1000, vR=10)
        device_op.go_to(axis=1, val=1000, pol='+')
        GSC02_control.disconnect()
        pass
# %%
