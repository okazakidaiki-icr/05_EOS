#%%
import serial
import time
import GSC_02_control
import numpy as np
from PyDAQmx import Task, DAQError
from PyDAQmx.DAQmxConstants import (
    DAQmx_Val_Diff, DAQmx_Val_RSE, DAQmx_Val_NRSE,
    DAQmx_Val_Volts, DAQmx_Val_Rising,
    DAQmx_Val_ContSamps, DAQmx_Val_GroupByChannel
)
from ctypes import byref, c_int32

# ==== NI-DAQ 設定（環境に合わせて変更）====
DEV_CH     = b"Dev1/ai0"    # 例：b"Dev1/ai0"（多chは b"Dev1/ai0:3"）
CFG        = DAQmx_Val_Diff # 配線：Diff/RSE/NRSE
V_MINMAX   = (-10.0, 10.0)  # 入力レンジ [V]
RATE       = 20000.0        # サンプリング周波数 [Hz]
CHUNK      = 2000           # 1回で各chから読む点数（=0.1 s @ 20 kS/s）
DRIVER_BUF = int(RATE)      # 内部バッファ（目安1秒分）
TIMEOUT    = 0.5            # 読み取りタイムアウト [s]
SAVE_BIN   = None           # 生バイナリ保存したい場合は "stream_f64le.bin"
# =========================================

CRLF = '\r\n'

class scanner:
    def __init__(self, device_op):
        self.device_op = device_op  # GSC02control.DeviceOperation のインスタンス

        # ---- DAQ 初期化（ContSamps 前提）----
        self.task = Task()
        self.read = c_int32()
        self._outfile = None

        # チャンネル数の推定（"ai0:3" → 4ch，単一なら1）
        self.NCH = 1
        try:
            if b":" in DEV_CH:
                tail = DEV_CH.split(b"/")[-1]   # b"ai0:3"
                rng  = tail.replace(b"ai", b"") # b"0:3"
                L, R = map(int, rng.split(b":"))
                self.NCH = R - L + 1
        except Exception:
            self.NCH = 1

        # 受け皿バッファ（総要素数 = CHUNK * NCH）
        self.buf = np.zeros(CHUNK * self.NCH, dtype=np.float64)

        # 物理チャネル作成（bytes で渡す）
        self.task.CreateAIVoltageChan(
            DEV_CH, b"", CFG, V_MINMAX[0], V_MINMAX[1], DAQmx_Val_Volts, None
        )
        # 連続サンプル
        self.task.CfgSampClkTiming(
            b"", RATE, DAQmx_Val_Rising, DAQmx_Val_ContSamps, DRIVER_BUF
        )

    # ---- 即時ステータス問い合わせ（従来のまま）----
    def _query_status_once(self):
        ser = self.device_op.ser
        try:
            if not ser.is_open:
                ser.open()
            ser.reset_input_buffer()
            ser.write(f'!:{CRLF}'.encode())
            resp = ser.readline().decode(errors='ignore').strip()
            if resp == 'R':   # Ready
                return True
            elif resp == 'B': # Busy
                return False
            else:
                return None
        except Exception:
            return None

    # ---- DAQ 開始／停止（追加）----
    def _daq_start(self):
        if SAVE_BIN and self._outfile is None:
            self._outfile = open(SAVE_BIN, "ab")
        self.task.StartTask()

    def _daq_stop(self):
        try:
            self.task.StopTask()
            self.task.ClearTask()
        except Exception:
            pass
        if self._outfile:
            self._outfile.close()
            self._outfile = None

    # ---- 1チャンク読み出し（追加）----
    def _read_chunk(self):
        # 受け皿サイズは CHUNK * NCH
        need = CHUNK * self.NCH
        if self.buf.size != need:
            self.buf = np.zeros(need, dtype=np.float64)

        self.task.ReadAnalogF64(
            CHUNK, TIMEOUT, DAQmx_Val_GroupByChannel,
            self.buf, need, byref(self.read), None
        )
        n = int(self.read.value)  # 各chで実際に読めたサンプル数
        if n <= 0:
            return (np.empty((0,), np.float64) if self.NCH == 1
                    else np.empty((self.NCH, 0), np.float64)), 0

        if self.NCH == 1:
            block = self.buf[:n].copy()  # shape = (n,)
        else:
            total = n * self.NCH
            block = self.buf[:total].reshape(self.NCH, n).copy()  # shape=(NCH,n)

        if self._outfile is not None:
            self._outfile.write((block if self.NCH == 1 else block.ravel()).tobytes())
        return block, n

    # ==== ここを実データ取得に置換 ====
    def getdata(self, tag=None):
        """
        装置（NI-DAQ）から 1 チャンク分の波形を取得して返す．
        戻り値: 単一chなら shape=(n,), 多chなら shape=(NCH, n)
        """
        block, n = self._read_chunk()
        if self.NCH == 1:
            print(f"[getdata] n={n}，mean={float(block.mean()) if n>0 else np.nan:+.4f} V")
        else:
            means = ", ".join(f"{float(m):+.3f}" for m in (block.mean(axis=1) if n>0 else np.zeros(self.NCH)))
            print(f"[getdata] n/ch={n}，ch_means=[{means}] V")
        return block
    # ===============================

    # ---- 既存：点取りスキャン（最小変更：DAQ開始/停止を付与）----
    def scan(self, steps, stepsize):
        self._daq_start()
        try:
            for i in range(steps):
                self.device_op.go_to(axis=1, val=stepsize, pol='+')
                # 到達待ちしながら読み続ける（バッファ溢れ防止）
                while True:
                    _ = self.getdata(i)  # 実波形
                    st = self._query_status_once()
                    if st is True:
                        break
        finally:
            self._daq_stop()

    # ---- 既存：移動中に一定周期でデータ取得 → 常時読み続けてポーリング間引き ----
    def scan_while_moving(self, interval, move_distance):
        start_time = time.time()
        self._daq_start()
        total = 0
        try:
            self.device_op.go_to(axis=1, val=move_distance, pol='-')
            print(f"move start latency = {time.time()-start_time:.3f}s")

            last_poll = 0.0
            while True:
                # 常時読み（短いタイムアウトで 1 チャンクずつ）
                block, n = self._read_chunk()
                total += n

                # interval ごとにだけ Busy/Ready を問い合わせ
                now = time.time() - start_time
                if now - last_poll >= interval:
                    st = self._query_status_once()
                    if st is True:
                        break
                    print(f"is_finished=False, elapsed {now:.2f}s, total={total} samples")
                    last_poll = now

            # 必要なら残りを少し読んでから終了
            print(f"finished, total={total} samples")
        except KeyboardInterrupt:
            print("\nStopped by user.")
        except DAQError as e:
            print(f"DAQmx Error: {e}")
        finally:
            self._daq_stop()

if __name__ == '__main__':
    GSC02_control = GSC_02_control.GSC02control("COM8", 9600)
    GSC02_control.list_devices()
    if GSC02_control.connect():
        device_op = GSC02_control.DeviceOperation(GSC02_control.ser)
        device_op.version_confirmation()
        device_op.set_velocity(vS=1000, vF=1000, vR=1000)
        scanner = scanner(device_op)
        # scanner.scan(3, 50)
        scanner.scan_while_moving(0.5, 200)
        GSC02_control.disconnect()
        pass
# %%
