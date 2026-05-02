#%%
# Standard library
import datetime
import re
import sys
from pathlib import Path
import threading
import time
import uuid

# Third-party
import numpy as np
import pyqtgraph as pg
from scipy.signal import find_peaks  # noqa: F401  (kept for future use)

# PyQt5
from PyQt5 import QtGui, uic
from PyQt5.QtCore import (
    QObject,
    QRunnable,
    QSignalBlocker,
    QThreadPool,
    QTimer,
    Qt,
    pyqtSignal,
    pyqtSlot,
)
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow

# Local
try:
    import GSC_02_control
except ImportError:
    GSC_02_control = None


# ---- Constants / Globals ----
DELTA = "\u0394"       # Δ (U+0394)
LAMBDA0 = "\u03BB"     # λ (U+03BB)

WAIT_TIME_S = 0.02
TIMEOUT_S = 5.0
FONT_NAME = 'Yu Gothic UI'

VISA_LOCK = threading.Lock()
BG_SIGNAL = None
BG_CONTINUUM = None

# Spectrum conversion constants
C_M_S = 299_792_458.0
WL_MIN_UM = 0.1
WL_MAX_UM = 500.0


# --- Signals / Workers ---
class WorkerSignals(QObject):
    """Signals emitted by worker threads."""
    data = pyqtSignal(tuple)  # (worker_id, x, y)


class BaseWorker(QRunnable):
    """Generic QRunnable that repeatedly calls data_func() and emits results."""

    def __init__(self, data_func):
        super().__init__()
        self.worker_id = uuid.uuid4().hex
        self._running = True
        self._terminate = False
        self.data_func = data_func
        self.signals = WorkerSignals()

    def pause(self):
        self._running = False

    def resume(self):
        self._running = True

    def stop(self):
        self._terminate = True

    @staticmethod
    def _sanitize_xy(x, y):
        """Return (x,y) as float ndarrays if safe for plotting; else (None,None)."""
        if x is None or y is None:
            return None, None
        try:
            x = np.asarray(x, dtype=float).ravel()
            y = np.asarray(y, dtype=float).ravel()
        except Exception:
            return None, None
        if x.size == 0 or y.size == 0 or x.size != y.size:
            return None, None
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            return None, None
        return x, y

    @pyqtSlot()
    def run(self):
        while not self._terminate:
            if self._running:
                try:
                    x, y = self.data_func()
                    x, y = self._sanitize_xy(x, y)
                    if x is not None:
                        self.signals.data.emit((self.worker_id, x, y))
                except Exception as e:
                    print(f"[ERROR] Worker {self.worker_id} iteration error: {e}")
            time.sleep(0.01)

class TaskSignals(QObject):
    finished = pyqtSignal(object)  # result
    error = pyqtSignal(str)


class OneShotWorker(QRunnable):
    """Run a single function in threadpool (non-blocking for GUI)."""

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = TaskSignals()

    @pyqtSlot()
    def run(self):
        try:
            res = self.fn(*self.args, **self.kwargs)
            self.signals.finished.emit(res)
        except Exception as e:
            self.signals.error.emit(str(e))


# --- Generators ---
class BaseSignalGenerator:
    """Time-series generator storing all samples and returning plot-ready arrays."""

    def __init__(self):
        self.x = []  # all x
        self.y = []  # all y
        self.t_start = time.time()

    def __call__(self):
        raise NotImplementedError

    def get_plot_data(self, max_points: int = 65535):
        if len(self.x) > max_points:
            x_plot = self.x[-max_points:]
            y_plot = self.y[-max_points:]
        else:
            x_plot = self.x
            y_plot = self.y
        return np.asarray(x_plot, dtype=float), np.asarray(y_plot, dtype=float)

    def get_recent_stats(self, n: int = 10) -> float:
        if len(self.y) < n:
            return 0.0
        return float(np.mean(self.y[-n:]))


class BaseSpectrumGenerator:
    def __init__(self):
        self.latest_x = None
        self.latest_y = None

    def __call__(self):
        raise NotImplementedError


class DAQ_Ch1(BaseSignalGenerator):
    def __call__(self):
        time.sleep(1)
        # TODO: 実データ取得に置換
        # デモ動作用に1点ずつ追加する。ここをDAQ読み出しに差し替える。
        self.x.append(time.time() - self.t_start)
        self.y.append(float(np.random.normal(loc=0.0, scale=1.0)))
        return self.get_plot_data()


# --- Main Window ---
class MainWindow(QMainWindow):
    def __init__(self, *, device_op=None, gsc02=None):
        super().__init__()
        self.device_op = device_op   # ★追加
        self.gsc02 = gsc02           # ★追加（終了時disconnect等に使える）

        self.STAGE_AXIS = 1          # ★必要ならUI化。とりあえず軸1固定
        self.STAGE_SCALE = 1.0       # ★LineEdit_Targetの単位→装置P値 変換係数（必要なら変更）

        # FROGコードの命名規則に合わせて初期化メソッドを並べる。
        self.initThreads()
        self.initUI()
        self.initTimersAndValidators()
        self.connectSignals()
        self.setupPlots()
        self.initWorkers()
        self.setupWorkers()
        self.initMeasureSystem()

        self.workers_started = False
        self.dlg1.show()

    # =========================================================
    # Initialization
    # =========================================================
    def initUI(self):
        ui_path = Path(__file__).with_name("EOS_v0.ui")
        if not ui_path.exists():
            ui_path = Path("EOS_v0.ui")
        self.dlg1 = uic.loadUi(str(ui_path))

        # defaults
        defaults = {"LineEdit_Folders": r"C:\Users\owner\Desktop\Log"}
        for widget_name, value in defaults.items():
            w = getattr(self.dlg1, widget_name, None)
            if w is not None:
                w.setText(value)


    def initThreads(self):
        self.threadpool = QThreadPool()
        self.lines_DAQ_Ch1 = {}
        self.lines_DAQ_Ch1_mag = {}

    def initTimersAndValidators(self):
        # Debounce timers
        self._derived_timer = QTimer(self)
        self._derived_timer.setSingleShot(True)
        self._derived_timer.timeout.connect(self.updateDerivedParamsSafe)

        self._magnify_timer = QTimer(self)
        self._magnify_timer.setSingleShot(True)
        self._magnify_timer.timeout.connect(self.applyMagnifyXRange)

        self._wl_timer = QTimer(self)
        self._wl_timer.setSingleShot(True)
        self._wl_timer.timeout.connect(self.applyWLXRange)

        # Validators (allow negatives)
        v = QDoubleValidator(-1e9, 1e9, 6, self)
        v.setNotation(QDoubleValidator.StandardNotation)
        for name in (
            "LineEdit_Start", "LineEdit_Stop",
            "LineEdit_Start_Plot", "LineEdit_Stop_Plot",
            "LineEdit_WL_Start", "LineEdit_WL_Stop",
        ):
            w = getattr(self.dlg1, name, None)
            if w is not None:
                w.setValidator(v)

    def setupPlots(self):
        # Time plots
        for view_name in ("graphicsView_Time", "graphicsView_Time_Magnify"):
            view = getattr(self.dlg1, view_name, None)
            if view is not None:
                self.configurePlotTime(view.plotItem)

        # Frequency / Wavelength plots
        view_f = getattr(self.dlg1, "graphicsView_Frequency", None)
        if view_f is not None:
            self.configurePlotFrequency(view_f.plotItem)

        view_w = getattr(self.dlg1, "graphicsView_Wavelength", None)
        if view_w is not None:
            self.configurePlotWavelength(view_w.plotItem)

        # apply initial ranges if valid
        self.applyMagnifyXRange()
        self.applyWLXRange()

    def initWorkers(self):
        # worker map (extend later)
        self.worker_map = {"Temperature": (DAQ_Ch1, self.receiveDataTime)}

    def connectSignals(self):
        # Link label
        if hasattr(self.dlg1, "label_link_bug"):
            self.dlg1.label_link_bug.setText(
                '<a href="https://docs.google.com/forms/d/e/1FAIpQLSfjkT5eLgykSWo5wjoyUS-oBcDIWYgX9QkiCsB_rvq8Tn-DXw/viewform?usp=dialog">'
                " -->Report troubles and requirements here</a>"
            )
            self.dlg1.label_link_bug.setOpenExternalLinks(True)

        # Push Buttons
        push_buttons = {
            "Button_Close": self.closeApp,
            "Button_Folder": self.selectFolder,
            "Button_Go": self.moveStage,
            "Button_Origin": self.setOrigin,
            "Button_Measure": self.measureData,
            "Button_Update_Plot": self.updatePlot,
        }
        for btn_name, func in push_buttons.items():
            btn = getattr(self.dlg1, btn_name, None)
            if btn is not None:
                btn.clicked.connect(func)

        # Derived parameters update triggers
        for name in ("LineEdit_Start", "LineEdit_Stop"):
            le = getattr(self.dlg1, name, None)
            if le is not None:
                le.editingFinished.connect(self.scheduleDerivedUpdate)
                le.textEdited.connect(self.scheduleDerivedUpdate)

        for name in ("ComboBox_Velocity", "ComboBox_Integration", "ComboBox_Sa"):
            cb = getattr(self.dlg1, name, None)
            if cb is not None:
                cb.currentIndexChanged.connect(self.scheduleDerivedUpdate)

        # Time magnify range inputs
        for name in ("LineEdit_Start_Plot", "LineEdit_Stop_Plot"):
            le = getattr(self.dlg1, name, None)
            if le is not None:
                le.editingFinished.connect(self.scheduleMagnifyRangeUpdate)
                le.textEdited.connect(self.scheduleMagnifyRangeUpdate)

        # Wavelength range inputs
        for name in ("LineEdit_WL_Start", "LineEdit_WL_Stop"):
            le = getattr(self.dlg1, name, None)
            if le is not None:
                le.editingFinished.connect(self.scheduleWLRangeUpdate)
                le.textEdited.connect(self.scheduleWLRangeUpdate)

        # ComboBox logging
        cb = getattr(self.dlg1, "ComboBox_Velocity", None)
        if cb is not None:
            cb.currentTextChanged.connect(self.velocityChanged)

        cb = getattr(self.dlg1, "ComboBox_Integration", None)
        if cb is not None:
            cb.currentTextChanged.connect(self.integrationChanged)

        cb = getattr(self.dlg1, "ComboBox_Sa", None)
        if cb is not None:
            cb.currentTextChanged.connect(self.saChanged)

        # Monitor checkbox (UI objectName is 'checkBox')
        mon = getattr(self.dlg1, "checkBox", None)
        if mon is not None:
            mon.toggled.connect(self.monitorSwitch)

    # =========================================================
    # Plot configuration
    # =========================================================
    def configurePlot(self, plotItem, xlabel, ylabel, logY=False):
        font_family = FONT_NAME
        axis_pen = pg.mkPen(color="#fff", width=1.0)
        font_obj = QtGui.QFont(font_family)

        plotItem.setLabel("bottom", xlabel)
        plotItem.getAxis("bottom").setPen(axis_pen)
        plotItem.getAxis("bottom").tickFont = font_obj

        axis_left = plotItem.getAxis("left")
        axis_left.setLabel(text=ylabel)
        axis_left.setPen(axis_pen)
        axis_left.tickFont = font_obj

        plotItem.setLogMode(False, logY)
        plotItem.enableAutoRange("x", True)
        plotItem.enableAutoRange("y", True)
        plotItem.showGrid(x=True, y=True, alpha=0.75)

    def configurePlotTime(self, plotItem):
        self.configurePlot(
            plotItem,
            "Time (ps)",
            '<span style="color: #ff0;">EOS signal (uV)</span>',
        )
        plotItem.setYRange(-10, 10)

    def configurePlotFrequency(self, plotItem):
        self.configurePlot(
            plotItem,
            "Frequency (THz)",
            '<span style="color: #ff0;">|FFT| (a.u.)</span>',
        )

    def configurePlotWavelength(self, plotItem):
        self.configurePlot(
            plotItem,
            "Wavelength (um)",
            '<span style="color: #ff0;">|FFT| (a.u.)</span>',
        )

    # =========================================================
    # Basic UI helpers / events
    # =========================================================
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.closeApp()

    def log(self, message: str):
        te = getattr(self.dlg1, "textEdit", None)
        if te is None:
            return
        if hasattr(te, "append"):
            te.append(str(message))
        else:
            te.setText(str(message))

    def setLineEdit(self, name: str, value: str):
        le = getattr(self.dlg1, name, None)
        if le is not None and hasattr(le, "setText"):
            le.setText(value)

    def readFloatLineEdit(self, name: str, default=None):
        le = getattr(self.dlg1, name, None)
        if le is None:
            return default
        txt = le.text().strip()
        if not txt:
            return default
        try:
            return float(txt)
        except Exception:
            return default

    def updatePosFromTarget(self):
        target_le = getattr(self.dlg1, "LineEdit_Target", None)
        if target_le is None:
            self.log("LineEdit_Target not found.")
            return
        txt = target_le.text().strip()
        if not txt:
            self.log("Target is empty.")
            return
        try:
            target = float(txt)
        except ValueError:
            self.log(f"Invalid target: '{txt}'")
            return
        self.setLineEdit("LineEdit_Pos", f"{target}")
        self.log(f"{target}")

    def targetToPolVal(self, target: float) -> tuple[str, int]:
        """
        target(ユーザ入力) -> (pol, val) に変換。
        go_to は pol が '+'/'-'、val は基本的に正の整数を想定。
        """
        pol = "+" if target >= 0 else "-"
        val = int(round(abs(target) * float(self.STAGE_SCALE)))
        return pol, val


    def moveStageToTargetAsync(self, target: float):
        """device_op.go_to を threadpool で非同期実行してGUIフリーズを避ける。"""
        if self.device_op is None:
            self.log("Stage controller is not available (device_op is None).")
            return

        pol, val = self.targetToPolVal(target)

        # ボタン連打防止（任意）
        btn_go = getattr(self.dlg1, "Button_Go", None)
        if btn_go is not None:
            btn_go.setEnabled(False)

        self.log(f"[Stage] Moving to target={target} (axis={self.STAGE_AXIS}, pol={pol}, P={val})")

        def _do_move():
            # シリアル同時アクセス対策（他の計測で使い回す可能性があるのでロック）
            with VISA_LOCK:
                self.device_op.go_to(axis=self.STAGE_AXIS, val=val, pol=pol)
            return target  # “指令値”を返す（実位置読出しは次ステップで実装）

        worker = OneShotWorker(_do_move)
        worker.signals.finished.connect(self.stageMoveFinished)
        worker.signals.error.connect(self.stageMoveError)
        self.threadpool.start(worker)


    def stageMoveFinished(self, target):
        self.log(f"[Stage] Move done -> {target}")
        # いまは実位置を読めないので、表示は指令値で更新（実位置取得コマンド追加後に置換）
        self.setLineEdit("LineEdit_Pos", f"{target}")

        btn_go = getattr(self.dlg1, "Button_Go", None)
        if btn_go is not None:
            btn_go.setEnabled(True)


    def stageMoveError(self, msg: str):
        self.log(f"[Stage][ERROR] {msg}")
        btn_go = getattr(self.dlg1, "Button_Go", None)
        if btn_go is not None:
            btn_go.setEnabled(True)


    # =========================================================
    # ComboBox slots
    # =========================================================
    def velocityChanged(self, *_):
        cb = getattr(self.dlg1, "ComboBox_Velocity", None)
        if cb is not None:
            self.log(f"Velocity: {cb.currentText()}")

    def integrationChanged(self, *_):
        cb = getattr(self.dlg1, "ComboBox_Integration", None)
        if cb is not None:
            self.log(f"Integration: {cb.currentText()}")

    def saChanged(self, *_):
        cb = getattr(self.dlg1, "ComboBox_Sa", None)
        if cb is not None:
            self.log(f"Sa: {cb.currentText()}")

    # =========================================================
    # File / app lifecycle
    # =========================================================
    def selectFolder(self):
        folder = QFileDialog.getExistingDirectory()
        if not folder:
            return
        self.file_path = folder.replace("/", "\\") + "\\"

        if hasattr(self.dlg1, "textEdit"):
            self.dlg1.textEdit.setText(str(self.file_path))
            if hasattr(self.dlg1, "LineEdit_Folders"):
                self.dlg1.LineEdit_Folders.setText(str(self.file_path))
            if hasattr(self.dlg1.textEdit, "append"):
                self.dlg1.textEdit.append("Path: " + self.file_path)

    def controlWorkers(self, action: str):
        assert action in ("pause", "resume", "stop")
        for attr in dir(self):
            if attr.startswith("worker_"):
                w = getattr(self, attr, None)
                if w is not None and hasattr(w, action):
                    getattr(w, action)()

    def closeApp(self):
        self.controlWorkers("stop")
        time.sleep(0.2)

        if hasattr(self.dlg1, "textEdit") and hasattr(self.dlg1.textEdit, "append"):
            self.dlg1.textEdit.append(f"{datetime.datetime.now()} : The Application is closed")

        self.dlg1.close()
        print("Turned off")
        self.close()

    # =========================================================
    # Worker / live plotting
    # =========================================================
    def setupWorkers(self):
        """Instantiate workers and generators based on worker_map."""
        for name, (gen_cls, update_func) in self.worker_map.items():
            generator = gen_cls()
            worker = BaseWorker(generator)
            worker.signals.data.connect(update_func)

            setattr(self, f"generator_{name}", generator)
            setattr(self, f"worker_{name}", worker)

    def moveStage(self):
        self.updatePosFromTarget()

    # ★追加：ステージをターゲットへ移動（非同期）
        target = self.readFloatLineEdit("LineEdit_Target", None)
        if target is not None and np.isfinite(target):
            self.moveStageToTargetAsync(float(target))
        else:
            self.log("Invalid target value; stage move skipped.")

        if self.workers_started:
            if hasattr(self.dlg1, "textEdit") and hasattr(self.dlg1.textEdit, "append"):
                self.dlg1.textEdit.append("Workers already started.")
            return

        self.lines_DAQ_Ch1.clear()
        self.lines_DAQ_Ch1_mag.clear()

        for view_name in ("graphicsView_Time", "graphicsView_Time_Magnify"):
            view = getattr(self.dlg1, view_name, None)
            if view is not None:
                view.clear()

        for name in self.worker_map:
            worker = getattr(self, f"worker_{name}")
            self.threadpool.start(worker)

        self.workers_started = True
        if hasattr(self.dlg1, "textEdit") and hasattr(self.dlg1.textEdit, "append"):
            self.dlg1.textEdit.append("Started workers.")

    def receiveDataTime(self, data):
        worker_id, x, y = data

        # 線を描くなら2点必要（pyqtgraphのdrawLines落ち対策）
        if x is None or y is None or x.size < 2 or y.size < 2:
            return

        view_time = getattr(self.dlg1, "graphicsView_Time", None)
        if view_time is not None:
            if worker_id not in self.lines_DAQ_Ch1:
                self.lines_DAQ_Ch1[worker_id] = view_time.plot(
                    x, y, pen=pg.mkPen("y", width=2)
                )
            else:
                self.lines_DAQ_Ch1[worker_id].setData(x, y)

        # Magnify 側も同じデータを表示し、LineEdit_Start_Plot/Stop_Plot が
        # 有効な場合はその範囲だけに切り出す。
        view_mag = getattr(self.dlg1, "graphicsView_Time_Magnify", None)
        if view_mag is not None:
            pr = self.readPlotRangeOrNone()
            if pr is None:
                xm, ym = x, y
            else:
                x0, x1 = pr
                mask = (x >= x0) & (x <= x1)
                xm, ym = x[mask], y[mask]
                if xm.size < 2:
                    if worker_id in self.lines_DAQ_Ch1_mag:
                        self.lines_DAQ_Ch1_mag[worker_id].setVisible(False)
                    return

            if worker_id not in self.lines_DAQ_Ch1_mag:
                self.lines_DAQ_Ch1_mag[worker_id] = view_mag.plot(
                    xm, ym, pen=pg.mkPen("y", width=2)
                )
            else:
                self.lines_DAQ_Ch1_mag[worker_id].setVisible(True)
                self.lines_DAQ_Ch1_mag[worker_id].setData(xm, ym)

    # =========================================================
    # Range control: Time Magnify
    # =========================================================
    def scheduleMagnifyRangeUpdate(self):
        self._magnify_timer.start(250)

    def readPlotRangeOrNone(self):
        start = self.readFloatLineEdit("LineEdit_Start_Plot", None)
        stop = self.readFloatLineEdit("LineEdit_Stop_Plot", None)
        if start is None or stop is None:
            return None
        if not (np.isfinite(start) and np.isfinite(stop)):
            return None
        if start >= stop:
            return None
        return float(start), float(stop)

    def applyMagnifyXRange(self):
        view_mag = getattr(self.dlg1, "graphicsView_Time_Magnify", None)
        if view_mag is None:
            return

        plot_item = view_mag.plotItem
        pr = self.readPlotRangeOrNone()

        if pr is None:
            plot_item.enableAutoRange("x", True)
            return

        x0, x1 = pr
        plot_item.enableAutoRange("x", False)
        plot_item.setXRange(x0, x1, padding=0)

    # =========================================================
    # Range control: Wavelength
    # =========================================================
    def scheduleWLRangeUpdate(self):
        self._wl_timer.start(250)

    def readWLRangeOrNone(self):
        """
        LineEdit_WL_Start/Stop を読み、(start, stop) を返す。
        不正なら None。波長は 0.1–500 um にクランプ。
        """
        start = self.readFloatLineEdit("LineEdit_WL_Start", None)
        stop = self.readFloatLineEdit("LineEdit_WL_Stop", None)

        if start is None or stop is None:
            return None
        if not (np.isfinite(start) and np.isfinite(stop)):
            return None
        if start >= stop:
            return None

        start = max(WL_MIN_UM, float(start))
        stop = min(WL_MAX_UM, float(stop))
        if start >= stop:
            return None

        return start, stop

    def applyWLXRange(self):
        view_wl = getattr(self.dlg1, "graphicsView_Wavelength", None)
        if view_wl is None:
            return

        plot_item = view_wl.plotItem
        pr = self.readWLRangeOrNone()

        if pr is None:
            plot_item.enableAutoRange("x", True)
            return

        x0, x1 = pr
        plot_item.enableAutoRange("x", False)
        plot_item.setXRange(x0, x1, padding=0)

    # =========================================================
    # Derived parameters
    # =========================================================
    def scheduleDerivedUpdate(self):
        self._derived_timer.start(250)

    @staticmethod
    def parseTimeToSeconds(text: str) -> float:
        s = text.strip().lower()
        m = re.match(r"^\s*([0-9]*\.?[0-9]+)\s*(us|ms|s)\s*$", s)
        if not m:
            raise ValueError(f"bad time format: {text}")
        val = float(m.group(1))
        unit = m.group(2)
        return val * {"us": 1e-6, "ms": 1e-3, "s": 1.0}[unit]

    def updateDerivedParamsSafe(self):
        try:
            start_txt = self.dlg1.LineEdit_Start.text().strip()
            stop_txt = self.dlg1.LineEdit_Stop.text().strip()
            if not start_txt or not stop_txt:
                return

            if hasattr(self.dlg1.LineEdit_Start, "hasAcceptableInput") and not self.dlg1.LineEdit_Start.hasAcceptableInput():
                return
            if hasattr(self.dlg1.LineEdit_Stop, "hasAcceptableInput") and not self.dlg1.LineEdit_Stop.hasAcceptableInput():
                return

            start_nm = float(start_txt)
            stop_nm = float(stop_txt)
            if not (start_nm < stop_nm):
                return

            velocity_s = self.parseTimeToSeconds(self.dlg1.ComboBox_Velocity.currentText())
            integration = int(self.dlg1.ComboBox_Integration.currentText())
            sa_s = self.parseTimeToSeconds(self.dlg1.ComboBox_Sa.currentText())
            if velocity_s <= 0 or integration <= 0 or sa_s <= 0:
                return

            cutoff_thz, resolution_thz = self.computeCutoffAndResolution(
                start_nm=start_nm,
                stop_nm=stop_nm,
                velocity_s=velocity_s,
                integration=integration,
                sa_s=sa_s,
            )
            if (not np.isfinite(cutoff_thz)) or (not np.isfinite(resolution_thz)):
                return
            if cutoff_thz <= 0 or resolution_thz <= 0:
                return

            cutoff_wl_nm = 299792.458 / cutoff_thz  # λ[nm] = 299792.458 / f[THz]

            for w, txt in (
                (self.dlg1.LineEdit_Cutoff, f"{cutoff_thz:.3f}"),
                (self.dlg1.LineEdit_Resolution, f"{resolution_thz:.6f}"),
                (self.dlg1.LineEdit_Cutoff_WL, f"{cutoff_wl_nm:.2f}"),
            ):
                with QSignalBlocker(w):
                    w.setText(txt)

        except Exception:
            return

    def computeCutoffAndResolution(
        self, *,
        start_nm: float,
        stop_nm: float,
        velocity_s: float,
        integration: int,
        sa_s: float
    ) -> tuple[float, float]:
        # TODO: 正しい物理/仕様の式へ置換
        cutoff_thz = 1.0 / max(sa_s, 1e-9) * 1e-12
        resolution_thz = cutoff_thz / max(integration, 1)
        return cutoff_thz, resolution_thz

    # =========================================================
    # Measurement / Monitor (state machine via QTimer; no blocking)
    # =========================================================
    def setOrigin(self):
        self.setLineEdit("LineEdit_Pos", "0")
        self.log("0")

    def measureData(self):
        """Button_Measure: run scripted acquisition for N iterations (demo: random signal for 3 s)."""
        if self._meas_active:
            self.log("Measure already running.")
            return

        it_le = getattr(self.dlg1, "LineEdit_Iteration", None)
        if it_le is None:
            self.log("LineEdit_Iteration not found.")
            return

        txt = it_le.text().strip()
        if not txt:
            self.log("Iteration is empty.")
            return

        try:
            iterations = int(txt)
        except ValueError:
            self.log(f"Invalid iteration: '{txt}'")
            return

        if iterations <= 0:
            self.log("Iteration must be >= 1.")
            return

        self.startMeasureSequence(iterations=iterations, monitor_mode=False)

    def monitorSwitch(self, checked: bool):
        """Monitor checkbox: repeat demo acquisition while checked; on uncheck, run Button_Go logic and finish."""
        if checked:
            if self._meas_active:
                self._meas_monitor_mode = True
                self.log("Monitor ON (will repeat after current run).")
                return

            iterations = self.readIterationOrDefault(default=1)
            self.log("Monitor ON")
            self.startMeasureSequence(iterations=iterations, monitor_mode=True)
        else:
            self.log("Monitor OFF")
            self.stopMeasureSequence()
            self.clearMeasurePlots()
            self._meas_monitor_mode = False
            self.moveStage()

    def initMeasureSystem(self):
        self._measure_timer = QTimer(self)
        self._measure_timer.setInterval(30)  # ms
        self._measure_timer.timeout.connect(self.onMeasureTick)

        self._meas_active = False
        self._meas_monitor_mode = False
        self._meas_total_iterations = 0
        self._meas_iter_idx = 0

        self._meas_iter_duration_s = 3.0
        self._meas_iter_t0 = 0.0

        # reuse BaseSignalGenerator's get_plot_data (we won't call __call__)
        self._meas_gen = BaseSignalGenerator()

        self._meas_lines_time = []
        self._meas_lines_mag = []
        self._meas_current_line_time = None
        self._meas_current_line_mag = None

        self._meas_base_pos = 0.0
        self._meas_pos_step = 0.5  # mm (demo)

        # X mapping
        self._meas_x_start = 0.0
        self._meas_x_stop = self._meas_iter_duration_s

        # Spectrum curves
        self._spec_curve_freq = None
        self._spec_curve_wl = None

        # Last iteration pen (match colors)
        self._meas_last_pen = None

    def clearMeasurePlots(self):
        """Remove previously plotted *measurement* curves before drawing new ones."""
        view_time = getattr(self.dlg1, "graphicsView_Time", None)
        view_mag = getattr(self.dlg1, "graphicsView_Time_Magnify", None)

        for view, items in ((view_time, self._meas_lines_time), (view_mag, self._meas_lines_mag)):
            if view is None:
                continue
            for it in list(items):
                try:
                    view.removeItem(it)
                except Exception:
                    try:
                        it.setData([], [])
                    except Exception:
                        pass

        self._meas_lines_time.clear()
        self._meas_lines_mag.clear()
        self._meas_current_line_time = None
        self._meas_current_line_mag = None

    def readIterationOrDefault(self, default: int = 1) -> int:
        it_le = getattr(self.dlg1, "LineEdit_Iteration", None)
        if it_le is None:
            return default
        txt = it_le.text().strip()
        try:
            v = int(txt)
            return v if v > 0 else default
        except Exception:
            return default

    def readCurrentPos(self) -> float:
        le = getattr(self.dlg1, "LineEdit_Pos", None)
        if le is None:
            return 0.0
        try:
            return float(le.text().strip() or 0.0)
        except Exception:
            return 0.0

    def startMeasureSequence(self, *, iterations: int, monitor_mode: bool):
        self.clearMeasurePlots()

        self._meas_active = True
        self._meas_monitor_mode = monitor_mode
        self._meas_total_iterations = iterations
        self._meas_iter_idx = 0

        self._meas_base_pos = self.readCurrentPos()
        self.log(f"Measure start: iterations={iterations}, monitor={monitor_mode}")
        self.startNextIteration()

    def stopMeasureSequence(self):
        if self._measure_timer.isActive():
            self._measure_timer.stop()
        self._meas_active = False
        self.unlockTimeXRangeAfterMeasure()

    def lockTimeXRangeForMeasure(self):
        view_time = getattr(self.dlg1, "graphicsView_Time", None)
        if view_time is None:
            return
        pi = view_time.plotItem
        pi.enableAutoRange("x", False)
        pi.setXRange(self._meas_x_start, self._meas_x_stop, padding=0)

    def unlockTimeXRangeAfterMeasure(self):
        view_time = getattr(self.dlg1, "graphicsView_Time", None)
        if view_time is None:
            return
        view_time.plotItem.enableAutoRange("x", True)

    def startNextIteration(self):
        if not self._meas_active:
            return

        if self._meas_iter_idx >= self._meas_total_iterations:
            self.finishMeasureSequence()
            return

        # Iteration境界で前回曲線を消す
        if self._meas_iter_idx > 0:
            self.clearMeasurePlots()

        # stage movement (demo)
        pos = self._meas_base_pos + self._meas_pos_step * self._meas_iter_idx
        self.setLineEdit("LineEdit_Pos", f"{pos}")
        self.log(f"[Measure] Move stage -> {pos} mm (iter {self._meas_iter_idx + 1}/{self._meas_total_iterations})")

        # reset buffers
        self._meas_gen.x.clear()
        self._meas_gen.y.clear()
        self._meas_iter_t0 = time.perf_counter()

        # pen
        pen = pg.mkPen(pg.intColor(len(self._meas_lines_time), hues=12), width=2)
        self._meas_last_pen = pen

        view_time = getattr(self.dlg1, "graphicsView_Time", None)
        view_mag = getattr(self.dlg1, "graphicsView_Time_Magnify", None)

        if view_time is not None:
            line_t = view_time.plot([], [], pen=pen)
            self._meas_current_line_time = line_t
            self._meas_lines_time.append(line_t)

        if view_mag is not None:
            line_m = view_mag.plot([], [], pen=pen)
            self._meas_current_line_mag = line_m
            self._meas_lines_mag.append(line_m)

        # x mapping for this iteration (Start..Stop)
        x0 = self.readFloatLineEdit("LineEdit_Start", 0.0)
        x1 = self.readFloatLineEdit("LineEdit_Stop", self._meas_iter_duration_s)
        if (not np.isfinite(x0)) or (not np.isfinite(x1)) or (x0 == x1):
            x0, x1 = 0.0, self._meas_iter_duration_s
        self._meas_x_start, self._meas_x_stop = float(x0), float(x1)

        # lock ranges
        self.lockTimeXRangeForMeasure()
        self.applyMagnifyXRange()

        self._measure_timer.start()

    def onMeasureTick(self):
        if not self._meas_active:
            self._measure_timer.stop()
            return

        elapsed = time.perf_counter() - self._meas_iter_t0
        if elapsed >= self._meas_iter_duration_s:
            self._measure_timer.stop()

            # FFT update before next iteration
            self.updateSpectrumViewsFromLastIteration()

            self._meas_iter_idx += 1
            QTimer.singleShot(0, self.startNextIteration)
            return

        frac = elapsed / max(self._meas_iter_duration_s, 1e-9)
        x_val = self._meas_x_start + (self._meas_x_stop - self._meas_x_start) * frac

        self._meas_gen.x.append(float(x_val))
        self._meas_gen.y.append(float(np.random.normal(loc=0.0, scale=1.0)))

        x, y = self._meas_gen.get_plot_data()
        if x.size == 0 or y.size == 0 or x.size != y.size:
            return
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            return
        if x.size < 2:
            return

        if self._meas_current_line_time is not None:
            self._meas_current_line_time.setData(x, y)

        if self._meas_current_line_mag is not None:
            pr = self.readPlotRangeOrNone()
            if pr is None:
                self._meas_current_line_mag.setData(x, y)
            else:
                x0, x1 = pr
                mask = (x >= x0) & (x <= x1)
                xm, ym = x[mask], y[mask]
                if xm.size < 2:
                    self._meas_current_line_mag.setVisible(False)
                    return
                self._meas_current_line_mag.setVisible(True)
                self._meas_current_line_mag.setData(xm, ym)

    def finishMeasureSequence(self):
        self._meas_active = False
        self.log("Measure finished.")

        mon = getattr(self.dlg1, "checkBox", None)
        if self._meas_monitor_mode and mon is not None and mon.isChecked():
            iterations = self._meas_total_iterations
            QTimer.singleShot(100, lambda: self.startMeasureSequence(iterations=iterations, monitor_mode=True))
        else:
            self.unlockTimeXRangeAfterMeasure()

    # =========================================================
    # Spectrum (FFT) update
    # =========================================================
    def updateSpectrumViewsFromLastIteration(self):
        """
        直近 iteration の時間波形 (self._meas_gen.x, self._meas_gen.y) をFFTし、
        graphicsView_Frequency と graphicsView_Wavelength を更新する。
        波長は 0.1–500 um に制限し、描画後にWL範囲指定も即反映。
        """
        view_f = getattr(self.dlg1, "graphicsView_Frequency", None)
        view_w = getattr(self.dlg1, "graphicsView_Wavelength", None)
        if view_f is None and view_w is None:
            return

        x = np.asarray(self._meas_gen.x, dtype=float)
        y = np.asarray(self._meas_gen.y, dtype=float)

        if x.size < 4 or y.size < 4 or x.size != y.size:
            return
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            return

        # sort by x to estimate dt
        order = np.argsort(x)
        x = x[order]
        y = y[order]

        dx = np.diff(x)
        dx = dx[np.isfinite(dx) & (dx > 0)]
        if dx.size == 0:
            return

        # x is assumed ps (UI label: Time(ps))
        dt_ps = float(np.median(dx))
        dt_s = dt_ps * 1e-12
        if not np.isfinite(dt_s) or dt_s <= 0:
            return

        n = y.size
        y0 = y - np.mean(y)  # DC remove
        win = np.hanning(n)

        yf = np.fft.rfft(y0 * win)
        f_hz = np.fft.rfftfreq(n, dt_s)
        amp = np.abs(yf) / max(n, 1)

        # drop DC (0 Hz) to avoid wavelength divergence
        if f_hz.size < 2:
            return
        f_hz = f_hz[1:]
        amp = amp[1:]

        pen = self._meas_last_pen or pg.mkPen("y", width=2)

        # Frequency (THz)
        if view_f is not None:
            f_thz = f_hz / 1e12
            if self._spec_curve_freq is None:
                self._spec_curve_freq = view_f.plot([], [], pen=pen)
            else:
                self._spec_curve_freq.setPen(pen)
            self._spec_curve_freq.setVisible(True)
            self._spec_curve_freq.setData(f_thz, amp)

        # Wavelength (um) in [0.1, 500]
        if view_w is not None:
            lam_um = (C_M_S / f_hz) * 1e6

            mask = (
                np.isfinite(lam_um)
                & (lam_um >= WL_MIN_UM)
                & (lam_um <= WL_MAX_UM)
            )
            lam_um_m = lam_um[mask]
            amp_m = amp[mask]

            if lam_um_m.size < 2:
                if self._spec_curve_wl is not None:
                    self._spec_curve_wl.setVisible(False)
                return

            # sort for nicer plot (wavelength is inverse)
            o2 = np.argsort(lam_um_m)
            lam_um_m = lam_um_m[o2]
            amp_m = amp_m[o2]

            if self._spec_curve_wl is None:
                self._spec_curve_wl = view_w.plot([], [], pen=pen)
            else:
                self._spec_curve_wl.setPen(pen)
            self._spec_curve_wl.setVisible(True)
            self._spec_curve_wl.setData(lam_um_m, amp_m)

            # ★重要：ユーザー指定のWL範囲を即反映（AutoRange暴れ防止）
            self.applyWLXRange()

    # =========================================================
    # Misc / placeholders
    # =========================================================
    def updatePlot(self):
        return


if __name__ == "__main__":
    device_op = None
    gsc02 = None
    # try:
    #     gsc02 = GSC_02_control.GSC02control("COM8", 9600)
    #     gsc02.list_devices()
    #     # ここは connect() を使う流儀に寄せてもOK（現状でも動いているならそのままでOK）
    #     # gsc02.connect()

    #     device_op = gsc02.DeviceOperation(gsc02.ser)
    #     device_op.version_confirmation()
    # except Exception as e:
    #     device_op = None
    #     gsc02 = None
    #     print(f"GSC02 initialization failed: {e}")

    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow(device_op=device_op, gsc02=gsc02)
    sys.exit(app.exec_())
# %%
