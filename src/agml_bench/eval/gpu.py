"""NVML sampling of GPU memory, utilization and power."""

import threading
import time
from contextlib import nullcontext

import numpy as np


class GPUSampler:
    """Samples NVML every `interval` s in a thread. Use as a context manager; read .summary()."""

    def __init__(self, interval: float = 0.5):
        import pynvml

        self.nv, self.interval, self.samples = pynvml, interval, []
        pynvml.nvmlInit()
        self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
        self._stop = threading.Event()

    def info(self) -> dict:
        return {"gpus": [{"name": self.nv.nvmlDeviceGetName(h), "memory_gb": round(self.nv.nvmlDeviceGetMemoryInfo(h).total / 1e9, 1)}
                         for h in self.handles]}

    def _read(self) -> dict:
        mem = sum(self.nv.nvmlDeviceGetMemoryInfo(h).used for h in self.handles) / 1e9
        util = float(np.mean([self.nv.nvmlDeviceGetUtilizationRates(h).gpu for h in self.handles]))
        try:
            power = sum(self.nv.nvmlDeviceGetPowerUsage(h) for h in self.handles) / 1000
        except self.nv.NVMLError:
            power = 0.0
        return {"t": time.time(), "mem_gb": mem, "util": util, "power_w": power}

    def _loop(self):
        while not self._stop.is_set():
            self.samples.append(self._read())
            time.sleep(self.interval)

    def __enter__(self):
        self.baseline_gb = self._read()["mem_gb"]
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()

    def summary(self, t0: float = 0, t1: float = float("inf")) -> dict:
        s = [x for x in self.samples if t0 <= x["t"] <= t1]
        if not s:
            return {}
        return {"peak_vram_gb": round(max(x["mem_gb"] for x in s) - self.baseline_gb, 2),
                "mean_util": round(float(np.mean([x["util"] for x in s])), 1),
                "mean_power_w": round(float(np.mean([x["power_w"] for x in s])), 1)}


def sampler():
    """A started-on-enter GPUSampler, or a null context yielding None when NVML is unavailable."""
    try:
        return GPUSampler()
    except Exception:
        return nullcontext(None)
