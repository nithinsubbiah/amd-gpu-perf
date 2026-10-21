"""Read-only ROCm 7.2 monitoring for the single-GPU reproduction harness."""

import ctypes
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import threading
import time


def run_smi(*args):
    return subprocess.run(["rocm-smi", *args], check=True, capture_output=True,
                          text=True, timeout=10).stdout


def parse_pid_gpus(output):
    """ROCm 7.2 omits --showpidgpus from JSON output; parse its text form."""
    mapping = {}
    lines = output.splitlines()
    for index, line in enumerate(lines):
        match = re.fullmatch(r"PID (\d+) is using (\d+) DRM device\(s\):", line.strip())
        if match is None:
            continue
        pid, count = map(int, match.groups())
        devices = []
        for item in lines[index + 1:]:
            if not re.fullmatch(r"[\d \t]+", item):
                break
            devices.extend(map(int, item.split()))
        if len(devices) != count or pid in mapping:
            raise RuntimeError("Incomplete or duplicate rocm-smi PID mapping")
        mapping[pid] = devices
    return mapping


def hip_bus_id():
    root = Path(os.environ.get("ROCM_PATH", "/opt/rocm"))
    hip = ctypes.CDLL(str(root / "lib/libamdhip64.so"))
    query = hip.hipDeviceGetPCIBusId
    query.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.c_int, ctypes.c_int]
    query.restype = ctypes.c_int
    buf = ctypes.create_string_buffer(64)
    if query(buf, len(buf), 0) != 0:
        raise RuntimeError("Cannot resolve logical GPU 0 to a PCI bus address")
    return buf.value.decode().lower()


def validate_clocks(samples, min_sclk_mhz, max_clock_spread_pct,
                    min_busy_pct=90, min_samples=2):
    """Validate clocks observed under load, excluding before/after idle snapshots."""
    loaded = [s["sclk_mhz"] for s in samples
              if s["phase"] == "timing" and s["gpu_use_pct"] >= min_busy_pct]
    if len(loaded) < min_samples:
        raise RuntimeError("Insufficient clock observations under load; discard this run")
    if min(loaded) < min_sclk_mhz:
        raise RuntimeError(f"Loaded SCLK {min(loaded)} MHz below {min_sclk_mhz} MHz floor")
    median = statistics.median(loaded)
    spread = (max(loaded) - min(loaded)) / median * 100
    if max_clock_spread_pct is not None and spread > max_clock_spread_pct:
        raise RuntimeError(f"Loaded SCLK spread {spread:.1f}% exceeds {max_clock_spread_pct}%")
    return {"samples": len(loaded), "min_mhz": min(loaded), "max_mhz": max(loaded),
            "median_mhz": median, "spread_pct": spread}


class Monitor:
    """Reject observed competition or clock instability; never change GPU settings.

    This samples device state. Use an exclusively allocated GPU: polling cannot
    reserve the device or detect every short-lived competing process.
    """

    def __init__(self, min_sclk_mhz=1000, max_clock_spread_pct=None):
        self.bus = hip_bus_id()
        self.min_sclk_mhz = min_sclk_mhz
        self.max_clock_spread_pct = max_clock_spread_pct
        self.samples = []
        self.error = None
        self.stop = threading.Event()

    def snapshot(self, phase="check"):
        state = json.loads(run_smi("--showbus", "--showclocks", "--showuse", "--json"))
        cards = [(key, value) for key, value in state.items()
                 if value.get("PCI Bus", "").lower() == self.bus]
        if len(cards) != 1:
            raise RuntimeError(f"Cannot uniquely map PCI address {self.bus} in rocm-smi")
        card, values = cards[0]
        physical = int(re.fullmatch(r"card(\d+)", card)[1])
        mapping = parse_pid_gpus(run_smi("--showpidgpus"))
        pids = sorted(pid for pid, devices in mapping.items() if physical in devices)
        if os.getpid() not in pids:
            raise RuntimeError(f"Cannot verify PID {os.getpid()} on PCI {self.bus} "
                               f"(SMI {physical}); observed process mappings: {mapping}")
        if pids != [os.getpid()]:
            raise RuntimeError(f"Competing processes on PCI {self.bus}: {pids}")
        clock = re.fullmatch(r"\((\d+)Mhz\)", values.get("sclk clock speed:", ""))
        if clock is None:
            raise RuntimeError("Cannot read selected GPU SCLK from rocm-smi")
        mhz = int(clock[1])
        busy = int(values["GPU use (%)"])
        if not 0 <= busy <= 100:
            raise RuntimeError("Invalid selected GPU utilization")
        return {"time_ns": time.time_ns(), "pci_bus": self.bus,
                "smi_device": physical, "pids": pids, "sclk_mhz": mhz,
                "gpu_use_pct": busy, "phase": phase}

    def wait_idle(self, timeout=15.0):
        """Require three consecutive zero-utilization samples and no other GPU processes."""
        samples = []
        consecutive = 0
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            sample = self.snapshot(phase="idle_check")
            samples.append(sample)
            consecutive = consecutive + 1 if sample["gpu_use_pct"] == 0 else 0
            if consecutive == 3:
                return samples
            time.sleep(0.25)
        raise RuntimeError("Selected GPU did not become idle; no timing performed")

    def _poll(self):
        try:
            while not self.stop.wait(0.5):
                self.samples.append(self.snapshot(phase="timing"))
        except Exception as error:
            self.error = error

    def __enter__(self):
        self.samples = [self.snapshot(phase="before")]
        self.error = None
        self.stop.clear()
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop.set()
        self.thread.join(timeout=25)
        if self.thread.is_alive():
            raise RuntimeError("GPU monitor failed to finish")
        if exc_type is not None:
            return False
        if self.error is not None:
            raise RuntimeError("GPU monitoring failed; discard this run") from self.error
        self.samples.append(self.snapshot(phase="after"))
        self.clock_validation = validate_clocks(self.samples, self.min_sclk_mhz,
                                                self.max_clock_spread_pct)
        return False
