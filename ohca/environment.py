from __future__ import annotations
import os
import ctypes
import ctypes.wintypes
import datetime
import gc
import subprocess
from typing import Optional

import numba

# Clamped to the per-process cap: inside a joblib worker the cap is lower, and exceeding it raises.
numba.set_num_threads(max(1, min(8, numba.config.NUMBA_NUM_THREADS)))


def log(msg: str) -> None:
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def set_gpu(gpu: Optional[int] = None, memory_mb: Optional[int] = None, threads: Optional[int] = None) -> None:
    # One visible GPU (by default the least-used one) with mixed_float16 compute, optionally a memory cap and
    # a CPU thread cap (several workers per GPU or machine). TensorFlow is imported here, not at module level.
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")   # the order nvidia-smi numbers GPUs in
    import tensorflow as tf
    if threads is not None:
        pin_threads(threads)
        tf.config.threading.set_intra_op_parallelism_threads(threads)
        tf.config.threading.set_inter_op_parallelism_threads(min(2, threads))
    try:
        physical = tf.config.list_physical_devices("GPU")
        if not physical:
            return
        if gpu is None:
            gpu = least_used_gpu()
        tf.config.set_visible_devices(physical[gpu], "GPU")
        if memory_mb is None:
            tf.config.experimental.set_memory_growth(physical[gpu], True)
        else:
            tf.config.set_logical_device_configuration(
                physical[gpu], [tf.config.LogicalDeviceConfiguration(memory_limit=memory_mb)])
    except RuntimeError as e:   # TensorFlow already initialised its devices
        log(f"GPU setup skipped: {e}")
        return
    tf.keras.mixed_precision.set_global_policy("mixed_float16")


def least_used_gpu() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True, timeout=30).stdout
        rows = [tuple(int(v) for v in line.split(",")) for line in out.strip().splitlines() if line.strip()]
        return min(rows, key=lambda r: r[1])[0]
    except (OSError, subprocess.SubprocessError, ValueError):
        return 0


def process_alive(pid: int) -> bool:
    # When in doubt (no access to the process), it counts as alive, so its claim is respected.
    if os.name == "nt":
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.restype  = ctypes.wintypes.HANDLE
        k32.OpenProcess.argtypes = [ctypes.wintypes.DWORD, ctypes.wintypes.BOOL, ctypes.wintypes.DWORD]
        handle = k32.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return ctypes.get_last_error() == 5        # ERROR_ACCESS_DENIED: exists
        code = ctypes.wintypes.DWORD()
        ok   = k32.GetExitCodeProcess(ctypes.wintypes.HANDLE(handle), ctypes.byref(code))
        k32.CloseHandle(ctypes.wintypes.HANDLE(handle))
        return not ok or code.value == 259             # STILL_ACTIVE
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def pin_threads(n: int) -> None:
    # Debounce workers run many processes side by side; their numba kernels are parallel already.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(n)
    numba.set_num_threads(max(1, min(n, numba.config.NUMBA_NUM_THREADS)))


def free_memory() -> None:
    # On Windows, returns freed heap pages to the OS, so the next large allocation in a long fold loop
    # succeeds. Elsewhere only the garbage collection runs.
    gc.collect()
    if os.name != "nt":
        return
    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = ctypes.wintypes.HANDLE
    k32.GetProcessHeap.restype    = ctypes.wintypes.HANDLE
    ctypes.cdll.msvcrt._heapmin()
    k32.SetProcessWorkingSetSize(ctypes.wintypes.HANDLE(k32.GetCurrentProcess()), ctypes.c_size_t(-1),
                                 ctypes.c_size_t(-1))
    k32.HeapCompact(ctypes.wintypes.HANDLE(k32.GetProcessHeap()), 0)
