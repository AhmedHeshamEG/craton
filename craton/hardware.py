"""Hardware probe and precision planning.

Craton has to run on a laptop first: an RTX 2060 with 6 GB of VRAM. That GPU is Turing
(compute capability 7.5). It has fast fp16 tensor cores and no native bf16, so a bf16
request there quietly becomes fp16 with gradient scaling, and the probe says so out loud.
"""

from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Precision = Literal["fp32", "fp16", "bf16"]
Device = Literal["cuda", "cpu"]

#: bf16 tensor cores arrived with Ampere (compute capability 8.0).
BF16_MIN_CC = (8, 0)
#: fp16 tensor cores arrived with Volta (7.0); Turing (7.5) has them.
FP16_TENSOR_CORE_MIN_CC = (7, 0)


@dataclass(frozen=True)
class GPU:
    name: str
    vram_gib: float
    compute_capability: tuple[int, int]

    @property
    def architecture(self) -> str:
        return architecture_name(self.compute_capability)

    @property
    def bf16(self) -> bool:
        return self.compute_capability >= BF16_MIN_CC

    @property
    def fp16_tensor_cores(self) -> bool:
        return self.compute_capability >= FP16_TENSOR_CORE_MIN_CC


@dataclass(frozen=True)
class HardwareReport:
    python: str
    os: str
    torch_version: str | None
    torch_cuda: str | None
    gpus: tuple[GPU, ...]
    ram_gib: float | None
    #: Where the GPU facts came from: torch sees the GPU, only nvidia-smi does, or nothing.
    source: Literal["torch", "nvidia-smi", "none"]
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def gpu(self) -> GPU | None:
        return self.gpus[0] if self.gpus else None

    @property
    def cuda_usable(self) -> bool:
        """True only when PyTorch itself can use the GPU."""
        return self.source == "torch" and bool(self.gpus)


@dataclass(frozen=True)
class PrecisionPlan:
    requested: Precision
    dtype: Precision
    device: Device
    grad_scaler: bool
    warnings: tuple[str, ...]


def architecture_name(cc: tuple[int, int]) -> str:
    """Human name of an NVIDIA architecture from its compute capability."""
    names = {
        (6, 0): "Pascal",
        (6, 1): "Pascal",
        (7, 0): "Volta",
        (7, 2): "Volta",
        (7, 5): "Turing",
        (8, 0): "Ampere",
        (8, 6): "Ampere",
        (8, 7): "Ampere",
        (8, 9): "Ada Lovelace",
        (9, 0): "Hopper",
        (10, 0): "Blackwell",
        (12, 0): "Blackwell",
    }
    return names.get(cc, f"sm_{cc[0]}{cc[1]}")


def plan_precision(requested: Precision, report: HardwareReport) -> PrecisionPlan:
    """Decide the dtype a run actually uses on this machine.

    - No usable GPU: fp32 on the CPU (half precision on a CPU is slow and pointless).
    - bf16 on a GPU older than Ampere: fp16 + gradient scaling, with a warning.
    - fp16: always with a gradient scaler, because fp16's narrow range underflows small gradients.
    """
    warnings: list[str] = []
    gpu = report.gpu if report.cuda_usable else None

    if gpu is None:
        if requested != "fp32":
            warnings.append(f"No usable CUDA GPU, so {requested} falls back to fp32 on the CPU.")
        return PrecisionPlan(requested, "fp32", "cpu", False, tuple(warnings))

    if requested == "bf16" and not gpu.bf16:
        warnings.append(
            f"{gpu.name} is {gpu.architecture} (sm_{gpu.compute_capability[0]}"
            f"{gpu.compute_capability[1]}) and has no native bf16. Using fp16 with gradient "
            "scaling instead."
        )
        return PrecisionPlan(requested, "fp16", "cuda", True, tuple(warnings))

    if requested == "fp16":
        if not gpu.fp16_tensor_cores:
            warnings.append(f"{gpu.name} has no fp16 tensor cores; fp16 will not be faster.")
        return PrecisionPlan(requested, "fp16", "cuda", True, tuple(warnings))

    return PrecisionPlan(requested, requested, "cuda", False, tuple(warnings))


def probe() -> HardwareReport:
    """Look at this machine: Python, OS, RAM, PyTorch, and every NVIDIA GPU."""
    notes: list[str] = []
    torch_version: str | None = None
    torch_cuda: str | None = None
    gpus: tuple[GPU, ...] = ()
    source: Literal["torch", "nvidia-smi", "none"] = "none"

    try:
        import torch

        torch_version = str(torch.__version__)
        torch_cuda = torch.version.cuda
        if torch.cuda.is_available():
            found = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                found.append(
                    GPU(
                        name=str(props.name),
                        vram_gib=round(props.total_memory / 2**30, 2),
                        compute_capability=(int(props.major), int(props.minor)),
                    )
                )
            gpus = tuple(found)
            source = "torch"
    except ImportError:
        notes.append(
            "PyTorch is not installed. Run `uv sync --extra cu126` (NVIDIA) or `--extra cpu`."
        )

    if source == "none":
        smi = _nvidia_smi_gpus()
        if smi:
            gpus = smi
            source = "nvidia-smi"
            if torch_version is not None:
                notes.append(
                    "An NVIDIA GPU is present but PyTorch can't use it (a CPU-only build is "
                    "installed). Run `uv sync --extra cu126`."
                )

    return HardwareReport(
        python=platform.python_version(),
        os=_os_name(),
        torch_version=torch_version,
        torch_cuda=torch_cuda,
        gpus=gpus,
        ram_gib=_total_ram_gib(),
        source=source,
        notes=tuple(notes),
    )


def _os_name() -> str:
    system, release = platform.system(), platform.release()
    if system == "Windows":
        # Windows 11 still reports release "10"; the build number tells them apart.
        try:
            build = int(platform.version().split(".")[-1])
        except ValueError:
            build = 0
        if build >= 22000:
            release = "11"
    return f"{system} {release}"


def _nvidia_smi_gpus() -> tuple[GPU, ...]:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return ()
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return ()
    return parse_nvidia_smi(out)


def parse_nvidia_smi(csv_text: str) -> tuple[GPU, ...]:
    """Parse ``nvidia-smi --query-gpu=name,memory.total,compute_cap`` CSV (MiB, no units)."""
    gpus = []
    for line in csv_text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        name, mib, cc = parts
        try:
            major, minor = (int(x) for x in cc.split("."))
            gpus.append(GPU(name, round(float(mib) / 1024, 2), (major, minor)))
        except ValueError:
            continue
    return tuple(gpus)


def _total_ram_gib() -> float | None:
    try:
        if sys.platform == "win32":
            return _windows_ram_gib()
        if sys.platform == "darwin":
            out = subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True
            ).stdout
            return round(int(out) / 2**30, 1)
        meminfo = Path("/proc/meminfo")
        if meminfo.exists():
            for line in meminfo.read_text().splitlines():
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / 2**20, 1)
        pages = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        return round(pages / 2**30, 1)
    except (OSError, ValueError, subprocess.SubprocessError, AttributeError):
        return None


def _windows_ram_gib() -> float:
    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(MemoryStatusEx)
    windll = getattr(ctypes, "windll")  # noqa: B009 (only exists on Windows; keeps mypy portable)
    windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    total: int = status.ullTotalPhys
    return round(total / 2**30, 1)


@dataclass(frozen=True)
class SmokeResult:
    device: str
    dtype: str
    steps: int
    first_loss: float
    last_loss: float
    peak_vram_mib: float | None
    seconds: float

    @property
    def learned(self) -> bool:
        return self.last_loss < self.first_loss


def smoke_test(plan: PrecisionPlan, steps: int = 60, seed: int = 0) -> SmokeResult:
    """Train a tiny network for a few steps with the planned precision.

    Proves the whole mixed-precision path works on this machine: autocast, the gradient
    scaler, the optimiser step, and that the loss goes down. Needs PyTorch.
    """
    import time

    import torch

    torch.manual_seed(seed)
    device = torch.device(plan.device)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[plan.dtype]
    net = torch.nn.Sequential(
        torch.nn.Linear(64, 256), torch.nn.GELU(), torch.nn.Linear(256, 1)
    ).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3)
    scaler = torch.amp.GradScaler(device.type, enabled=plan.grad_scaler)
    x = torch.randn(512, 64, device=device)
    y = (x[:, :8].sum(dim=1, keepdim=True)).sin()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    losses = []
    for _ in range(steps):
        with torch.autocast(device.type, dtype=dtype, enabled=plan.dtype != "fp32"):
            loss = torch.nn.functional.mse_loss(net(x).float(), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        losses.append(float(loss.detach()))
    if device.type == "cuda":
        torch.cuda.synchronize()
    took = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None
    return SmokeResult(plan.device, plan.dtype, steps, losses[0], losses[-1], peak, took)
