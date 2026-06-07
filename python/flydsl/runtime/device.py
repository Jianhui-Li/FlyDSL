# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import functools
import os
import subprocess
from typing import Optional

_ROCM_AGENT_TIMEOUT_S = int(os.environ.get("FLYDSL_ROCM_AGENT_TIMEOUT", "300"))


def _arch_from_rocm_agent_enumerator() -> Optional[str]:
    """Query rocm_agent_enumerator (standard ROCm tool) for the first GPU arch."""
    try:
        out = subprocess.check_output(
            ["rocm_agent_enumerator", "-name"],
            text=True,
            timeout=_ROCM_AGENT_TIMEOUT_S,
            stderr=subprocess.DEVNULL,
        )
        for line in out.splitlines():
            name = line.strip()
            if name.startswith("gfx") and name != "gfx000":
                return name
    except Exception:
        pass
    return None


@functools.lru_cache(maxsize=None)
def _arch_from_hardware() -> str:
    """Cached hardware detection (rocm_agent_enumerator is slow)."""
    arch = _arch_from_rocm_agent_enumerator()
    if arch:
        return arch.split(":", 1)[0]
    return "gfx942"


def get_rocm_arch() -> str:
    """Best-effort ROCm GPU arch string (e.g. 'gfx942')."""
    env = os.environ.get("FLYDSL_GPU_ARCH") or os.environ.get("HSA_OVERRIDE_GFX_VERSION")
    if env:
        if env.startswith("gfx"):
            return env
        if env.count(".") == 2:
            parts = env.split(".")
            return f"gfx{parts[0]}{parts[1]}{parts[2]}"

    return _arch_from_hardware()


@functools.lru_cache(maxsize=None)
def get_rocm_device_count() -> int:
    """Best-effort ROCm visible GPU count via ``rocm_agent_enumerator`` (standard ROCm tool).

    Uses the same invocation as :func:`_arch_from_rocm_agent_enumerator`. Returns 0
    when the tool is unavailable or no discrete GPU agents are reported.
    """
    try:
        out = subprocess.check_output(
            ["rocm_agent_enumerator", "-name"],
            text=True,
            timeout=5,
            stderr=subprocess.DEVNULL,
        )
        n = 0
        for line in out.splitlines():
            name = line.strip()
            if name.startswith("gfx") and name != "gfx000":
                n += 1
        return n
    except Exception:
        return 0


def is_rdna_arch(arch: Optional[str] = None) -> bool:
    """Check if architecture is RDNA-based (gfx10/11/12, wave32).

    This is the single source of truth for CDNA vs RDNA classification.
    RDNA architectures use wave32 and have different buffer descriptor flags.

    If arch is None, the current GPU arch is auto-detected.
    """
    if arch is None:
        arch = get_rocm_arch()
    if not arch:
        return False
    arch = arch.lower()
    if arch.startswith("gfx10") or arch.startswith("gfx11"):
        return True
    if arch.startswith("gfx120"):
        return True
    return False


# ---------------------------------------------------------------------------
# Intel GPU detection (XeGPU backend).
# ---------------------------------------------------------------------------

# Known Intel GPU arch tokens used by the xegpu backend / upstream xevm.
# Extend as we onboard more sub-archs.
_INTEL_ARCH_TOKENS = {"pvc", "bmg"}

# Intel PCI device-id ranges that map to xevm zebin chip tokens. Probed via
# /sys/class/drm/card*/device/device. Sources:
#   - PVC (Ponte Vecchio / Data Center GPU Max 1100/1550): 0x0bd0..0x0bdf
#   - BMG (Battlemage / Arc B): 0xe200..0xe2ff (B580=0xe20b, B570=0xe211, ...)
_INTEL_PCI_RANGES = (
    (0x0BD0, 0x0BDF, "pvc"),
    (0xE200, 0xE2FF, "bmg"),
)


@functools.lru_cache(maxsize=None)
def _intel_arch_from_sysfs() -> Optional[str]:
    """Probe /sys/class/drm/card*/device/{vendor,device} for an Intel GPU."""
    import glob

    for card in sorted(glob.glob("/sys/class/drm/card[0-9]*")):
        try:
            with open(f"{card}/device/vendor") as f:
                vendor = int(f.read().strip(), 16)
            if vendor != 0x8086:
                continue
            with open(f"{card}/device/device") as f:
                dev = int(f.read().strip(), 16)
            for lo, hi, token in _INTEL_PCI_RANGES:
                if lo <= dev <= hi:
                    return token
        except (OSError, ValueError):
            continue
    return None


def get_intel_arch() -> str:
    """Best-effort Intel GPU arch token (e.g. 'pvc', 'bmg').

    Resolution order:
      1. ``FLYDSL_GPU_ARCH`` or ``ARCH`` env var (must match a known token).
      2. /sys/class/drm/card*/device/device PCI-ID lookup.
      3. ``"bmg"`` fallback.
    """
    arch = os.environ.get("FLYDSL_GPU_ARCH") or os.environ.get("ARCH") or ""
    arch = arch.strip().lower()
    if arch in _INTEL_ARCH_TOKENS:
        return arch
    detected = _intel_arch_from_sysfs()
    if detected:
        return detected
    return "bmg"


def is_intel_arch(arch: Optional[str] = None) -> bool:
    """Check if *arch* is an Intel GPU target (XeGPU backend).

    Single source of truth alongside :func:`is_rdna_arch`. Intel archs run
    SIMD16 lanes in this build (see ``get_warp_size``). If *arch* is None,
    falls back to :func:`get_intel_arch`.
    """
    if arch is None:
        arch = get_intel_arch()
    if not arch:
        return False
    return arch.strip().lower() in _INTEL_ARCH_TOKENS
