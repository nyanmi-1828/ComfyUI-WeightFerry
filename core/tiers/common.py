# SPDX-License-Identifier: MIT-0
# Copyright (c) 2026 nyanmi-1828
"""
Shared constants of the tiered store (MB, I/O alignment, platform) and the Windows kernel32 file / memory API bindings (only on Windows).
"""

import os


MB = 1024 ** 2
_ALIGN_IO = 4096
_IS_WINDOWS = os.name == "nt"


def align_up(n, a=_ALIGN_IO):
    return (n + a - 1) // a * a


# ---------------------------------------------------------------------------------------------------------------------
# Low-level file I/O (unbuffered on Windows)
# ---------------------------------------------------------------------------------------------------------------------
if _IS_WINDOWS:
    import ctypes
    from ctypes import wintypes

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _GENERIC_READ = 0x80000000
    _FILE_SHARE_READ = 0x00000001
    _OPEN_EXISTING = 3
    _FILE_FLAG_NO_BUFFERING = 0x20000000
    _FILE_FLAG_SEQUENTIAL_SCAN = 0x08000000
    _INVALID_HANDLE = ctypes.c_void_p(-1).value

    _k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                                 wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    _k32.CreateFileW.restype = wintypes.HANDLE
    _k32.ReadFile.argtypes = [wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD,
                              ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID]
    _k32.ReadFile.restype = wintypes.BOOL
    _GENERIC_WRITE = 0x40000000
    _k32.WriteFile.argtypes = [wintypes.HANDLE, wintypes.LPCVOID, wintypes.DWORD,
                               ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID]
    _k32.WriteFile.restype = wintypes.BOOL
    _k32.SetFilePointerEx.argtypes = [wintypes.HANDLE, ctypes.c_longlong, ctypes.POINTER(ctypes.c_longlong),
                                      wintypes.DWORD]
    _k32.SetFilePointerEx.restype = wintypes.BOOL
    _k32.CloseHandle.argtypes = [wintypes.HANDLE]
    _k32.CloseHandle.restype = wintypes.BOOL
