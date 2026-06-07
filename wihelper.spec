# -*- mode: python ; coding: utf-8 -*-
"""
WiHelper PyInstaller 打包配置（安全性加固版）
用法: pyinstaller wihelper.spec
"""

import os
import sys

block_cipher = None

here = os.path.dirname(os.path.abspath(SPECPATH))

a = Analysis(
    ['wihelper.py'],
    pathex=[here],
    binaries=[],
    datas=[],
    hiddenimports=[
        'onnxruntime',
        'mss',
        'PIL',
        'pynput',
        'pynput.keyboard',
        'pynput.keyboard._win32',
        'win32gui',
        'win32api',
        'win32con',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'torch',
        'torchvision',
        'tensorflow',
        'keras',
        'matplotlib',
        'scipy',
        'pandas',
        'tkinter',
        'cv2',
        'sklearn',
        'IPython',
        'jupyter',
        'pytest',
        'setuptools',
        'pip',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='WiHelper',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,       # 不用 UPX，避免被 AV 标记
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # 版本信息伪装
    version='version_info.txt',
    icon=None,
)
