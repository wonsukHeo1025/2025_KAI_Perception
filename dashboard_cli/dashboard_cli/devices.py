from dataclasses import dataclass
from glob import glob
import os
import subprocess
from typing import List, Optional


@dataclass
class DeviceInfo:
    dev: str
    vendor: str
    model: str
    by_id: Optional[str] = None
    cls: Optional[str] = None  # heuristic class: Arduino/GPS/IMU/Camera/Serial/Video


_VENDOR_KEYS = (
    'ID_VENDOR',
    'ID_VENDOR_FROM_DATABASE',
)
_MODEL_KEYS = (
    'ID_MODEL',
    'ID_MODEL_FROM_DATABASE',
    'ID_V4L_PRODUCT',
)


def _udev_props(dev: str) -> dict:
    try:
        out = subprocess.check_output(['udevadm', 'info', '-q', 'property', '-n', dev], text=True)
    except Exception:
        return {}
    props = {}
    for line in out.splitlines():
        if '=' in line:
            k, v = line.split('=', 1)
            props[k.strip()] = v.strip()
    return props


def _pick(props: dict, keys) -> str:
    for k in keys:
        v = props.get(k)
        if v:
            return v
    return ''


def _symlink_by_id(dev: str) -> Optional[str]:
    try:
        for link in glob('/dev/serial/by-id/*'):
            try:
                target = os.path.realpath(link)
            except Exception:
                continue
            if target == dev:
                return os.path.basename(link)
    except Exception:
        pass
    return None


def _classify(vendor: str, model: str, dev: str) -> Optional[str]:
    v = (vendor or '').lower()
    m = (model or '').lower()
    d = dev.lower()
    if 'arduino' in v or 'arduino' in m:
        return 'Arduino'
    if 'ublox' in v or 'u-blox' in v or 'ublox' in m or 'gps' in m:
        return 'GPS'
    if 'xsens' in v or 'xsens' in m or 'myahrs' in v or 'myahrs' in m or 'imu' in m:
        return 'IMU'
    if d.startswith('/dev/video') or 'camera' in m or 'uvc' in m:
        return 'Camera'
    if d.startswith('/dev/tty'):
        return 'Serial'
    return None


def list_devices() -> List[DeviceInfo]:
    devs = []
    paths = []
    try:
        paths.extend(sorted(glob('/dev/ttyACM*')))
        paths.extend(sorted(glob('/dev/ttyUSB*')))
        paths.extend(sorted(glob('/dev/video*')))
    except Exception:
        pass
    for p in paths:
        props = _udev_props(p)
        vendor = _pick(props, _VENDOR_KEYS)
        model = _pick(props, _MODEL_KEYS)
        # VID/PID fallback
        if not vendor:
            vid = props.get('ID_VENDOR_ID')
            vendor = f'0x{vid}' if vid else ''
        if not model:
            pid = props.get('ID_MODEL_ID')
            model = f'0x{pid}' if pid else ''

        by_id = _symlink_by_id(p)
        cls = _classify(vendor, model, p)
        devs.append(DeviceInfo(dev=p, vendor=vendor, model=model, by_id=by_id, cls=cls))
    return devs


class DeviceScanner:
    def __init__(self):
        self.devices: List[DeviceInfo] = []

    def scan(self) -> None:
        self.devices = list_devices()

