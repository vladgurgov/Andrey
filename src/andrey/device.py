"""ADB device manager for screenshots and input actions."""

import logging
from typing import Optional

import adbutils
from PIL import Image

from andrey.models import ActionType, GameAction, ScreenInfo

logger = logging.getLogger(__name__)


class DeviceError(Exception):
    """Raised when device operations fail."""


class DeviceManager:
    """Manages ADB device connection and input operations."""

    def __init__(
        self,
        serial: Optional[str] = None,
        adb_host: str = "127.0.0.1",
        adb_port: int = 5037,
    ):
        self._serial = serial
        self._adb_host = adb_host
        self._adb_port = adb_port
        self._device = None
        self._screen_info: Optional[ScreenInfo] = None

    def connect(self) -> None:
        """Connect to ADB device. Raises DeviceError if connection fails."""
        try:
            client = adbutils.AdbClient(host=self._adb_host, port=self._adb_port)
            devices = client.device_list()

            if not devices:
                raise DeviceError(
                    "No ADB devices found. Is the device connected and USB debugging enabled?"
                )

            if self._serial:
                self._device = client.device(serial=self._serial)
            elif len(devices) == 1:
                self._device = devices[0]
            else:
                serials = [d.serial for d in devices]
                raise DeviceError(
                    f"Multiple devices found: {serials}. "
                    f"Specify one with --device or in config.yaml."
                )

            self._screen_info = self._get_screen_info()
            logger.info(
                f"Connected to device {self._device.serial} "
                f"({self._screen_info.width}x{self._screen_info.height})"
            )
        except adbutils.AdbError as e:
            raise DeviceError(f"ADB connection failed: {e}")

    @property
    def screen_info(self) -> ScreenInfo:
        if not self._screen_info:
            raise DeviceError("Not connected. Call connect() first.")
        return self._screen_info

    def _get_screen_info(self) -> ScreenInfo:
        """Query device for screen dimensions and rotation."""
        w, h = self._device.window_size()
        rotation = self._device.rotation()
        return ScreenInfo(width=w, height=h, rotation=rotation)

    def screenshot(self, resize_width: Optional[int] = None) -> Image.Image:
        """Capture device screenshot as PIL Image, optionally resized."""
        if not self._device:
            raise DeviceError("Not connected.")

        try:
            img = self._device.screenshot()
        except Exception as e:
            raise DeviceError(f"Screenshot capture failed: {e}")

        if resize_width and img.width > resize_width:
            ratio = resize_width / img.width
            new_height = int(img.height * ratio)
            img = img.resize((resize_width, new_height), Image.LANCZOS)
            logger.debug(f"Resized screenshot to {resize_width}x{new_height}")

        return img

    def execute_action(self, action: GameAction) -> None:
        """Execute a validated game action on the device."""
        if not self._device:
            raise DeviceError("Not connected.")

        self._validate_coordinates(action)

        match action.action:
            case ActionType.TAP:
                logger.info(f"TAP ({action.x}, {action.y}) - {action.reasoning}")
                self._device.click(action.x, action.y)

            case ActionType.SWIPE:
                duration = action.duration or 0.5
                logger.info(
                    f"SWIPE ({action.x},{action.y})->({action.x2},{action.y2}) "
                    f"duration={duration}s - {action.reasoning}"
                )
                self._device.swipe(
                    action.x, action.y, action.x2, action.y2, duration
                )

            case ActionType.LONG_PRESS:
                duration = action.duration or 1.0
                logger.info(
                    f"LONG_PRESS ({action.x}, {action.y}) {duration}s - {action.reasoning}"
                )
                self._device.swipe(
                    action.x, action.y, action.x, action.y, duration
                )

            case ActionType.KEY:
                logger.info(f"KEY {action.key} - {action.reasoning}")
                self._device.keyevent(action.key)

            case ActionType.TYPE_TEXT:
                logger.info(f"TYPE '{action.text}' - {action.reasoning}")
                self._device.send_keys(action.text)

            case ActionType.WAIT:
                logger.info(f"WAIT - {action.reasoning}")

            case ActionType.GAME_OVER:
                logger.info(f"GAME_OVER detected - {action.reasoning}")

    def _validate_coordinates(self, action: GameAction) -> None:
        """Clamp coordinates to screen bounds with a warning."""
        si = self.screen_info

        def clamp(val, max_val):
            if val is None:
                return None
            clamped = max(0, min(val, max_val))
            if clamped != val:
                logger.warning(
                    f"Coordinate {val} clamped to {clamped} (screen max: {max_val})"
                )
            return clamped

        if action.x is not None:
            action.x = clamp(action.x, si.width)
        if action.y is not None:
            action.y = clamp(action.y, si.height)
        if action.x2 is not None:
            action.x2 = clamp(action.x2, si.width)
        if action.y2 is not None:
            action.y2 = clamp(action.y2, si.height)

    def launch_app(self, package: str) -> None:
        """Launch an app by its package name using monkey."""
        if not self._device:
            raise DeviceError("Not connected.")

        logger.info(f"Launching app: {package}")
        output = self._device.shell(
            f"monkey -p {package} -c android.intent.category.LAUNCHER 1"
        )
        if "No activities found" in output:
            raise DeviceError(
                f"App '{package}' not found on device. Is it installed?"
            )
        logger.info(f"App launched: {package}")

    def get_foreground_package(self) -> Optional[str]:
        """Return the package name of the app currently in the foreground."""
        if not self._device:
            return None
        try:
            output = self._device.shell(
                "dumpsys activity activities | grep mResumedActivity"
            )
            # Output like: mResumedActivity: ActivityRecord{... com.youxi.spades/.MainActivity ...}
            if "/" in output:
                # Extract package name before the /
                parts = output.split()
                for part in parts:
                    if "/" in part and "." in part:
                        return part.split("/")[0].strip("{")
            return None
        except Exception:
            return None

    def press_back(self) -> None:
        """Press the Android back button."""
        if not self._device:
            raise DeviceError("Not connected.")
        logger.info("Pressing Android BACK button")
        self._device.keyevent("KEYCODE_BACK")

    def is_connected(self) -> bool:
        """Check if device is still reachable."""
        try:
            self._device.shell("echo ping")
            return True
        except Exception:
            return False

    @staticmethod
    def list_devices(
        adb_host: str = "127.0.0.1", adb_port: int = 5037
    ) -> list[str]:
        """List all connected device serials."""
        client = adbutils.AdbClient(host=adb_host, port=adb_port)
        return [d.serial for d in client.device_list()]
