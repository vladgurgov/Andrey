"""
ADB Controller for Android device interactions.

Handles screenshot capture, touch events, text input, and navigation commands.
"""

import os
import time
import subprocess
from PIL import Image
from typing import Tuple, Optional
import logging

from .config import MobileAgentConfig


class ADBController:
    """Controller for Android Debug Bridge interactions."""
    
    def __init__(self, config: MobileAgentConfig):
        """Initialize ADB controller with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Ensure screenshot directory exists
        os.makedirs(config.screenshot_path, exist_ok=True)
        
        # Check if ADB is available
        self._check_adb_availability()
        
        # Cache device dimensions for coordinate transformation
        self._device_width = None
        self._device_height = None
        self._get_cached_device_dimensions()
    
    def _check_adb_availability(self) -> None:
        """Check if ADB is available and accessible."""
        try:
            result = subprocess.run(
                [self.config.adb_path, "version"],
                capture_output=True,
                text=True,
                timeout=self.config.adb_timeout
            )
            if result.returncode != 0:
                raise RuntimeError(f"ADB not found at {self.config.adb_path}")
            self.logger.info("ADB connection verified")
        except subprocess.TimeoutExpired:
            raise RuntimeError("ADB command timed out")
        except FileNotFoundError:
            raise RuntimeError(f"ADB not found at {self.config.adb_path}")
    
    def get_screenshot(self, filename: str = "screenshot") -> str:
        """
        Capture screenshot from Android device and save as JPG.
        
        Args:
            filename: Base filename for the screenshot (without extension)
            
        Returns:
            Path to the saved screenshot file
        """
        self.logger.info(f"📸 SCREENSHOT: Starting capture for '{filename}'")
        
        # Remove any existing screenshot from device
        self.logger.info("🗑️  Cleaning up old screenshot from device")
        self._run_adb_command("shell rm /sdcard/screenshot.png")
        time.sleep(0.2)
        
        # Capture new screenshot
        self.logger.info("📱 Taking new screenshot on device")
        self._run_adb_command("shell screencap -p /sdcard/screenshot.png")
        time.sleep(self.config.screenshot_delay)
        
        # Pull screenshot to local machine
        local_png_path = os.path.join(self.config.screenshot_path, f"{filename}.png")
        self.logger.info(f"⬇️  Pulling screenshot from device to {local_png_path}")
        self._run_adb_command(f"pull /sdcard/screenshot.png {local_png_path}")
        
        # Convert and resize screenshot
        local_jpg_path = os.path.join(self.config.screenshot_path, f"{filename}.jpg")
        self.logger.info(f"🖼️  Processing: PNG → JPG with {self.config.screenshot_quality}x scale")
        self._process_screenshot(local_png_path, local_jpg_path)
        
        # Clean up PNG file
        if os.path.exists(local_png_path):
            os.remove(local_png_path)
            self.logger.debug(f"🗑️  Cleaned up temporary PNG: {local_png_path}")
        
        self.logger.info(f"✅ Screenshot saved to {local_jpg_path}")
        return local_jpg_path
    
    def _process_screenshot(self, input_path: str, output_path: str) -> None:
        """Process screenshot: resize and convert to JPG."""
        try:
            with Image.open(input_path) as image:
                # Calculate new dimensions
                original_width, original_height = image.size
                new_width = int(original_width * self.config.screenshot_quality)
                new_height = int(original_height * self.config.screenshot_quality)
                
                # Resize and save as JPG
                resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                resized_image.convert("RGB").save(output_path, "JPEG", quality=85)
        except Exception as e:
            self.logger.error(f"Error processing screenshot: {e}")
            raise
    
    def tap(self, x: int, y: int) -> None:
        """
        Tap at specified coordinates.
        
        Args:
            x: X coordinate (from screenshot coordinate system)
            y: Y coordinate (from screenshot coordinate system)
        """
        # Get cached device dimensions for coordinate transformation
        device_width, device_height = self._get_cached_device_dimensions()
        
        # Calculate screenshot dimensions (scaled down)
        screenshot_width = int(device_width * self.config.screenshot_quality)
        screenshot_height = int(device_height * self.config.screenshot_quality)
        
        # Transform coordinates from screenshot coordinate system to device coordinate system
        # Both screenshot and device now use top-left origin (0,0), Y increases downward - no flipping needed
        scale_factor = 1 / self.config.screenshot_quality
        
        # Scale up coordinates (no Y-flipping needed with top-left origin)
        scaled_x = int(x * scale_factor)
        scaled_y = int(y * scale_factor)
        
        # Ensure coordinates are within device bounds
        scaled_x = max(0, min(scaled_x, device_width - 1))
        scaled_y = max(0, min(scaled_y, device_height - 1))
        
        self.logger.info(f"👆 TAP: Screenshot coords ({x}, {y}) → Device coords ({scaled_x}, {scaled_y})")
        self.logger.info(f"     Screenshot size: {screenshot_width}x{screenshot_height}, Device size: {device_width}x{device_height}")
        self.logger.info(f"     Scale factor: {scale_factor:.2f} (top-left origin, no Y-flip needed)")
        
        command = f"shell input tap {scaled_x} {scaled_y}"
        self._run_adb_command(command)
        self.logger.info(f"✅ Tap completed at device coords ({scaled_x}, {scaled_y})")
        time.sleep(self.config.step_delay)
    
    def type_text(self, text: str) -> None:
        """
        Type text on the device.
        
        Args:
            text: Text to type
        """
        # Replace newlines with spaces for simplicity
        original_text = text
        text = text.replace("\\n", " ").replace("\n", " ")
        
        self.logger.info(f"⌨️  TYPE: '{original_text}' → Processed: '{text}'")
        
        # Clear any existing text in the input field
        self.logger.info("🔄 Clearing existing text with Ctrl+A")
        self._run_adb_command("shell input keyevent KEYCODE_CTRL_A")
        time.sleep(0.1)
        
        # Use ADB input method for better compatibility
        escaped_text = text.replace('"', '\\"').replace("'", "\\'")
        self.logger.info(f"✍️  Typing escaped text: '{escaped_text}'")
        command = f'shell input text "{escaped_text}"'
        self._run_adb_command(command)
        
        self.logger.info(f"✅ Text typed successfully: '{text}'")
        time.sleep(self.config.step_delay)
    
    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration: int = 500) -> None:
        """
        Swipe from one point to another.
        
        Args:
            x1, y1: Starting coordinates (from screenshot coordinate system)
            x2, y2: Ending coordinates (from screenshot coordinate system)
            duration: Swipe duration in milliseconds
        """
        # Get cached device dimensions for coordinate transformation
        device_width, device_height = self._get_cached_device_dimensions()
        
        # Calculate screenshot dimensions
        screenshot_width = int(device_width * self.config.screenshot_quality)
        screenshot_height = int(device_height * self.config.screenshot_quality)
        
        # Transform coordinates from screenshot coordinate system to device coordinate system
        # Both screenshot and device now use top-left origin (0,0), Y increases downward - no flipping needed
        scale_factor = 1 / self.config.screenshot_quality
        
        # Scale up coordinates (no Y-flipping needed with top-left origin)
        scaled_x1 = int(x1 * scale_factor)
        scaled_y1 = int(y1 * scale_factor)
        scaled_x2 = int(x2 * scale_factor)
        scaled_y2 = int(y2 * scale_factor)
        
        # Ensure coordinates are within device bounds
        scaled_x1 = max(0, min(scaled_x1, device_width - 1))
        scaled_y1 = max(0, min(scaled_y1, device_height - 1))
        scaled_x2 = max(0, min(scaled_x2, device_width - 1))
        scaled_y2 = max(0, min(scaled_y2, device_height - 1))
        
        self.logger.info(f"👆 SWIPE: Screenshot coords ({x1}, {y1}) → ({x2}, {y2})")
        self.logger.info(f"       Device coords ({scaled_x1}, {scaled_y1}) → ({scaled_x2}, {scaled_y2}) [{duration}ms]")
        self.logger.info(f"       Scale factor: {scale_factor:.2f} (top-left origin, no Y-flip needed)")
        
        command = f"shell input swipe {scaled_x1} {scaled_y1} {scaled_x2} {scaled_y2} {duration}"
        self._run_adb_command(command)
        self.logger.info(f"✅ Swipe completed from device coords ({scaled_x1}, {scaled_y1}) to ({scaled_x2}, {scaled_y2})")
        time.sleep(self.config.step_delay)
    
    def press_back(self) -> None:
        """Press the back button."""
        self.logger.info("🔙 BACK: Pressing back button")
        self._run_adb_command("shell input keyevent KEYCODE_BACK")
        self.logger.info("✅ Back button pressed")
        time.sleep(self.config.step_delay)
    
    def press_home(self) -> None:
        """Press the home button."""
        self.logger.info("🏠 HOME: Pressing home button")
        self._run_adb_command("shell input keyevent KEYCODE_HOME")
        self.logger.info("✅ Home button pressed")
        time.sleep(self.config.step_delay)
    
    def go_to_home_screen(self) -> None:
        """Navigate to the home screen."""
        self.logger.info("🏠 HOME SCREEN: Launching home screen intent")
        command = "shell am start -a android.intent.action.MAIN -c android.intent.category.HOME"
        self._run_adb_command(command)
        self.logger.info("✅ Home screen launched")
        time.sleep(self.config.step_delay)
    
    def open_app(self, package_name: str) -> None:
        """
        Open an app by package name.
        
        Args:
            package_name: Android package name (e.g., com.google.android.youtube)
        """
        self.logger.info(f"📱 OPEN APP: Launching {package_name}")
        command = f"shell monkey -p {package_name} -c android.intent.category.LAUNCHER 1"
        self._run_adb_command(command)
        self.logger.info(f"✅ App launched: {package_name} (waiting {self.config.step_delay * 2}s for startup)")
        time.sleep(self.config.step_delay * 2)  # Apps take longer to open
    
    def get_device_info(self) -> dict:
        """Get basic device information."""
        try:
            # Get device model
            model_result = self._run_adb_command("shell getprop ro.product.model")
            model = model_result.stdout.strip() if model_result.stdout else "Unknown"
            
            # Get Android version
            version_result = self._run_adb_command("shell getprop ro.build.version.release")
            version = version_result.stdout.strip() if version_result.stdout else "Unknown"
            
            # Get screen size
            size_result = self._run_adb_command("shell wm size")
            size = size_result.stdout.strip() if size_result.stdout else "Unknown"
            
            return {
                "model": model,
                "android_version": version,
                "screen_size": size
            }
        except Exception as e:
            self.logger.error(f"Error getting device info: {e}")
            return {}
    
    def _get_cached_device_dimensions(self) -> tuple:
        """Get device dimensions and cache them for coordinate transformation."""
        if self._device_width is None or self._device_height is None:
            device_info = self.get_device_info()
            screen_size = device_info.get('screen_size', '')
            
            # Extract screen dimensions from "Physical size: WIDTHxHEIGHT"
            try:
                if 'Physical size:' in screen_size:
                    size_part = screen_size.split('Physical size:')[1].strip()
                    width_str, height_str = size_part.split('x')
                    self._device_width = int(width_str)
                    self._device_height = int(height_str)
                else:
                    # Default fallback dimensions
                    self._device_width, self._device_height = 720, 1600
                    self.logger.warning(f"Could not parse screen size '{screen_size}', using default {self._device_width}x{self._device_height}")
            except (ValueError, IndexError):
                self._device_width, self._device_height = 720, 1600
                self.logger.warning(f"Could not parse screen size '{screen_size}', using default {self._device_width}x{self._device_height}")
            
            self.logger.info(f"📐 Cached device dimensions: {self._device_width}x{self._device_height}")
        
        return self._device_width, self._device_height
    
    def _run_adb_command(self, command: str) -> subprocess.CompletedProcess:
        """
        Run an ADB command and return the result.
        
        Args:
            command: ADB command (without 'adb' prefix)
            
        Returns:
            subprocess.CompletedProcess object
        """
        full_command = f"{self.config.adb_path} {command}"
        
        # Log the exact command being executed
        self.logger.info(f"🔧 ADB Command: {full_command}")
        
        try:
            result = subprocess.run(
                full_command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.config.adb_timeout
            )
            
            # Log command result
            if result.returncode == 0:
                if result.stdout:
                    self.logger.debug(f"✅ ADB Output: {result.stdout.strip()}")
                else:
                    self.logger.debug(f"✅ ADB Command completed successfully")
            else:
                self.logger.warning(f"⚠️ ADB Command failed (exit code {result.returncode})")
                if result.stderr:
                    self.logger.warning(f"❌ ADB Error: {result.stderr.strip()}")
            
            return result
        except subprocess.TimeoutExpired:
            self.logger.error(f"⏰ ADB command timed out: {full_command}")
            raise
        except Exception as e:
            self.logger.error(f"💥 Error running ADB command '{full_command}': {e}")
            raise 