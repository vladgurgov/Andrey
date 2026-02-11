"""Main game-playing loop: multi-turn tool-use with OmniParser."""

import logging
import signal
import time
from pathlib import Path
from typing import Optional

from PIL import Image

from andrey.config import AppConfig
from andrey.device import DeviceError, DeviceManager
from andrey.models import (
    TOOL_DEFINITIONS,
    ActionType,
    GameAction,
    ToolCall,
)
from andrey.omniparser import OmniParserClient, ParseResult, UIElement
from andrey.prompts import build_system_prompt, load_game_profile
from andrey.vision import ConversationClient, VisionError

logger = logging.getLogger(__name__)


class AgentLoop:
    """Orchestrates the screenshot-analyze-act game loop.

    Each step: capture screenshot → OmniParser → Claude tool call → execute → repeat.
    """

    def __init__(self, config: AppConfig):
        self._config = config
        self._device = DeviceManager(
            serial=config.device.serial,
            adb_host=config.device.adb_host,
            adb_port=config.device.adb_port,
        )
        self._game_profile = load_game_profile(config.game_profile)
        self._conversation: Optional[ConversationClient] = None

        # OmniParser setup
        if config.omniparser.enabled:
            self._omniparser = OmniParserClient(
                omniparser_path=config.omniparser.omniparser_path,
                weights_path=config.omniparser.weights_path,
                device=config.omniparser.device,
                box_threshold=config.omniparser.box_threshold,
                iou_threshold=config.omniparser.iou_threshold,
                use_paddleocr=config.omniparser.use_paddleocr,
            )
        else:
            self._omniparser = None

        # State
        self._step = 0
        self._consecutive_errors = 0
        self._no_change_count = 0
        self._last_screenshot_hash: Optional[int] = None
        self._running = False
        self._screenshot_dir = Path(config.screenshot_dir)

        # Current detected elements (updated each step)
        self._elements: list[UIElement] = []

    def run(self, extra_context: Optional[str] = None) -> None:
        """Start the game-playing loop."""
        self._running = True
        original_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_interrupt)

        try:
            self._device.connect()
            screen_info = self._device.screen_info

            # Check OmniParser availability
            omniparser_ok = self._omniparser and self._omniparser.available
            if self._omniparser and not omniparser_ok:
                logger.warning(
                    "OmniParser not available. Running without element detection."
                )

            # Build system prompt
            system_prompt = build_system_prompt(
                screen_info,
                self._game_profile,
                omniparser_enabled=omniparser_ok,
            )

            if extra_context:
                system_prompt += f"\n\n## Additional Context\n{extra_context}"

            # Initialize conversation client
            self._conversation = ConversationClient(
                api_key=self._config.anthropic.api_key,
                model=self._config.anthropic.model,
                max_tokens=self._config.anthropic.max_tokens,
                tools=TOOL_DEFINITIONS,
                system_prompt=system_prompt,
                max_images=self._config.conversation.max_images,
            )

            # Launch app
            app_package = self._game_profile.get("app_package")
            if app_package:
                self._device.launch_app(app_package)
                time.sleep(2)

            if self._config.save_screenshots or self._config.save_annotated:
                self._screenshot_dir.mkdir(parents=True, exist_ok=True)

            max_steps = self._config.loop.max_steps
            logger.info(
                f"Starting game loop: profile={self._config.game_profile}, "
                f"omniparser={'ON' if omniparser_ok else 'OFF'}, "
                f"max_steps={max_steps}"
            )

            # Initial screenshot + send to Claude
            self._send_initial_screenshot()

            # Main loop: Claude returns tool calls, we execute and feed back
            while self._running and self._step < max_steps:
                try:
                    self._run_step()
                    self._consecutive_errors = 0
                except DeviceError as e:
                    self._handle_error(f"Device error: {e}")
                    if not self._device.is_connected():
                        logger.error("Device disconnected. Stopping.")
                        break
                except VisionError as e:
                    self._handle_error(f"Vision error: {e}")

                if self._consecutive_errors >= self._config.loop.error_threshold:
                    logger.error(
                        f"Reached error threshold "
                        f"({self._config.loop.error_threshold}). Stopping."
                    )
                    break

        finally:
            signal.signal(signal.SIGINT, original_sigint)
            self._print_summary()

    def _send_initial_screenshot(self) -> None:
        """Capture the first screenshot and send it to Claude."""
        self._check_foreground_app()
        screenshot = self._capture_stable_screenshot()
        parse_result = self._parse_screenshot(screenshot)
        self._elements = parse_result.elements

        self._save_screenshot(
            screenshot, f"step_{self._step:04d}",
            annotated=parse_result.annotated_image,
        )

        screen_h = self._device.screen_info.height
        elements_text = OmniParserClient.format_elements_text(
            self._elements, screen_height=screen_h
        )
        self._last_response = self._conversation.send_screenshot(
            parse_result.annotated_image, elements_text
        )

    def _run_step(self) -> None:
        """Execute one step: process Claude's tool call, send result back."""
        response = self._last_response

        # If Claude ended the turn without tool calls, start a fresh observation
        if response.stop_reason != "tool_use":
            if response.text:
                logger.info(f"  Claude: {response.text[:200]}")
                logger.debug(f"  Claude (full): {response.text}")
            text_lower = response.text.lower()
            if "game over" in text_lower or "game ended" in text_lower:
                logger.info("Game over detected in response text.")
                self._running = False
                return

            # Re-send a fresh screenshot to keep the conversation going
            time.sleep(self._config.loop.delay_seconds)
            self._send_initial_screenshot()
            return

        # Process tool calls from Claude's response
        tool_results = []

        if response.text:
            logger.debug(f"  Claude text: {response.text}")

        for tool_call in response.tool_calls:
            self._step += 1
            logger.info(
                f"  Step {self._step}: {tool_call.tool_name} "
                f"| {tool_call.reasoning}"
            )
            logger.debug(f"  Tool input: {tool_call.tool_input}")

            # Get pre-action screenshot hash for change detection
            pre_hash = self._last_screenshot_hash

            # Execute the tool
            result = self._execute_tool(tool_call, self._elements)
            logger.debug(f"  Result: {result.get('text_result', '')}")

            if result.get("should_stop"):
                self._running = False
                return

            # Wait for game animations to settle before capturing
            if tool_call.tool_name not in ("wait", "game_over"):
                time.sleep(self._config.loop.delay_seconds)

            # Check foreground app
            self._check_foreground_app()

            # Capture result screenshot
            result_screenshot = self._capture_stable_screenshot()

            # Screen change detection
            post_hash = self._screenshot_hash(result_screenshot)
            if pre_hash is not None and post_hash == pre_hash:
                self._no_change_count += 1
                result["text_result"] += (
                    " WARNING: The screen did NOT change after this action. "
                    "Your target was probably wrong."
                )
                if self._no_change_count >= 3:
                    result["text_result"] += (
                        f" You have made {self._no_change_count} actions with "
                        f"no screen change. Try a completely different approach."
                    )
            else:
                self._no_change_count = 0
            self._last_screenshot_hash = post_hash

            # Parse result screenshot with OmniParser
            result_parse = self._parse_screenshot(result_screenshot)
            self._elements = result_parse.elements

            self._save_screenshot(
                result_screenshot, f"step_{self._step:04d}",
                annotated=result_parse.annotated_image,
            )

            result["image"] = result_parse.annotated_image
            result["elements_text"] = OmniParserClient.format_elements_text(
                self._elements,
                screen_height=self._device.screen_info.height,
            )
            logger.debug(
                f"  Elements detected: {len(self._elements)} "
                f"(filtered text sent to Claude below)"
            )
            logger.debug(f"  {result['elements_text']}")

            tool_results.append(result)

            if self._step >= self._config.loop.max_steps:
                logger.warning(f"Reached max steps ({self._config.loop.max_steps}).")
                break

        # If we hit max_steps mid-batch, submit error results for remaining tool calls
        if self._step >= self._config.loop.max_steps and len(tool_results) < len(response.tool_calls):
            for tc in response.tool_calls[len(tool_results):]:
                tool_results.append({
                    "tool_use_id": tc.tool_use_id,
                    "text_result": "Step limit reached. Stopping.",
                    "is_error": True,
                })

        # Submit all tool results back to Claude
        self._last_response = self._conversation.submit_tool_results(tool_results)

    def _execute_tool(
        self, tool_call: ToolCall, elements: list[UIElement]
    ) -> dict:
        """Execute a tool call on the device. Returns a result dict."""
        name = tool_call.tool_name
        inp = tool_call.tool_input

        if name == "tap_element":
            element_id = inp.get("element_id", -1)
            # Find element by ID
            element = None
            for el in elements:
                if el.id == element_id:
                    element = el
                    break

            if element is None:
                logger.warning(f"Element ID {element_id} not found in detected elements")
                return {
                    "tool_use_id": tool_call.tool_use_id,
                    "text_result": (
                        f"Error: Element ID {element_id} not found. "
                        f"Available IDs: {[e.id for e in elements]}. "
                        f"Use tap(x, y) with coordinates instead."
                    ),
                    "is_error": True,
                }

            logger.info(
                f"  tap_element({element_id}) → "
                f"({element.center_x}, {element.center_y}) "
                f"'{element.content}'"
            )
            action = GameAction(
                action=ActionType.TAP,
                x=element.center_x,
                y=element.center_y,
                reasoning=inp.get("reasoning", ""),
            )
            self._device.execute_action(action)
            return {
                "tool_use_id": tool_call.tool_use_id,
                "text_result": (
                    f"Tapped element [{element_id}] '{element.content}' "
                    f"at ({element.center_x}, {element.center_y}). "
                    f"Here is the resulting screen."
                ),
            }

        elif name == "tap":
            x, y = inp.get("x", 0), inp.get("y", 0)
            action = GameAction(
                action=ActionType.TAP, x=x, y=y,
                reasoning=inp.get("reasoning", ""),
            )
            self._device.execute_action(action)
            return {
                "tool_use_id": tool_call.tool_use_id,
                "text_result": (
                    f"Tapped at ({x}, {y}). Here is the resulting screen."
                ),
            }

        elif name == "swipe":
            action = GameAction(
                action=ActionType.SWIPE,
                x=inp.get("x1", 0),
                y=inp.get("y1", 0),
                x2=inp.get("x2", 0),
                y2=inp.get("y2", 0),
                duration=inp.get("duration", 0.5),
                reasoning=inp.get("reasoning", ""),
            )
            self._device.execute_action(action)
            return {
                "tool_use_id": tool_call.tool_use_id,
                "text_result": (
                    f"Swiped from ({inp.get('x1')},{inp.get('y1')}) "
                    f"to ({inp.get('x2')},{inp.get('y2')}). "
                    f"Here is the resulting screen."
                ),
            }

        elif name == "long_press":
            action = GameAction(
                action=ActionType.LONG_PRESS,
                x=inp.get("x", 0),
                y=inp.get("y", 0),
                duration=inp.get("duration", 1.0),
                reasoning=inp.get("reasoning", ""),
            )
            self._device.execute_action(action)
            return {
                "tool_use_id": tool_call.tool_use_id,
                "text_result": (
                    f"Long pressed at ({inp.get('x')}, {inp.get('y')}) "
                    f"for {inp.get('duration', 1.0)}s. Here is the resulting screen."
                ),
            }

        elif name == "press_key":
            key = inp.get("key", "BACK")
            key_map = {
                "BACK": "KEYCODE_BACK",
                "HOME": "KEYCODE_HOME",
                "ENTER": "KEYCODE_ENTER",
            }
            keycode = key_map.get(key, "KEYCODE_BACK")
            self._device._device.keyevent(keycode)
            logger.info(f"  Pressed {key}")
            return {
                "tool_use_id": tool_call.tool_use_id,
                "text_result": (
                    f"Pressed {key} key. Here is the resulting screen."
                ),
            }

        elif name == "wait":
            wait_secs = inp.get("seconds", 2.0)
            logger.info(f"  Waiting {wait_secs}s: {inp.get('reasoning', '')}")
            time.sleep(wait_secs)
            return {
                "tool_use_id": tool_call.tool_use_id,
                "text_result": (
                    f"Waited {wait_secs} seconds. Here is the resulting screen."
                ),
            }

        elif name == "game_over":
            reason = inp.get("reason", "Game ended")
            logger.info(f"  Game over: {reason}")
            return {
                "tool_use_id": tool_call.tool_use_id,
                "text_result": f"Game over acknowledged: {reason}",
                "should_stop": True,
            }

        else:
            logger.warning(f"  Unknown tool: {name}")
            return {
                "tool_use_id": tool_call.tool_use_id,
                "text_result": f"Unknown tool: {name}",
                "is_error": True,
            }

    def _parse_screenshot(self, screenshot: Image.Image) -> ParseResult:
        """Run OmniParser on a screenshot, or return raw image if disabled."""
        if self._omniparser and self._omniparser.available:
            return self._omniparser.parse(screenshot)
        return ParseResult(annotated_image=screenshot, elements=[])

    def _capture_stable_screenshot(self) -> Image.Image:
        """Capture screenshot, waiting for screen to stabilize."""
        timeout = self._config.conversation.stabilization_timeout
        interval = self._config.conversation.stabilization_interval

        prev_hash = None
        stable_count = 0
        start = time.monotonic()
        screenshot = None

        while time.monotonic() - start < timeout:
            screenshot = self._device.screenshot()
            current_hash = self._screenshot_hash(screenshot)

            if current_hash == prev_hash:
                stable_count += 1
                if stable_count >= 2:
                    return screenshot
            else:
                stable_count = 0

            prev_hash = current_hash
            time.sleep(interval)

        if screenshot is None:
            screenshot = self._device.screenshot()

        logger.debug("Screen stabilization timed out, using latest screenshot")
        return screenshot

    @staticmethod
    def _screenshot_hash(image: Image.Image) -> int:
        """Compute a quick perceptual hash for change detection.

        Crops the bottom 10% of the screen to ignore ad banners that
        cycle independently of game state.
        """
        w, h = image.size
        cropped = image.crop((0, 0, w, int(h * 0.9)))
        thumb = cropped.resize((16, 16)).convert("L")
        return hash(thumb.tobytes())

    def _check_foreground_app(self) -> None:
        """Ensure the correct app is in the foreground."""
        app_package = self._game_profile.get("app_package")
        if not app_package:
            return

        fg = self._device.get_foreground_package()
        if fg and fg != app_package:
            logger.warning(
                f"Wrong app in foreground: {fg} (expected {app_package}). "
                f"Re-launching..."
            )
            self._device.launch_app(app_package)
            time.sleep(2)

    def _save_screenshot(
        self, screenshot: Image.Image, name: str, annotated: Image.Image = None
    ) -> None:
        """Save screenshot(s) to disk if configured."""
        if self._config.save_screenshots:
            path = self._screenshot_dir / f"{name}.jpg"
            screenshot.save(str(path), quality=85)
        if annotated is not None and self._config.save_annotated:
            path = self._screenshot_dir / f"{name}_annotated.jpg"
            annotated.save(str(path), quality=85)

    def _handle_error(self, message: str) -> None:
        self._consecutive_errors += 1
        logger.error(
            f"{message} (consecutive errors: "
            f"{self._consecutive_errors}/{self._config.loop.error_threshold})"
        )

    def _handle_interrupt(self, signum, frame) -> None:
        logger.info("\nInterrupt received. Stopping after current action...")
        self._running = False

    def _print_summary(self) -> None:
        logger.info("=== Session Summary ===")
        logger.info(f"Total steps: {self._step}")
        if self._conversation:
            logger.info(
                f"Total tokens: "
                f"in={self._conversation.total_input_tokens}, "
                f"out={self._conversation.total_output_tokens}"
            )
            logger.info(f"Conversation turns: {self._conversation.turn_count}")
