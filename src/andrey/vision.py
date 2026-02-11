"""Claude vision API clients for screenshot analysis."""

import base64
import io
import logging
import time
from typing import Optional

import anthropic
from PIL import Image

from andrey.models import ApiResponse, ToolCall

logger = logging.getLogger(__name__)


class VisionError(Exception):
    """Raised when vision API calls fail."""


class VisionClient:
    """Handles Claude vision API calls for screenshot analysis (legacy single-turn)."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ):
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    def encode_image(
        self, image: Image.Image, fmt: str = "JPEG", quality: int = 85
    ) -> tuple[str, str]:
        """Encode PIL Image to base64. Returns (base64_data, media_type)."""
        buffer = io.BytesIO()
        if fmt.upper() == "JPEG":
            if image.mode == "RGBA":
                image = image.convert("RGB")
            image.save(buffer, format="JPEG", quality=quality)
            media_type = "image/jpeg"
        else:
            image.save(buffer, format="PNG")
            media_type = "image/png"

        base64_data = base64.standard_b64encode(buffer.getvalue()).decode("utf-8")
        logger.debug(
            f"Encoded image: {image.size[0]}x{image.size[1]}, "
            f"format={fmt}, base64 size={len(base64_data)} chars"
        )
        return base64_data, media_type

    def describe_screenshot(self, image: Image.Image) -> str:
        """Simple screenshot description without game context."""
        base64_data, media_type = self.encode_image(image)

        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=0.0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": "Describe what you see on this Android phone screen in detail.",
                        },
                    ],
                }
            ],
        )
        return response.content[0].text


class ConversationClient:
    """Multi-turn conversation client using Claude's tool-use API.

    Manages a running conversation where Claude sees screenshots,
    calls tools (tap, swipe, etc.), and receives results with new
    screenshots. This enables multi-step UI flows.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1024,
        tools: list[dict] = None,
        system_prompt: str = "",
        max_images: int = 8,
    ):
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens
        self._tools = tools or []
        self._system_prompt = system_prompt
        self._messages: list[dict] = []
        self._max_images = max_images
        self._turn_count = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    @property
    def turn_count(self) -> int:
        return self._turn_count

    @property
    def total_input_tokens(self) -> int:
        return self._total_input_tokens

    @property
    def total_output_tokens(self) -> int:
        return self._total_output_tokens

    def send_screenshot(
        self, image: Image.Image, elements_text: str = ""
    ) -> ApiResponse:
        """Send a screenshot as a user message and get Claude's response.

        Args:
            image: The annotated screenshot (with OmniParser bounding boxes if available)
            elements_text: Text list of detected UI elements
        """
        base64_data, media_type = self._encode_image(image)

        user_content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64_data,
                },
            },
        ]

        text_msg = "Here is the current screenshot."
        if elements_text:
            text_msg += f"\n\n{elements_text}"
        text_msg += "\n\nDecide what action to take."

        user_content.append({"type": "text", "text": text_msg})

        self._messages.append({"role": "user", "content": user_content})
        self._turn_count += 1

        return self._call_api()

    def submit_tool_results(
        self,
        results: list[dict],
    ) -> ApiResponse:
        """Submit tool execution results and continue the conversation.

        Args:
            results: List of tool result dicts, each with:
                - tool_use_id: str
                - text_result: str
                - image: Optional[Image.Image] (result screenshot)
                - elements_text: Optional[str] (detected elements in result)
                - is_error: bool
        """
        tool_result_blocks = []

        for result in results:
            content = []

            # Add result screenshot if provided
            if result.get("image"):
                b64, media = self._encode_image(result["image"])
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media,
                            "data": b64,
                        },
                    }
                )

            # Build text result
            text = result.get("text_result", "Action executed.")
            if result.get("elements_text"):
                text += f"\n\n{result['elements_text']}"
            content.append({"type": "text", "text": text})

            tool_result_blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": result["tool_use_id"],
                    "content": content,
                    "is_error": result.get("is_error", False),
                }
            )

        self._messages.append({"role": "user", "content": tool_result_blocks})
        self._turn_count += 1

        return self._call_api()

    def reset(self, summary: Optional[str] = None) -> None:
        """Reset the conversation, optionally with a game state summary."""
        self._messages.clear()
        self._turn_count = 0

        if summary:
            self._messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Previous game context: {summary}",
                        }
                    ],
                }
            )
            # Need a placeholder assistant response to maintain alternation
            self._messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "Understood. I'll continue playing based on this context. Send me the current screenshot.",
                        }
                    ],
                }
            )

        logger.info("Conversation reset" + (f" with summary" if summary else ""))

    def _call_api(self) -> ApiResponse:
        """Make the API call with current conversation state."""
        self._trim_conversation()

        try:
            t0 = time.monotonic()

            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=0.0,
                system=self._system_prompt,
                messages=self._messages,
                tools=self._tools,
            )

            elapsed_ms = (time.monotonic() - t0) * 1000
            self._total_input_tokens += response.usage.input_tokens
            self._total_output_tokens += response.usage.output_tokens

            logger.info(
                f"API response in {elapsed_ms:.0f}ms "
                f"(in={response.usage.input_tokens}, out={response.usage.output_tokens}, "
                f"stop={response.stop_reason})"
            )

            # Append assistant response to conversation (preserve tool_use blocks).
            # Convert ContentBlock pydantic objects to plain dicts to avoid
            # serialization issues when passing back to the API.
            content_dicts = []
            for block in response.content:
                if block.type == "text":
                    content_dicts.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    content_dicts.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": dict(block.input),
                    })
                else:
                    content_dicts.append(block.model_dump())
            self._messages.append(
                {"role": "assistant", "content": content_dicts}
            )

            return self._parse_response(response)

        except anthropic.RateLimitError as e:
            raise VisionError(f"Rate limited: {e}")
        except anthropic.APIError as e:
            # Check for context overflow
            if "context" in str(e).lower() or "token" in str(e).lower():
                logger.warning(f"Context overflow detected: {e}. Resetting conversation.")
                self.reset()
                raise VisionError(f"Context overflow, conversation reset: {e}")
            raise VisionError(f"API error: {e}")

    def _parse_response(self, response) -> ApiResponse:
        """Extract tool calls and text from the API response."""
        tool_calls = []
        text_parts = []

        for block in response.content:
            if block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        tool_use_id=block.id,
                        tool_name=block.name,
                        tool_input=dict(block.input),
                        reasoning=block.input.get("reasoning", ""),
                    )
                )
            elif block.type == "text":
                text_parts.append(block.text)

        return ApiResponse(
            stop_reason=response.stop_reason,
            tool_calls=tool_calls,
            text="\n".join(text_parts),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    def _trim_conversation(self) -> None:
        """Manage context window by removing old images from conversation."""
        if len(self._messages) <= 4:
            return

        # Count messages with images
        image_indices = []
        for i, msg in enumerate(self._messages):
            if self._message_has_image(msg):
                image_indices.append(i)

        # Strip images from oldest messages to stay under limit
        while len(image_indices) > self._max_images:
            oldest_idx = image_indices.pop(0)
            self._strip_images(self._messages[oldest_idx])
            logger.debug(f"Stripped images from message {oldest_idx}")

    def _message_has_image(self, message: dict) -> bool:
        """Check if a message contains image content."""
        content = message.get("content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "image":
                        return True
                    # Check inside tool_result blocks
                    if block.get("type") == "tool_result":
                        inner = block.get("content", [])
                        if isinstance(inner, list):
                            for inner_block in inner:
                                if isinstance(inner_block, dict) and inner_block.get("type") == "image":
                                    return True
        return False

    def _strip_images(self, message: dict) -> None:
        """Replace image blocks with text placeholders."""
        content = message.get("content", [])
        if not isinstance(content, list):
            return

        new_content = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "image":
                    new_content.append(
                        {"type": "text", "text": "[Screenshot removed to save context]"}
                    )
                elif block.get("type") == "tool_result":
                    # Strip images inside tool_result blocks
                    inner = block.get("content", [])
                    if isinstance(inner, list):
                        new_inner = []
                        for inner_block in inner:
                            if isinstance(inner_block, dict) and inner_block.get("type") == "image":
                                new_inner.append(
                                    {"type": "text", "text": "[Screenshot removed]"}
                                )
                            else:
                                new_inner.append(inner_block)
                        block = {**block, "content": new_inner}
                    new_content.append(block)
                else:
                    new_content.append(block)
            else:
                new_content.append(block)

        message["content"] = new_content

    @staticmethod
    def _encode_image(
        image: Image.Image, fmt: str = "JPEG", quality: int = 85
    ) -> tuple[str, str]:
        """Encode PIL Image to base64."""
        buffer = io.BytesIO()
        if fmt.upper() == "JPEG":
            if image.mode == "RGBA":
                image = image.convert("RGB")
            image.save(buffer, format="JPEG", quality=quality)
            media_type = "image/jpeg"
        else:
            image.save(buffer, format="PNG")
            media_type = "image/png"

        return base64.standard_b64encode(buffer.getvalue()).decode("utf-8"), media_type
