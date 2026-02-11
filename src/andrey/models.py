"""Data models for actions, responses, and device info."""

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class ActionType(str, Enum):
    TAP = "tap"
    SWIPE = "swipe"
    LONG_PRESS = "long_press"
    KEY = "key"
    TYPE_TEXT = "type_text"
    WAIT = "wait"
    GAME_OVER = "game_over"


class GameAction(BaseModel):
    """Parsed and validated action from LLM response."""

    action: ActionType
    x: Optional[int] = None
    y: Optional[int] = None
    x2: Optional[int] = None
    y2: Optional[int] = None
    duration: Optional[float] = None
    key: Optional[str] = None
    text: Optional[str] = None
    reasoning: str = ""
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)

    @field_validator("x", "y", "x2", "y2")
    @classmethod
    def coordinates_non_negative(cls, v):
        if v is not None and v < 0:
            raise ValueError(f"Coordinate must be non-negative, got {v}")
        return v


class ActionRecord(BaseModel):
    """Recorded action for history tracking."""

    iteration: int
    timestamp: float
    action: GameAction
    screenshot_path: Optional[str] = None
    llm_latency_ms: float = 0.0


class ScreenInfo(BaseModel):
    """Device screen metadata."""

    width: int
    height: int
    density: Optional[int] = None
    rotation: int = 0


# --- Multi-turn tool-use models ---


class ToolCall(BaseModel):
    """A tool call extracted from Claude's API response."""

    tool_use_id: str
    tool_name: str
    tool_input: dict[str, Any]
    reasoning: str = ""


class ToolResult(BaseModel):
    """Result of executing a tool call on the device."""

    tool_use_id: str
    text_result: str = ""
    should_stop: bool = False
    is_error: bool = False

    class Config:
        arbitrary_types_allowed = True


class ApiResponse(BaseModel):
    """Parsed response from a Claude API call."""

    stop_reason: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


# --- Tool definitions for Claude API ---

TOOL_DEFINITIONS: list[dict] = [
    {
        "name": "tap_element",
        "description": (
            "Tap on a detected UI element by its ID number. This is the preferred way "
            "to interact with elements shown in the screenshot with numbered bounding boxes. "
            "The system will tap the center of the element's bounding box."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "element_id": {
                    "type": "integer",
                    "description": "The ID number of the detected element to tap",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of why you are tapping this element",
                },
            },
            "required": ["element_id", "reasoning"],
        },
    },
    {
        "name": "tap",
        "description": (
            "Tap at specific pixel coordinates on the screen. Use this only when "
            "the element you need is NOT in the detected elements list, or when you "
            "need to tap a specific location that wasn't detected as an element. "
            "Coordinates are in pixels with (0,0) at top-left."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "x": {
                    "type": "integer",
                    "description": "X coordinate in pixels (0 = left edge)",
                },
                "y": {
                    "type": "integer",
                    "description": "Y coordinate in pixels (0 = top edge)",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of why you are tapping here",
                },
            },
            "required": ["x", "y", "reasoning"],
        },
    },
    {
        "name": "swipe",
        "description": (
            "Swipe from one point to another on the screen. "
            "Use for scrolling, dragging, or gesture-based interactions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "x1": {"type": "integer", "description": "Start X coordinate"},
                "y1": {"type": "integer", "description": "Start Y coordinate"},
                "x2": {"type": "integer", "description": "End X coordinate"},
                "y2": {"type": "integer", "description": "End Y coordinate"},
                "duration": {
                    "type": "number",
                    "description": "Duration of swipe in seconds (default 0.5)",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Why you are swiping",
                },
            },
            "required": ["x1", "y1", "x2", "y2", "reasoning"],
        },
    },
    {
        "name": "long_press",
        "description": "Long press on a specific point on the screen.",
        "input_schema": {
            "type": "object",
            "properties": {
                "x": {"type": "integer", "description": "X coordinate"},
                "y": {"type": "integer", "description": "Y coordinate"},
                "duration": {
                    "type": "number",
                    "description": "Duration in seconds (default 1.0)",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Why you are long pressing",
                },
            },
            "required": ["x", "y", "reasoning"],
        },
    },
    {
        "name": "press_key",
        "description": (
            "Press an Android system key. "
            "Use for BACK (dismiss dialogs), HOME, or ENTER."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "enum": ["BACK", "HOME", "ENTER"],
                    "description": "The key to press",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Why you are pressing this key",
                },
            },
            "required": ["key", "reasoning"],
        },
    },
    {
        "name": "wait",
        "description": (
            "Wait without taking any action. Use when: "
            "an animation is playing, a loading screen is shown, "
            "or it is NOT your turn to play. "
            "Do NOT use this when the game is waiting for YOUR action."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "seconds": {
                    "type": "number",
                    "description": "Seconds to wait (default 2.0)",
                },
                "reasoning": {
                    "type": "string",
                    "description": "What you are waiting for",
                },
            },
            "required": ["reasoning"],
        },
    },
    {
        "name": "game_over",
        "description": (
            "Signal that the game has ended. Use when you see a final score screen, "
            "game over message, or the game has clearly concluded."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Description of the final game state",
                },
            },
            "required": ["reason"],
        },
    },
]
