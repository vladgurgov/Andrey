"""Configuration loading from YAML files with environment variable overrides."""

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel


class AnthropicConfig(BaseModel):
    api_key: str = ""
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 1024
    temperature: float = 0.0


class LoopConfig(BaseModel):
    delay_seconds: float = 1.0
    max_steps: int = 100  # total actions (screenshots) before stopping
    error_threshold: int = 5
    screenshot_resize_width: int = 0  # 0 = no resize, send full resolution


class DeviceConfig(BaseModel):
    serial: Optional[str] = None
    adb_host: str = "127.0.0.1"
    adb_port: int = 5037


class ConversationConfig(BaseModel):
    max_images: int = 8  # max screenshots kept in context window
    stabilization_timeout: float = 2.0  # max seconds to wait for screen to settle
    stabilization_interval: float = 0.3  # check interval during stabilization


class OmniParserConfig(BaseModel):
    enabled: bool = True
    omniparser_path: str = ""  # path to OmniParser repo (auto-detect if empty)
    weights_path: str = ""  # path to weights dir (auto-detect if empty)
    device: str = "mps"  # torch device: mps, cuda, cpu
    box_threshold: float = 0.05  # YOLO detection confidence threshold
    iou_threshold: float = 0.7  # NMS overlap threshold
    use_paddleocr: bool = False  # False = EasyOCR (safer on macOS)


class AppConfig(BaseModel):
    anthropic: AnthropicConfig = AnthropicConfig()
    loop: LoopConfig = LoopConfig()
    conversation: ConversationConfig = ConversationConfig()
    omniparser: OmniParserConfig = OmniParserConfig()
    device: DeviceConfig = DeviceConfig()
    game_profile: str = "default"
    screenshot_dir: str = "./screenshots"
    save_screenshots: bool = True
    save_annotated: bool = False  # save OmniParser annotated screenshots (debug)


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Load config from YAML file, with env var overrides for the API key."""
    config_data = {}

    if config_path:
        path = Path(config_path)
    else:
        path = Path("config.yaml")

    if path.exists():
        with open(path) as f:
            config_data = yaml.safe_load(f) or {}

    config = AppConfig(**config_data)

    env_key = os.environ.get("ANTHROPIC_API_KEY")
    if env_key:
        config.anthropic.api_key = env_key

    return config
