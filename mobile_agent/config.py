"""
Configuration management for Mobile Agent Library.

Handles API keys, timeouts, and other configurable parameters.
"""

import os
from typing import Optional
from dataclasses import dataclass


@dataclass
class MobileAgentConfig:
    """Configuration class for Mobile Agent."""
    
    # OpenAI API settings
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"  # Updated to current vision model
    max_tokens: int = 1000
    
    # ADB settings
    adb_path: str = "adb"
    screenshot_path: str = "./screenshots"
    screenshot_quality: float = 0.5  # Scale factor for screenshots
    
    # Agent behavior settings
    max_iterations: int = 20  # Maximum steps before stopping
    step_delay: float = 2.0   # Delay between actions in seconds
    screenshot_delay: float = 1.0  # Delay after taking screenshot
    
    # Timeouts
    adb_timeout: int = 30
    openai_timeout: int = 60
    
    def __post_init__(self):
        """Initialize configuration after creation."""
        # Try to get OpenAI API key from environment if not provided
        if not self.openai_api_key:
            self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        
        # Create screenshot directory if it doesn't exist
        os.makedirs(self.screenshot_path, exist_ok=True)
    
    def validate(self) -> bool:
        """Validate that required configuration is present."""
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass it to config.")
        
        return True


def create_default_config(
    openai_api_key: Optional[str] = None,
    adb_path: str = "adb",
    **kwargs
) -> MobileAgentConfig:
    """Create a default configuration with optional overrides."""
    config = MobileAgentConfig(
        openai_api_key=openai_api_key or "",
        adb_path=adb_path,
        **kwargs
    )
    config.validate()
    return config 