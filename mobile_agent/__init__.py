"""
Mobile Agent Library - An AI-powered Android automation tool using ADB and OpenAI Vision API.

This library provides intelligent mobile device automation by combining:
- ADB (Android Debug Bridge) for device control
- OpenAI Vision API for screenshot analysis
- Automated action execution based on natural language instructions

Example:
    from mobile_agent import MobileAgent
    
    agent = MobileAgent(openai_api_key="your-key", adb_path="adb")
    agent.execute_instruction("Open YouTube and play the first video")
"""

from .agent import MobileAgent
from .adb_controller import ADBController
from .vision_analyzer import VisionAnalyzer
from .action_executor import ActionExecutor

__version__ = "1.0.0"
__author__ = "Mobile Agent Team"
__email__ = "support@mobileagent.dev"

__all__ = [
    "MobileAgent",
    "ADBController", 
    "VisionAnalyzer",
    "ActionExecutor",
] 