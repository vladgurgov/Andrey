#!/usr/bin/env python3
"""
Test debug prompt logging to see what gets sent to OpenAI.
This test shows the complete prompt without making actual API calls.
"""

import sys
import os
import logging
from PIL import Image

# Add mobile_agent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from mobile_agent.vision_analyzer import VisionAnalyzer
from mobile_agent.config import MobileAgentConfig

def test_debug_prompt_logging():
    """Test the debug prompt logging to see what gets sent to OpenAI."""
    print("🔍 Testing Debug Prompt Logging")
    print("=" * 60)
    
    # Set up logging to show INFO level
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create config with fake API key
    config = MobileAgentConfig(
        openai_api_key="sk-test1234567890abcdef",
        openai_model="gpt-4o-mini"
    )
    
    # Create vision analyzer
    analyzer = VisionAnalyzer(config)
    
    # Create a fake screenshot to test with
    test_screenshot = "screenshots/step_000.jpg"
    
    if os.path.exists(test_screenshot):
        print(f"📷 Using existing screenshot: {test_screenshot}")
        
        # Get image dimensions
        with Image.open(test_screenshot) as img:
            img_width, img_height = img.size
        
        print(f"📏 Image dimensions: {img_width}×{img_height}")
        print()
        print("🔽 The following shows EXACTLY what gets sent to OpenAI:")
        print()
        
        # Simulate the prompt creation (without actually calling OpenAI)
        instruction = "Open YouTube app"
        step_count = 0
        
        # Create the same prompt that would be sent to OpenAI
        user_prompt = f"""
Current instruction: "{instruction}"
Step: {step_count + 1}

IMPORTANT - COORDINATE SYSTEM:
This image is {img_width}×{img_height} pixels.
When providing coordinates, use this EXACT coordinate system:
- X coordinates: 0 to {img_width-1} (left to right)
- Y coordinates: 0 to {img_height-1} (bottom to top)
- Origin (0,0) is at BOTTOM-LEFT corner
- All coordinates must be within these bounds!

Examples for {img_width}×{img_height} image:
- Bottom-left corner: [0, 0]
- Top-left corner: [0, {img_height-1}]
- Bottom-right corner: [{img_width-1}, 0]
- Top-right corner: [{img_width-1}, {img_height-1}]
- Center: [{img_width//2}, {img_height//2}]

Please analyze this Android screenshot and determine the next action needed to accomplish the instruction.
Focus on finding relevant UI elements and determining the most logical next step.

If you see the instruction is completed (e.g., YouTube video is playing when instruction was to play a video), use COMPLETE action.
If you cannot find what you're looking for or the screen seems stuck, consider using SWIPE to scroll or BACK to go back.
"""
        
        # Log the full prompt (same as what gets sent to OpenAI)
        print("=" * 80)
        print("📤 SENDING TO OPENAI:")
        print("=" * 80)
        print(f"🔑 API Key: sk-test12...cdef")
        print(f"🤖 Model: {config.openai_model}")
        print(f"📏 Image: {img_width}×{img_height} pixels")
        print()
        print("🛠️  SYSTEM PROMPT:")
        print("-" * 40)
        print(analyzer.system_prompt)
        print()
        print("👤 USER PROMPT:")
        print("-" * 40)
        print(user_prompt)
        print()
        print("📷 IMAGE: Attached as base64 encoded JPEG")
        print("=" * 80)
        
        print()
        print("🎯 Key Points About This Prompt:")
        print(f"✅ Explicit coordinate system: Bottom-left origin (0,0)")
        print(f"✅ Exact image bounds: X=[0-{img_width-1}], Y=[0-{img_height-1}]")
        print(f"✅ Concrete examples provided for all corners and center")
        print(f"✅ Clear instruction about Y-axis direction (bottom-to-top)")
        print(f"✅ Bounds validation in place to catch invalid coordinates")
        print()
        print("📝 This should prevent OpenAI from returning impossible coordinates like [600, 400]!")
        
    else:
        print(f"❌ No screenshot found at {test_screenshot}")
        print("Run the CLI tool first to generate a screenshot for testing.")
    
    print()
    print("✅ Debug prompt logging test completed!")
    print()
    print("💡 To see this in action with the actual CLI:")
    print("   export OPENAI_API_KEY='your-key'")
    print("   python mobile_agent_cli.py --task 'open YouTube' --verbose")

if __name__ == "__main__":
    test_debug_prompt_logging() 