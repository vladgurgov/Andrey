"""
Vision Analyzer using OpenAI Vision API.

Analyzes screenshots and determines the next action to take based on the current instruction.
"""

import base64
import json
import logging
from typing import Dict, List, Optional, Tuple
import openai
from openai import OpenAI
from PIL import Image

from .config import MobileAgentConfig


class VisionAnalyzer:
    """Analyzer that uses OpenAI Vision API to understand screenshots and plan actions."""
    
    def __init__(self, config: MobileAgentConfig):
        """Initialize the vision analyzer with OpenAI client."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=config.openai_api_key)
        
        # System prompt for the vision model
        self.system_prompt = """You are Andrey, an AI assistant that controls Android devices by analyzing screenshots and providing precise actions.

Your job is to:
1. Analyze the current screenshot of an Android device
2. Understand the current state and what's visible on screen
3. Determine the next action needed to accomplish the given instruction
4. Provide specific coordinates and actions in a structured format

CRITICAL - COORDINATE SYSTEM:
- You will be told the exact image dimensions in each request
- ALWAYS use coordinates within the specified bounds
- Origin (0,0) is at BOTTOM-LEFT corner
- X increases left-to-right, Y increases bottom-to-top
- Check your coordinates are valid before responding!

Available actions:
- TAP: Tap at specific x,y coordinates
- TYPE: Type text into input fields
- SWIPE: Swipe from one point to another (for scrolling)
- BACK: Press back button
- HOME: Go to home screen
- WAIT: Wait and observe (if loading or transitioning)
- COMPLETE: Task is finished successfully
- ERROR: Cannot proceed (explain why)

Response format (JSON):
{
    "action": "TAP|TYPE|SWIPE|BACK|HOME|WAIT|COMPLETE|ERROR",
    "coordinates": [x, y],  // for TAP - MUST be within image bounds!
    "text": "text to type",  // for TYPE
    "swipe_coords": [[x1, y1], [x2, y2]],  // for SWIPE - MUST be within image bounds!
    "reasoning": "Explain what you see and why you chose this action",
    "confidence": 0.0-1.0  // How confident you are in this action
}

Be precise with coordinates. Look carefully at UI elements like buttons, text fields, icons, etc.
Always explain your reasoning clearly."""

    def analyze_screenshot(self, screenshot_path: str, instruction: str, step_count: int = 0) -> Dict:
        """
        Analyze a screenshot and determine the next action.
        
        Args:
            screenshot_path: Path to the screenshot image
            instruction: The overall instruction being executed
            step_count: Current step number for context
            
        Returns:
            Dictionary containing the action plan
        """
        try:
            # Get image dimensions
            with Image.open(screenshot_path) as img:
                img_width, img_height = img.size
            
            # Encode image to base64
            image_base64 = self._encode_image(screenshot_path)
            
            # Create the prompt with context and explicit coordinate system
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

            # Debug: Show API key being used (partially masked for security)
            api_key_preview = f"{self.config.openai_api_key[:8]}...{self.config.openai_api_key[-4:]}" if len(self.config.openai_api_key) > 12 else "***masked***"
            
            # Debug: Log the full prompt being sent to OpenAI
            self.logger.info("=" * 80)
            self.logger.info("📤 SENDING TO OPENAI:")
            self.logger.info("=" * 80)
            self.logger.info(f"🔑 API Key: {api_key_preview}")
            self.logger.info(f"🤖 Model: {self.config.openai_model}")
            self.logger.info(f"📏 Image: {img_width}×{img_height} pixels")
            self.logger.info("")
            self.logger.info("🛠️  SYSTEM PROMPT:")
            self.logger.info("-" * 40)
            self.logger.info(self.system_prompt)
            self.logger.info("")
            self.logger.info("👤 USER PROMPT:")
            self.logger.info("-" * 40)
            self.logger.info(user_prompt)
            self.logger.info("")
            self.logger.info("📷 IMAGE: Attached as base64 encoded JPEG")
            self.logger.info("=" * 80)
            
            # Call OpenAI Vision API
            response = self.client.chat.completions.create(
                model=self.config.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": self.system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": user_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=self.config.max_tokens,
                temperature=0.1  # Low temperature for more consistent responses
            )
            
            # Parse the response
            response_text = response.choices[0].message.content.strip()
            
            # Debug: Log the OpenAI response clearly
            self.logger.info("")
            self.logger.info("=" * 80)
            self.logger.info("📥 OPENAI RESPONSE:")
            self.logger.info("=" * 80)
            self.logger.info(response_text)
            self.logger.info("=" * 80)
            
            # Try to parse JSON response
            try:
                action_plan = json.loads(response_text)
                # Validate the action plan with image dimensions
                if not self._validate_action_plan(action_plan, img_width, img_height):
                    raise ValueError("Invalid action plan format")
                return action_plan
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract action from text
                self.logger.warning("Failed to parse JSON response, attempting text extraction")
                return self._extract_action_from_text(response_text)
                
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Error analyzing screenshot: {e}")
            
            # Provide specific guidance for common issues
            if "model_not_found" in error_msg or "deprecated" in error_msg.lower():
                reasoning = f"OpenAI model error: {error_msg}. Try updating to 'gpt-4o' or 'gpt-4o-mini' in config."
            elif "insufficient_quota" in error_msg or "billing" in error_msg.lower():
                reasoning = f"OpenAI billing/quota issue: {error_msg}. Check your OpenAI account."
            else:
                reasoning = f"Failed to analyze screenshot: {error_msg}"
            
            return {
                "action": "ERROR",
                "reasoning": reasoning,
                "confidence": 0.0
            }
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 string."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            self.logger.error(f"Error encoding image: {e}")
            raise
    
    def _validate_action_plan(self, action_plan: Dict, img_width: int = None, img_height: int = None) -> bool:
        """Validate that the action plan has required fields and valid coordinates."""
        if not isinstance(action_plan, dict):
            return False
        
        if "action" not in action_plan:
            return False
        
        valid_actions = ["TAP", "TYPE", "SWIPE", "BACK", "HOME", "WAIT", "COMPLETE", "ERROR"]
        if action_plan["action"] not in valid_actions:
            return False
        
        # Check action-specific requirements
        action = action_plan["action"]
        if action == "TAP":
            if "coordinates" not in action_plan:
                return False
            # Validate coordinates are within image bounds
            if img_width is not None and img_height is not None:
                coords = action_plan["coordinates"]
                if not (isinstance(coords, list) and len(coords) == 2):
                    return False
                x, y = coords
                if not (0 <= x < img_width and 0 <= y < img_height):
                    self.logger.warning(f"Invalid TAP coordinates [{x}, {y}] for image {img_width}×{img_height}")
                    return False
        elif action == "TYPE" and "text" not in action_plan:
            return False
        elif action == "SWIPE":
            if "swipe_coords" not in action_plan:
                return False
            # Validate swipe coordinates are within image bounds
            if img_width is not None and img_height is not None:
                coords = action_plan["swipe_coords"]
                if not (isinstance(coords, list) and len(coords) == 2):
                    return False
                for coord_pair in coords:
                    if not (isinstance(coord_pair, list) and len(coord_pair) == 2):
                        return False
                    x, y = coord_pair
                    if not (0 <= x < img_width and 0 <= y < img_height):
                        self.logger.warning(f"Invalid SWIPE coordinates [{x}, {y}] for image {img_width}×{img_height}")
                        return False
        
        return True
    
    def _extract_action_from_text(self, text: str) -> Dict:
        """Extract action information from plain text response."""
        text_lower = text.lower()
        
        # Default error response
        default_response = {
            "action": "WAIT",
            "reasoning": "Could not parse clear action from response, waiting to observe",
            "confidence": 0.3
        }
        
        # Try to identify action type
        if "tap" in text_lower or "click" in text_lower:
            # Try to extract coordinates
            import re
            coord_match = re.search(r'(\d+)[,\s]+(\d+)', text)
            if coord_match:
                return {
                    "action": "TAP",
                    "coordinates": [int(coord_match.group(1)), int(coord_match.group(2))],
                    "reasoning": f"Extracted tap action from text: {text[:200]}",
                    "confidence": 0.6
                }
        elif "type" in text_lower or "enter" in text_lower:
            return {
                "action": "TYPE",
                "text": "search",  # Default text
                "reasoning": f"Detected type action from text: {text[:200]}",
                "confidence": 0.5
            }
        elif "swipe" in text_lower or "scroll" in text_lower:
            return {
                "action": "SWIPE",
                "swipe_coords": [[400, 800], [400, 400]],  # Default scroll up
                "reasoning": f"Detected swipe action from text: {text[:200]}",
                "confidence": 0.5
            }
        elif "back" in text_lower:
            return {
                "action": "BACK",
                "reasoning": f"Detected back action from text: {text[:200]}",
                "confidence": 0.7
            }
        elif "complete" in text_lower or "finished" in text_lower or "done" in text_lower:
            return {
                "action": "COMPLETE",
                "reasoning": f"Task appears complete based on text: {text[:200]}",
                "confidence": 0.8
            }
        
        return default_response
    
    def get_app_package_name(self, app_name: str) -> str:
        """
        Get the package name for common apps.
        
        Args:
            app_name: Common name of the app
            
        Returns:
            Package name string
        """
        app_packages = {
            "youtube": "com.google.android.youtube",
            "chrome": "com.android.chrome",
            "gmail": "com.google.android.gm",
            "maps": "com.google.android.apps.maps",
            "settings": "com.android.settings",
            "camera": "com.android.camera2",
            "photos": "com.google.android.apps.photos",
            "play store": "com.android.vending",
            "calculator": "com.android.calculator2",
            "contacts": "com.android.contacts",
            "phone": "com.android.dialer",
            "messages": "com.google.android.apps.messaging",
            "calendar": "com.google.android.calendar",
            "clock": "com.google.android.deskclock",
            "files": "com.google.android.documentsui"
        }
        
        app_name_lower = app_name.lower()
        for key, package in app_packages.items():
            if key in app_name_lower:
                return package
        
        # If not found, return a generic package name format
        return f"com.{app_name_lower.replace(' ', '').replace('-', '.')}" 