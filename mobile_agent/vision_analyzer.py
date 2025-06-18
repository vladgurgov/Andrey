"""
Vision Analyzer using OpenAI Vision API.

Analyzes screenshots and determines the next action to take based on the current instruction.
"""

import base64
import io
import json
import logging
import os
import random
from typing import Dict, List, Optional, Tuple
import openai
from openai import OpenAI
from PIL import Image, ImageDraw

from .config import MobileAgentConfig


class VisionAnalyzer:
    """Analyzer that uses OpenAI Vision API to understand screenshots and plan actions."""
    
    def __init__(self, config: MobileAgentConfig):
        """Initialize the vision analyzer with OpenAI client."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=config.openai_api_key)
        
        # Track conversation history for context
        self.conversation_history: List[Dict] = []
        
        # Track OpenAI message history for true conversation context
        self.openai_messages: List[Dict] = []
        
        # Initialize scaling attributes
        self.current_scale_factor = 1.0
        self.original_dimensions = (0, 0)
        self.scaled_dimensions = (0, 0)
        
        # Initialize grid attributes
        self.grid_size = 20  # 20px grid cells
        self.grid_cells_x = 0
        self.grid_cells_y = 0
        
        # System prompt for the vision model
        self.system_prompt = """You are Andrey, an AI assistant that controls Android devices by analyzing screenshots and providing precise actions.

Your job is to:
1. Analyze the current screenshot of an Android device
2. Understand the current state and what's visible on screen
3. Determine the next action needed to accomplish the given instruction
4. Provide specific coordinates and actions in a structured format
5. MAINTAIN CONTEXT across all previous steps - this is not an isolated screenshot analysis

CONTEXT AWARENESS:
- You will be provided with the history of all previous actions and their outcomes
- Use this context to understand the progression and avoid repeating failed actions
- Learn from previous steps to make better decisions
- If a previous action didn't achieve the expected result, try a different approach
- Remember what you've already tried and what worked or didn't work

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
    "coordinates": [cell_x, cell_y],  // for TAP - GRID CELL coordinates (NOT pixels!)
    "text": "text to type",  // for TYPE
    "swipe_coords": [[cell_x1, cell_y1], [cell_x2, cell_y2]],  // for SWIPE - GRID CELL coordinates!
    "reasoning": "Explain what you see and why you chose this action",
    "confidence": 0.0-1.0  // How confident you are in this action
}

Be precise with coordinates. Look carefully at UI elements like buttons, text fields, icons, etc.
Always explain your reasoning clearly."""

    def analyze_screenshot(self, screenshot_path: str, instruction: str, step_count: int = 0, previous_action: Dict = None) -> Dict:
        """
        Analyze a screenshot and determine the next action.
        
        Args:
            screenshot_path: Path to the screenshot image
            instruction: The overall instruction being executed
            step_count: Current step number for context
            previous_action: The previous action taken and its outcome
            
        Returns:
            Dictionary containing the action plan
        """
        try:
            # Add previous action to conversation history if provided
            if previous_action:
                self.conversation_history.append({
                    "step": step_count,
                    "action": previous_action,
                    "screenshot_path": screenshot_path
                })
            
            # Get original image dimensions
            with Image.open(screenshot_path) as img:
                original_width, original_height = img.size
            
            # Generate scaled image path for logging
            screenshot_dir = os.path.dirname(screenshot_path)
            screenshot_name = os.path.basename(screenshot_path)
            # Replace .jpg with _scaled.jpg (e.g., step_000.jpg -> step_000_scaled.jpg)
            name_without_ext = os.path.splitext(screenshot_name)[0]
            scaled_image_path = os.path.join(screenshot_dir, f"{name_without_ext}_scaled.jpg")
            
            # Encode image to base64 (this will scale it down and save scaled version)
            image_base64 = self._encode_image(screenshot_path, max_width=200, save_scaled_path=scaled_image_path)
            
            # Calculate scaled dimensions that the LLM will see
            max_width = 200
            if original_width > max_width:
                scale_factor = max_width / original_width
                scaled_width = max_width
                scaled_height = int(original_height * scale_factor)
            else:
                scale_factor = 1.0
                scaled_width = original_width
                scaled_height = original_height
            
            # Note: scaling info and grid information are set in _encode_image method
            
            # Build conversation context
            context_summary = self._build_context_summary()
            
            # Create the prompt with context and explicit coordinate system
            user_prompt = f"""
Current instruction: "{instruction}"
Step: {step_count + 1}

CONVERSATION CONTEXT:
{context_summary}

IMPORTANT - GRID COORDINATE SYSTEM:
This image is {scaled_width}×{scaled_height} pixels with a 20×20 pixel RED GRID OVERLAY for easier coordinate selection.
The grid divides the image into {self.grid_cells_x}×{self.grid_cells_y} cells (each cell is 20×20 pixels).
You can see the red grid lines that create this coordinate system.

COORDINATE SYSTEM - USE GRID CELLS, NOT PIXELS:
- Provide coordinates as GRID CELL positions: [cell_x, cell_y]
- Origin (0,0) is at TOP-LEFT corner (like screen coordinates)
- X coordinates: 0 to {self.grid_cells_x-1} (left to right)
- Y coordinates: 0 to {self.grid_cells_y-1} (top to bottom)
- Each red grid line marks a cell boundary

Examples for {self.grid_cells_x}×{self.grid_cells_y} grid:
- Top-left cell: [0, 0]
- Top-right cell: [{self.grid_cells_x-1}, 0]
- Bottom-left cell: [0, {self.grid_cells_y-1}]
- Bottom-right cell: [{self.grid_cells_x-1}, {self.grid_cells_y-1}]
- Center cell: [{self.grid_cells_x//2}, {self.grid_cells_y//2}]

WHY GRID CELLS: This simulates realistic finger tapping - humans don't tap precise pixels but rather approximate areas. Each cell represents a comfortable finger-tap zone.

Please analyze this Android screenshot and determine the next action needed to accomplish the instruction.
Focus on finding relevant UI elements and determining the most logical next step.

IMPORTANT - RED DOT HANDLING:
If you see a red dot or red circle on the screenshot, this is likely a visual artifact from previous tap actions (showing where the agent tapped before).
- IGNORE red dots when making coordinate decisions
- DO NOT tap on red dots thinking they are UI elements
- Focus on actual app UI elements like buttons, text fields, icons, etc.
- Red dots are just markers from previous steps and should not influence your next action

RETRY STRATEGY:
If you suspect the previous tap action didn't work (same screen appears unchanged), try these approaches:
- Look for alternative tap targets nearby (different parts of buttons, slightly offset coordinates)
- Try tapping on text labels instead of icons, or vice versa
- Consider that some UI elements might need a longer press or might be temporarily disabled
- Check if there are multiple similar elements and try a different one

AD HANDLING:
If you see a full-screen advertisement or popup that's blocking the main interface:
- Look for close buttons: "X", "×", "Close", "Skip", "Skip Ad"
- These are usually in corners (top-right most common) or bottom of screen
- Some ads have "Skip" buttons that appear after a few seconds
- If you can't find a close button, try tapping outside the ad area or use BACK action

If you think you accidentally triggered a full-screen ad by tapping the wrong element:
- Don't panic - this is common when navigating mobile apps
- First priority: look for any close/dismiss UI elements on the ad screen
- If no close button is visible, use the BACK action to return to the previous screen
- After closing the ad, re-evaluate the original screen and choose a different tap target

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
            self.logger.info(f"📏 Original: {original_width}×{original_height} → Scaled: {scaled_width}×{scaled_height} (factor: {scale_factor:.3f})")
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
            
            # Build messages with conversation history
            messages = self._build_openai_messages(user_prompt, image_base64)
            
            # Call OpenAI Vision API
            response = self.client.chat.completions.create(
                model=self.config.openai_model,
                messages=messages,
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
            
            # Store assistant response in OpenAI message history
            self.openai_messages.append({
                "role": "assistant",
                "content": response_text
            })
            
            # Try to parse JSON response
            try:
                action_plan = json.loads(response_text)
                # Validate the action plan with grid cell dimensions
                if not self._validate_action_plan(action_plan, self.grid_cells_x, self.grid_cells_y, use_grid=True):
                    raise ValueError("Invalid action plan format")
                
                # Convert grid coordinates to device coordinates for execution
                action_plan = self._convert_grid_to_device_coordinates(action_plan)
                
                # Store this action in conversation history for next step
                self._add_action_to_history(step_count + 1, action_plan, screenshot_path)
                
                return action_plan
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract action from text
                self.logger.warning("Failed to parse JSON response, attempting text extraction")
                action_plan = self._extract_action_from_text(response_text)
                
                # Convert grid coordinates to device coordinates for execution
                action_plan = self._convert_grid_to_device_coordinates(action_plan)
                
                # Store this action in conversation history for next step
                self._add_action_to_history(step_count + 1, action_plan, screenshot_path)
                
                return action_plan
                
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
    
    def _encode_image(self, image_path: str, max_width: int = 200, save_scaled_path: str = None) -> str:
        """
        Scale image down and encode to base64 string.
        
        Args:
            image_path: Path to the image file
            max_width: Maximum width for the scaled image (default: 200px)
            save_scaled_path: Optional path to save the scaled image for logging
            
        Returns:
            Base64 encoded string of the scaled image
        """
        try:
            with Image.open(image_path) as img:
                # Get original dimensions
                original_width, original_height = img.size
                
                # Calculate scaling factor
                if original_width > max_width:
                    scale_factor = max_width / original_width
                    new_width = max_width
                    new_height = int(original_height * scale_factor)
                    
                    # Resize image maintaining aspect ratio
                    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    
                    self.logger.info(f"📏 Scaled image: {original_width}×{original_height} → {new_width}×{new_height} (scale: {scale_factor:.3f})")
                else:
                    # Image is already small enough
                    img_resized = img
                    scale_factor = 1.0
                    new_width = original_width
                    new_height = original_height
                    self.logger.info(f"📏 Image already small: {original_width}×{original_height}, no scaling needed")
                
                # Store scaling info for coordinate conversion
                self.current_scale_factor = scale_factor
                self.original_dimensions = (original_width, original_height)
                self.scaled_dimensions = (new_width, new_height)
                
                # Calculate grid information for the scaled image
                self.grid_size = 20  # 20px grid cells
                self.grid_cells_x = new_width // self.grid_size
                self.grid_cells_y = new_height // self.grid_size
                
                # Convert to RGB if necessary (for JPEG encoding)
                if img_resized.mode in ('RGBA', 'P'):
                    img_resized = img_resized.convert('RGB')
                
                # Add grid overlay for easier coordinate selection
                img_with_grid = self._add_grid_overlay(img_resized, grid_size=20)
                
                # Save scaled image with grid to file for logging if path provided
                if save_scaled_path:
                    try:
                        # Ensure directory exists
                        os.makedirs(os.path.dirname(save_scaled_path), exist_ok=True)
                        
                        # Save the scaled image with grid overlay
                        img_with_grid.save(save_scaled_path, format='JPEG', quality=85, optimize=True)
                        self.logger.info(f"💾 Saved scaled image with grid to: {save_scaled_path}")
                    except Exception as e:
                        self.logger.warning(f"Could not save scaled image to {save_scaled_path}: {e}")
                
                # Save to bytes buffer (send the grid version to OpenAI)
                buffer = io.BytesIO()
                img_with_grid.save(buffer, format='JPEG', quality=85, optimize=True)
                buffer.seek(0)
                
                # Encode to base64
                image_bytes = buffer.getvalue()
                return base64.b64encode(image_bytes).decode('utf-8')
                
        except Exception as e:
            self.logger.error(f"Error encoding/scaling image: {e}")
            raise
    
    def _validate_action_plan(self, action_plan: Dict, max_x: int = None, max_y: int = None, use_grid: bool = False) -> bool:
        """
        Validate that the action plan has required fields and valid coordinates.
        
        Args:
            action_plan: The action plan to validate
            max_x: Maximum X coordinate (pixels or grid cells)
            max_y: Maximum Y coordinate (pixels or grid cells)
            use_grid: Whether coordinates are grid cells (True) or pixels (False)
        """
        if not isinstance(action_plan, dict):
            return False
        
        if "action" not in action_plan:
            return False
        
        valid_actions = ["TAP", "TYPE", "SWIPE", "BACK", "HOME", "WAIT", "COMPLETE", "ERROR"]
        if action_plan["action"] not in valid_actions:
            return False
        
        coord_type = "grid cells" if use_grid else "pixels"
        
        # Check action-specific requirements
        action = action_plan["action"]
        if action == "TAP":
            if "coordinates" not in action_plan:
                return False
            # Validate coordinates are within bounds
            if max_x is not None and max_y is not None:
                coords = action_plan["coordinates"]
                if not (isinstance(coords, list) and len(coords) == 2):
                    return False
                x, y = coords
                if not (0 <= x < max_x and 0 <= y < max_y):
                    self.logger.warning(f"Invalid TAP {coord_type} [{x}, {y}] for bounds {max_x}×{max_y}")
                    return False
        elif action == "TYPE" and "text" not in action_plan:
            return False
        elif action == "SWIPE":
            if "swipe_coords" not in action_plan:
                return False
            # Validate swipe coordinates are within bounds
            if max_x is not None and max_y is not None:
                coords = action_plan["swipe_coords"]
                if not (isinstance(coords, list) and len(coords) == 2):
                    return False
                for coord_pair in coords:
                    if not (isinstance(coord_pair, list) and len(coord_pair) == 2):
                        return False
                    x, y = coord_pair
                    if not (0 <= x < max_x and 0 <= y < max_y):
                        self.logger.warning(f"Invalid SWIPE {coord_type} [{x}, {y}] for bounds {max_x}×{max_y}")
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
            # Try to extract coordinates (assume they might be grid cells)
            import re
            coord_match = re.search(r'(\d+)[,\s]+(\d+)', text)
            if coord_match:
                # Use extracted coordinates as grid cells
                grid_x = max(0, min(int(coord_match.group(1)), getattr(self, 'grid_cells_x', 20) - 1))
                grid_y = max(0, min(int(coord_match.group(2)), getattr(self, 'grid_cells_y', 44) - 1))
                return {
                    "action": "TAP",
                    "coordinates": [grid_x, grid_y],
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
            # Default scroll up using grid cells (center x, from top to bottom for scroll down)
            center_x = getattr(self, 'grid_cells_x', 20) // 2
            top_y = 5  # Near top
            bottom_y = getattr(self, 'grid_cells_y', 44) - 5  # Near bottom
            return {
                "action": "SWIPE",
                "swipe_coords": [[center_x, top_y], [center_x, bottom_y]],  # Default scroll down
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
    
    def _add_grid_overlay(self, image: Image.Image, grid_size: int = 20) -> Image.Image:
        """
        Add a red grid overlay to the image for easier coordinate selection.
        
        Args:
            image: PIL Image to add grid to
            grid_size: Size of each grid cell in pixels (default: 20)
            
        Returns:
            Image with grid overlay
        """
        # Create a copy to avoid modifying original
        img_with_grid = image.copy()
        draw = ImageDraw.Draw(img_with_grid)
        
        width, height = img_with_grid.size
        
        # Draw vertical lines (every grid_size pixels)
        for x in range(0, width, grid_size):
            draw.line([(x, 0), (x, height-1)], fill='red', width=1)
        
        # Draw horizontal lines (every grid_size pixels)  
        for y in range(0, height, grid_size):
            draw.line([(0, y), (width-1, y)], fill='red', width=1)
        
        # Log grid info
        cells_x = width // grid_size
        cells_y = height // grid_size
        self.logger.info(f"🔲 Added {cells_x}×{cells_y} grid overlay ({grid_size}px cells)")
        
        return img_with_grid
    
    def _convert_grid_to_device_coordinates(self, action_plan: Dict) -> Dict:
        """
        Convert grid cell coordinates to device pixel coordinates.
        
        Args:
            action_plan: Action plan with grid cell coordinates
            
        Returns:
            Action plan with device pixel coordinates
        """
        if not hasattr(self, 'current_scale_factor') or not hasattr(self, 'grid_size'):
            # No grid conversion needed
            return action_plan
        
        # Create a copy to avoid modifying the original
        device_plan = action_plan.copy()
        
        action = device_plan.get("action", "").upper()
        
        if action == "TAP" and "coordinates" in device_plan:
            # Convert TAP grid coordinates
            grid_coords = device_plan["coordinates"]
            if isinstance(grid_coords, list) and len(grid_coords) == 2:
                grid_x, grid_y = grid_coords
                
                # Convert grid cell to scaled pixel coordinates with random offset
                pixel_x = self._grid_to_pixel_with_offset(grid_x, self.grid_size)
                pixel_y = self._grid_to_pixel_with_offset(grid_y, self.grid_size)
                
                # Scale up to screenshot coordinates (the input image is already the ADB screenshot)
                # Vision analyzer receives ADB screenshot and scales it down, now scale back up
                # The input image dimensions are the ADB screenshot dimensions
                adb_screenshot_width, adb_screenshot_height = self.original_dimensions
                
                # Scale factor from our processed image back to ADB screenshot
                scale_to_adb_screenshot = adb_screenshot_width / self.scaled_dimensions[0]
                
                screenshot_x = int(pixel_x * scale_to_adb_screenshot)
                screenshot_y = int(pixel_y * scale_to_adb_screenshot)
                
                # Ensure coordinates are within ADB screenshot bounds  
                screenshot_x = max(0, min(screenshot_x, adb_screenshot_width - 1))
                screenshot_y = max(0, min(screenshot_y, adb_screenshot_height - 1))
                
                device_plan["coordinates"] = [screenshot_x, screenshot_y]
                
                self.logger.info(f"🎯 Grid cell [{grid_x}, {grid_y}] → vision pixel [{pixel_x:.1f}, {pixel_y:.1f}] → ADB screenshot [{screenshot_x}, {screenshot_y}]")
                self.logger.info(f"    Scale factor: {scale_to_adb_screenshot:.2f} (vision {self.scaled_dimensions[0]}px → ADB screenshot {adb_screenshot_width}px)")
        
        elif action == "SWIPE" and "swipe_coords" in device_plan:
            # Convert SWIPE grid coordinates
            grid_swipe = device_plan["swipe_coords"]
            if isinstance(grid_swipe, list) and len(grid_swipe) == 2:
                start_grid, end_grid = grid_swipe
                
                if (isinstance(start_grid, list) and len(start_grid) == 2 and
                    isinstance(end_grid, list) and len(end_grid) == 2):
                    
                    # Convert start grid coordinates
                    start_grid_x, start_grid_y = start_grid
                    start_pixel_x = self._grid_to_pixel_with_offset(start_grid_x, self.grid_size)
                    start_pixel_y = self._grid_to_pixel_with_offset(start_grid_y, self.grid_size)
                    start_device_x = int(start_pixel_x / self.current_scale_factor)
                    start_device_y = int(start_pixel_y / self.current_scale_factor)
                    
                    # Convert end grid coordinates
                    end_grid_x, end_grid_y = end_grid
                    end_pixel_x = self._grid_to_pixel_with_offset(end_grid_x, self.grid_size)
                    end_pixel_y = self._grid_to_pixel_with_offset(end_grid_y, self.grid_size)
                    end_device_x = int(end_pixel_x / self.current_scale_factor)
                    end_device_y = int(end_pixel_y / self.current_scale_factor)
                    
                    # Ensure coordinates are within device bounds
                    original_width, original_height = self.original_dimensions
                    start_device_x = max(0, min(start_device_x, original_width - 1))
                    start_device_y = max(0, min(start_device_y, original_height - 1))
                    end_device_x = max(0, min(end_device_x, original_width - 1))
                    end_device_y = max(0, min(end_device_y, original_height - 1))
                    
                    device_plan["swipe_coords"] = [[start_device_x, start_device_y], [end_device_x, end_device_y]]
                    
                    self.logger.info(f"🔄 Grid swipe [{start_grid_x}, {start_grid_y}]→[{end_grid_x}, {end_grid_y}] to device [{start_device_x}, {start_device_y}]→[{end_device_x}, {end_device_y}]")
        
        return device_plan
    
    def _grid_to_pixel_with_offset(self, grid_coord: int, grid_size: int) -> float:
        """
        Convert grid cell coordinate to pixel coordinate with random offset.
        
        Args:
            grid_coord: Grid cell coordinate
            grid_size: Size of each grid cell in pixels
            
        Returns:
            Pixel coordinate with random offset within the cell
        """
        # Calculate the center of the grid cell (top-left origin, no flipping needed)
        cell_center = (grid_coord * grid_size) + (grid_size // 2)
        
        # Add random offset within the cell (±40% of cell size for natural variation)
        max_offset = grid_size * 0.4
        offset = random.uniform(-max_offset, max_offset)
        
        # Calculate bounds for the cell
        min_bound = grid_coord * grid_size
        max_bound = (grid_coord + 1) * grid_size - 1
        
        # Ensure we stay within the cell bounds
        pixel_coord = max(min_bound, min(cell_center + offset, max_bound))
        
        return pixel_coord
    
    def _scale_coordinates_to_original(self, action_plan: Dict) -> Dict:
        """
        Scale coordinates from scaled image back to original device size.
        
        Args:
            action_plan: Action plan with coordinates based on scaled image
            
        Returns:
            Action plan with coordinates scaled to original device size
        """
        if not hasattr(self, 'current_scale_factor') or self.current_scale_factor == 1.0:
            # No scaling was applied, return as-is
            return action_plan
        
        # Create a copy to avoid modifying the original
        scaled_plan = action_plan.copy()
        
        action = scaled_plan.get("action", "").upper()
        
        if action == "TAP" and "coordinates" in scaled_plan:
            # Scale TAP coordinates
            scaled_coords = scaled_plan["coordinates"]
            if isinstance(scaled_coords, list) and len(scaled_coords) == 2:
                x_scaled, y_scaled = scaled_coords
                
                # Scale back to original size
                x_original = int(x_scaled / self.current_scale_factor)
                y_original = int(y_scaled / self.current_scale_factor)
                
                # Ensure coordinates are within device bounds
                original_width, original_height = self.original_dimensions
                x_original = max(0, min(x_original, original_width - 1))
                y_original = max(0, min(y_original, original_height - 1))
                
                scaled_plan["coordinates"] = [x_original, y_original]
                
                self.logger.info(f"🔄 Scaled TAP coordinates: [{x_scaled}, {y_scaled}] → [{x_original}, {y_original}] (factor: {1/self.current_scale_factor:.3f})")
        
        elif action == "SWIPE" and "swipe_coords" in scaled_plan:
            # Scale SWIPE coordinates
            scaled_swipe = scaled_plan["swipe_coords"]
            if isinstance(scaled_swipe, list) and len(scaled_swipe) == 2:
                start_coords, end_coords = scaled_swipe
                
                if (isinstance(start_coords, list) and len(start_coords) == 2 and
                    isinstance(end_coords, list) and len(end_coords) == 2):
                    
                    # Scale start coordinates
                    x1_scaled, y1_scaled = start_coords
                    x1_original = int(x1_scaled / self.current_scale_factor)
                    y1_original = int(y1_scaled / self.current_scale_factor)
                    
                    # Scale end coordinates  
                    x2_scaled, y2_scaled = end_coords
                    x2_original = int(x2_scaled / self.current_scale_factor)
                    y2_original = int(y2_scaled / self.current_scale_factor)
                    
                    # Ensure coordinates are within device bounds
                    original_width, original_height = self.original_dimensions
                    x1_original = max(0, min(x1_original, original_width - 1))
                    y1_original = max(0, min(y1_original, original_height - 1))
                    x2_original = max(0, min(x2_original, original_width - 1))
                    y2_original = max(0, min(y2_original, original_height - 1))
                    
                    scaled_plan["swipe_coords"] = [[x1_original, y1_original], [x2_original, y2_original]]
                    
                    self.logger.info(f"🔄 Scaled SWIPE coordinates: [{x1_scaled}, {y1_scaled}]→[{x2_scaled}, {y2_scaled}] to [{x1_original}, {y1_original}]→[{x2_original}, {y2_original}]")
        
        return scaled_plan
    
    def _build_context_summary(self) -> str:
        """Build a summary of conversation history for context."""
        if not self.conversation_history:
            return "This is the first step - no previous actions taken."
        
        context_lines = []
        context_lines.append(f"Previous {len(self.conversation_history)} step(s) taken:")
        
        for i, entry in enumerate(self.conversation_history[-5:]):  # Show last 5 steps for context
            step_num = entry.get("step", i + 1)
            action = entry.get("action", {})
            
            action_type = action.get("action", "UNKNOWN")
            reasoning = action.get("reasoning", "No reasoning provided")
            
            # Build action description
            action_desc = f"Step {step_num}: {action_type}"
            if action_type == "TAP" and "coordinates" in action:
                coords = action["coordinates"]
                action_desc += f" at [{coords[0]}, {coords[1]}]"
            elif action_type == "TYPE" and "text" in action:
                text = action["text"]
                action_desc += f' text: "{text}"'
            elif action_type == "SWIPE" and "swipe_coords" in action:
                swipe_coords = action["swipe_coords"]
                action_desc += f" from {swipe_coords[0]} to {swipe_coords[1]}"
            
            context_lines.append(f"- {action_desc}")
            context_lines.append(f"  Reasoning: {reasoning[:100]}...")
        
        context_lines.append("")
        context_lines.append("Use this context to:")
        context_lines.append("- Avoid repeating failed actions")
        context_lines.append("- Build upon successful progress")
        context_lines.append("- Try different approaches if previous attempts didn't work")
        context_lines.append("- Understand the current state based on previous actions")
        
        return "\n".join(context_lines)
    
    def _add_action_to_history(self, step: int, action: Dict, screenshot_path: str):
        """Add an action to the conversation history."""
        self.conversation_history.append({
            "step": step,
            "action": action,
            "screenshot_path": screenshot_path
        })
        
        # Keep only last 10 steps to avoid context getting too long
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
    
    def clear_history(self):
        """Clear the conversation history (useful for starting a new task)."""
        self.conversation_history = []
        self.openai_messages = []
        
        # Reset grid attributes for new task
        self.grid_cells_x = 0
        self.grid_cells_y = 0
    
    def _build_openai_messages(self, current_user_prompt: str, current_image_base64: str) -> List[Dict]:
        """
        Build the complete message history for OpenAI API call.
        
        Args:
            current_user_prompt: The current user prompt text
            current_image_base64: Base64 encoded current screenshot
            
        Returns:
            List of messages for OpenAI API
        """
        messages = []
        
        # Always start with system prompt
        messages.append({
            "role": "system",
            "content": self.system_prompt
        })
        
        # Add previous conversation messages
        # For performance and cost reasons, only include images from last 2 steps
        recent_image_count = 0
        max_recent_images = 2
        
        for message in self.openai_messages:
            # Add previous messages, but limit images to avoid token/cost issues
            if message["role"] == "user" and "content" in message:
                content = message["content"]
                
                # If this is a user message with image, check if we should include the image
                if isinstance(content, list) and recent_image_count < max_recent_images:
                    # Include the full message with image
                    messages.append(message)
                    recent_image_count += 1
                elif isinstance(content, list):
                    # Extract just the text portion, skip the image
                    text_content = ""
                    for item in content:
                        if item.get("type") == "text":
                            text_content = item.get("text", "")
                            break
                    if text_content:
                        messages.append({
                            "role": "user",
                            "content": f"[Previous step - image omitted for efficiency]\n{text_content}"
                        })
                else:
                    # Text-only message
                    messages.append(message)
            else:
                # Assistant message - always include
                messages.append(message)
        
        # Add current user message with image
        current_message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": current_user_prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{current_image_base64}",
                        "detail": "high"
                    }
                }
            ]
        }
        
        messages.append(current_message)
        
        # Store current user message in history for next call
        self.openai_messages.append(current_message)
        
        # Keep message history manageable (last 20 messages)
        if len(self.openai_messages) > 20:
            self.openai_messages = self.openai_messages[-20:]
        
        self.logger.info(f"📧 Built message history with {len(messages)} messages ({recent_image_count + 1} with images)")
        
        return messages
    
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