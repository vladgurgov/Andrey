"""
Action Executor for Mobile Agent.

Executes actions determined by the vision analyzer on the Android device.
"""

import logging
import time
from typing import Dict, Optional

from .adb_controller import ADBController
from .config import MobileAgentConfig


class ActionExecutor:
    """Executes actions on Android device based on vision analyzer output."""
    
    def __init__(self, adb_controller: ADBController, config: MobileAgentConfig):
        """Initialize action executor with ADB controller."""
        self.adb_controller = adb_controller
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def execute_action(self, action_plan: Dict) -> bool:
        """
        Execute an action based on the action plan from vision analyzer.
        
        Args:
            action_plan: Dictionary containing action details
            
        Returns:
            True if action was executed successfully, False otherwise
        """
        action = action_plan.get("action", "").upper()
        reasoning = action_plan.get("reasoning", "No reasoning provided")
        confidence = action_plan.get("confidence", 0.0)
        
        self.logger.info(f"Executing action: {action}")
        self.logger.info(f"Reasoning: {reasoning}")
        self.logger.info(f"Confidence: {confidence}")
        
        try:
            if action == "TAP":
                return self._execute_tap(action_plan)
            elif action == "TYPE":
                return self._execute_type(action_plan)
            elif action == "SWIPE":
                return self._execute_swipe(action_plan)
            elif action == "BACK":
                return self._execute_back()
            elif action == "HOME":
                return self._execute_home()
            elif action == "WAIT":
                return self._execute_wait(action_plan)
            elif action == "COMPLETE":
                self.logger.info("Task completed successfully!")
                return True
            elif action == "ERROR":
                self.logger.error(f"Vision analyzer reported error: {reasoning}")
                return False
            else:
                self.logger.error(f"Unknown action: {action}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error executing action {action}: {e}")
            return False
    
    def _execute_tap(self, action_plan: Dict) -> bool:
        """Execute tap action."""
        coordinates = action_plan.get("coordinates")
        if not coordinates or len(coordinates) != 2:
            self.logger.error("Invalid coordinates for tap action")
            return False
        
        x, y = coordinates
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            self.logger.error(f"Invalid coordinate types: {type(x)}, {type(y)}")
            return False
        
        # Ensure coordinates are positive
        if x < 0 or y < 0:
            self.logger.error(f"Negative coordinates not allowed: ({x}, {y})")
            return False
        
        self.adb_controller.tap(int(x), int(y))
        return True
    
    def _execute_type(self, action_plan: Dict) -> bool:
        """Execute type action."""
        text = action_plan.get("text")
        if not text:
            self.logger.error("No text provided for type action")
            return False
        
        if not isinstance(text, str):
            self.logger.error(f"Text must be string, got {type(text)}")
            return False
        
        # Limit text length for safety
        if len(text) > 500:
            text = text[:500]
            self.logger.warning("Text truncated to 500 characters")
        
        self.adb_controller.type_text(text)
        return True
    
    def _execute_swipe(self, action_plan: Dict) -> bool:
        """Execute swipe action."""
        swipe_coords = action_plan.get("swipe_coords")
        if not swipe_coords or len(swipe_coords) != 2:
            self.logger.error("Invalid swipe coordinates")
            return False
        
        start_coords, end_coords = swipe_coords
        if (len(start_coords) != 2 or len(end_coords) != 2 or
            not all(isinstance(c, (int, float)) for c in start_coords + end_coords)):
            self.logger.error("Invalid swipe coordinate format")
            return False
        
        x1, y1 = start_coords
        x2, y2 = end_coords
        
        # Ensure coordinates are positive
        if any(c < 0 for c in [x1, y1, x2, y2]):
            self.logger.error(f"Negative coordinates not allowed: {swipe_coords}")
            return False
        
        # Get duration if specified, otherwise use default
        duration = action_plan.get("duration", 500)
        
        self.adb_controller.swipe(int(x1), int(y1), int(x2), int(y2), duration)
        return True
    
    def _execute_back(self) -> bool:
        """Execute back button press."""
        self.adb_controller.press_back()
        return True
    
    def _execute_home(self) -> bool:
        """Execute home button press."""
        self.adb_controller.press_home()
        return True
    
    def _execute_wait(self, action_plan: Dict) -> bool:
        """Execute wait action."""
        # Get wait duration from action plan, default to step delay
        wait_duration = action_plan.get("duration", self.config.step_delay)
        
        # Ensure reasonable wait duration (max 10 seconds)
        wait_duration = min(wait_duration, 10.0)
        
        self.logger.info(f"Waiting {wait_duration} seconds...")
        time.sleep(wait_duration)
        return True
    
    def open_app_by_name(self, app_name: str) -> bool:
        """
        Open an app by its common name.
        
        Args:
            app_name: Common name of the app (e.g., "YouTube", "Chrome")
            
        Returns:
            True if app opening was attempted, False otherwise
        """
        try:
            # Try to open by package name first
            from .vision_analyzer import VisionAnalyzer
            vision_analyzer = VisionAnalyzer(self.config)
            package_name = vision_analyzer.get_app_package_name(app_name)
            
            self.logger.info(f"Attempting to open {app_name} with package {package_name}")
            self.adb_controller.open_app(package_name)
            return True
            
        except Exception as e:
            self.logger.error(f"Error opening app {app_name}: {e}")
            return False
    
    def handle_common_scenarios(self, action_plan: Dict, current_instruction: str) -> bool:
        """
        Handle common scenarios that might require special logic.
        
        Args:
            action_plan: Current action plan
            current_instruction: The instruction being executed
            
        Returns:
            True if scenario was handled, False if normal execution should continue
        """
        action = action_plan.get("action", "").upper()
        reasoning = action_plan.get("reasoning", "").lower()
        
        # Handle app opening requests
        if "open" in current_instruction.lower() and action == "TAP":
            app_names = ["youtube", "chrome", "gmail", "maps", "settings", "camera"]
            for app_name in app_names:
                if app_name in current_instruction.lower():
                    self.logger.info(f"Detected app opening request for {app_name}")
                    # First try normal tap, if that doesn't work, try direct app opening
                    return False  # Let normal execution proceed first
        
        # Handle search scenarios
        if ("search" in reasoning or "type" in reasoning) and action == "TYPE":
            # Add small delay before typing to ensure input field is ready
            time.sleep(0.5)
        
        # Handle loading scenarios
        if "loading" in reasoning or "wait" in reasoning:
            # Increase wait time for loading scenarios
            if action == "WAIT" and "duration" not in action_plan:
                action_plan["duration"] = self.config.step_delay * 2
        
        return False  # Continue with normal execution
    
    def get_execution_summary(self, actions_taken: list) -> Dict:
        """
        Get a summary of actions executed.
        
        Args:
            actions_taken: List of action plans that were executed
            
        Returns:
            Dictionary with execution summary
        """
        action_counts = {}
        total_actions = len(actions_taken)
        
        for action_plan in actions_taken:
            action = action_plan.get("action", "UNKNOWN")
            action_counts[action] = action_counts.get(action, 0) + 1
        
        return {
            "total_actions": total_actions,
            "action_breakdown": action_counts,
            "last_action": actions_taken[-1] if actions_taken else None
        } 