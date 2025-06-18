"""
Main Mobile Agent class.

Orchestrates the ADB controller, vision analyzer, and action executor to perform
automated tasks on Android devices.
"""

import logging
import time
from typing import Dict, List, Optional, Callable
import os

from .config import MobileAgentConfig, create_default_config
from .adb_controller import ADBController
from .vision_analyzer import VisionAnalyzer
from .action_executor import ActionExecutor


class MobileAgent:
    """
    Main Mobile Agent class for Android device automation.
    
    Uses OpenAI Vision API to analyze screenshots and control Android devices
    through ADB commands.
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        adb_path: str = "adb",
        config: Optional[MobileAgentConfig] = None,
        **kwargs
    ):
        """
        Initialize Mobile Agent.
        
        Args:
            openai_api_key: OpenAI API key (can also be set via OPENAI_API_KEY env var)
            adb_path: Path to ADB executable
            config: Custom configuration object
            **kwargs: Additional configuration parameters
        """
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Create configuration
        if config is None:
            config = create_default_config(
                openai_api_key=openai_api_key,
                adb_path=adb_path,
                **kwargs
            )
        self.config = config
        
        # Initialize components
        self.adb_controller = ADBController(config)
        self.vision_analyzer = VisionAnalyzer(config)
        self.action_executor = ActionExecutor(self.adb_controller, config)
        
        # Execution state
        self.current_instruction = ""
        self.actions_taken = []
        self.screenshots_taken = []
        self.is_running = False
        
        self.logger.info("Mobile Agent initialized successfully")
        
        # Get device info for logging
        device_info = self.adb_controller.get_device_info()
        self.logger.info(f"Connected device: {device_info}")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('mobile_agent.log')
            ]
        )
    
    def execute_instruction(
        self,
        instruction: str,
        max_steps: Optional[int] = None,
        progress_callback: Optional[Callable] = None
    ) -> Dict:
        """
        Execute a natural language instruction on the Android device.
        
        Args:
            instruction: Natural language instruction (e.g., "Open YouTube and play first video")
            max_steps: Maximum number of steps to execute (default from config)
            progress_callback: Optional callback function to report progress
            
        Returns:
            Dictionary with execution results
        """
        if not instruction or not instruction.strip():
            raise ValueError("Instruction cannot be empty")
        
        self.current_instruction = instruction.strip()
        self.actions_taken = []
        self.screenshots_taken = []
        self.is_running = True
        
        # Clear conversation history for new instruction
        self.vision_analyzer.clear_history()
        
        max_steps = max_steps or self.config.max_iterations
        
        self.logger.info(f"Starting execution of instruction: '{self.current_instruction}'")
        self.logger.info(f"Maximum steps allowed: {max_steps}")
        
        start_time = time.time()
        step_count = 0
        last_action = None
        
        try:
            while self.is_running and step_count < max_steps:
                self.logger.info(f"\n--- Step {step_count + 1} ---")
                
                # Take screenshot
                screenshot_path = self._take_screenshot(step_count)
                if not screenshot_path:
                    break
                
                # Analyze screenshot
                action_plan = self.vision_analyzer.analyze_screenshot(
                    screenshot_path,
                    self.current_instruction,
                    step_count,
                    previous_action=last_action
                )
                
                if not action_plan:
                    self.logger.error("Failed to get action plan from vision analyzer")
                    break
                
                # Check for completion or error
                action = action_plan.get("action", "").upper()
                if action == "COMPLETE":
                    self.logger.info("Task completed successfully!")
                    self.is_running = False
                    break
                elif action == "ERROR":
                    self.logger.error(f"Vision analyzer reported error: {action_plan.get('reasoning', 'Unknown error')}")
                    break
                
                # Handle common scenarios
                self.action_executor.handle_common_scenarios(action_plan, self.current_instruction)
                
                # Execute action
                success = self.action_executor.execute_action(action_plan)
                if not success:
                    self.logger.warning(f"Failed to execute action: {action}")
                    # Try to continue with next step unless it's a critical failure
                    if action in ["TAP", "TYPE"]:
                        # For critical actions, we might want to retry or abort
                        pass
                
                # Store action for history
                action_plan["step"] = step_count + 1
                action_plan["success"] = success
                action_plan["timestamp"] = time.time()
                self.actions_taken.append(action_plan)
                
                # Call progress callback if provided
                if progress_callback:
                    progress_callback(step_count + 1, action_plan)
                
                # Detect if we're stuck in a loop
                if self._detect_action_loop(action_plan, last_action):
                    self.logger.warning("Detected potential action loop, trying alternative approach")
                    # Take a screenshot and try a different approach
                    alternative_action = self._get_alternative_action(action_plan)
                    if alternative_action:
                        self.action_executor.execute_action(alternative_action)
                
                last_action = action_plan
                step_count += 1
                
                # Small delay between steps
                time.sleep(0.5)
        
        except KeyboardInterrupt:
            self.logger.info("Execution interrupted by user")
            self.is_running = False
        except Exception as e:
            self.logger.error(f"Unexpected error during execution: {e}", exc_info=True)
        
        # Calculate execution summary
        execution_time = time.time() - start_time
        summary = self._create_execution_summary(step_count, execution_time)
        
        self.logger.info(f"Execution completed in {execution_time:.2f} seconds with {step_count} steps")
        
        return summary
    
    def _take_screenshot(self, step_count: int) -> Optional[str]:
        """Take a screenshot and return the path."""
        try:
            filename = f"step_{step_count:03d}"
            screenshot_path = self.adb_controller.get_screenshot(filename)
            self.screenshots_taken.append(screenshot_path)
            return screenshot_path
        except Exception as e:
            self.logger.error(f"Failed to take screenshot: {e}")
            return None
    
    def _detect_action_loop(self, current_action: Dict, last_action: Optional[Dict]) -> bool:
        """Detect if we're stuck in an action loop."""
        if not last_action:
            return False
        
        # Check if same action type is repeated
        if (current_action.get("action") == last_action.get("action") and
            current_action.get("coordinates") == last_action.get("coordinates")):
            return True
        
        # Check if we're alternating between two actions
        if len(self.actions_taken) >= 4:
            recent_actions = [a.get("action") for a in self.actions_taken[-4:]]
            if len(set(recent_actions)) <= 2:  # Only 1 or 2 different actions
                return True
        
        return False
    
    def _get_alternative_action(self, stuck_action: Dict) -> Optional[Dict]:
        """Get an alternative action when stuck in a loop."""
        action = stuck_action.get("action", "").upper()
        
        if action == "TAP":
            # Try scrolling instead
            return {
                "action": "SWIPE",
                "swipe_coords": [[400, 800], [400, 400]],
                "reasoning": "Alternative action: scrolling to find different elements",
                "confidence": 0.5
            }
        elif action == "SWIPE":
            # Try going back
            return {
                "action": "BACK",
                "reasoning": "Alternative action: going back to previous screen",
                "confidence": 0.7
            }
        
        return None
    
    def _create_execution_summary(self, steps_taken: int, execution_time: float) -> Dict:
        """Create a summary of the execution."""
        return {
            "instruction": self.current_instruction,
            "steps_taken": steps_taken,
            "execution_time": execution_time,
            "max_steps": self.config.max_iterations,
            "completed": any(a.get("action") == "COMPLETE" for a in self.actions_taken),
            "actions_summary": self.action_executor.get_execution_summary(self.actions_taken),
            "screenshots_count": len(self.screenshots_taken),
            "final_status": self._get_final_status()
        }
    
    def _get_final_status(self) -> str:
        """Get the final status of execution."""
        if not self.actions_taken:
            return "NO_ACTIONS_TAKEN"
        
        last_action = self.actions_taken[-1]
        last_action_type = last_action.get("action", "").upper()
        
        if last_action_type == "COMPLETE":
            return "COMPLETED"
        elif last_action_type == "ERROR":
            return "ERROR"
        elif len(self.actions_taken) >= self.config.max_iterations:
            return "MAX_STEPS_REACHED"
        elif not self.is_running:
            return "INTERRUPTED"
        else:
            return "UNKNOWN"
    
    def stop_execution(self):
        """Stop the current execution."""
        self.is_running = False
        self.logger.info("Execution stop requested")
    
    def get_current_screenshot(self) -> Optional[str]:
        """Get a fresh screenshot of the current device state."""
        try:
            return self.adb_controller.get_screenshot("current_state")
        except Exception as e:
            self.logger.error(f"Failed to get current screenshot: {e}")
            return None
    
    def go_to_home(self):
        """Navigate to device home screen."""
        self.adb_controller.go_to_home_screen()
    
    def open_app(self, app_name: str):
        """Open an app by name."""
        return self.action_executor.open_app_by_name(app_name)
    
    def get_device_info(self) -> Dict:
        """Get information about the connected device."""
        return self.adb_controller.get_device_info()
    
    def cleanup(self, keep_screenshots: bool = True):
        """
        Clean up resources and temporary files.
        
        Args:
            keep_screenshots: Whether to preserve screenshots (default: True)
        """
        try:
            if keep_screenshots:
                self.logger.info(f"Preserving {len(self.screenshots_taken)} screenshots:")
                for screenshot_path in self.screenshots_taken:
                    if os.path.exists(screenshot_path):
                        self.logger.info(f"  📁 {screenshot_path}")
            else:
                # Clean up screenshot files if requested
                self.logger.info("Cleaning up screenshot files...")
                for screenshot_path in self.screenshots_taken:
                    if os.path.exists(screenshot_path):
                        try:
                            os.remove(screenshot_path)
                            self.logger.debug(f"Removed screenshot: {screenshot_path}")
                        except Exception as e:
                            self.logger.warning(f"Could not remove screenshot {screenshot_path}: {e}")
            
            self.logger.info("Cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup() 