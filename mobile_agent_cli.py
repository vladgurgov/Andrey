#!/usr/bin/env python3
"""
Mobile Agent Command Line Interface

Run mobile automation tasks from the command line with natural language instructions.

Examples:
    python mobile_agent_cli.py --task "Open YouTube and play first video"
    python mobile_agent_cli.py --task "Go to Settings and change WiFi" --max-steps 10
    python mobile_agent_cli.py --task "Take a screenshot" --model gpt-4o-mini
"""

import argparse
import os
import sys
import logging
from typing import Optional

# Add the package to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mobile_agent import MobileAgent
from mobile_agent.config import MobileAgentConfig


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('mobile_agent_cli.log')
        ]
    )


def progress_callback(step: int, action_plan: dict):
    """Progress callback for command line interface."""
    action = action_plan.get("action", "UNKNOWN")
    reasoning = action_plan.get("reasoning", "No reasoning")
    confidence = action_plan.get("confidence", 0.0)
    
    print(f"📱 Step {step}: {action} (confidence: {confidence:.2f})")
    print(f"   💭 {reasoning}")
    print()


def check_prerequisites() -> bool:
    """Check if all prerequisites are met."""
    print("🔍 Checking prerequisites...")
    
    # Check OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not set")
        print("   Set it with: export OPENAI_API_KEY='your-key-here'")
        return False
    
    # Check ADB availability
    try:
        import subprocess
        result = subprocess.run(["adb", "version"], capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            print("❌ ADB not found or not working")
            print("   Install Android SDK Platform Tools")
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("❌ ADB not found")
        print("   Install from: https://developer.android.com/studio/command-line/adb")
        return False
    
    # Check device connection
    try:
        result = subprocess.run(["adb", "devices"], capture_output=True, text=True, timeout=5)
        devices = result.stdout.strip().split('\n')[1:]
        if not any(device.strip() for device in devices):
            print("❌ No Android devices found")
            print("   Connect your device and enable USB debugging")
            return False
    except subprocess.TimeoutExpired:
        print("❌ ADB devices command timed out")
        return False
    
    print("✅ All prerequisites met!")
    return True


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Mobile Agent - AI-powered Android automation via natural language",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --task "Open YouTube and play first video"
  %(prog)s --task "Go to Settings" --max-steps 5
  %(prog)s --task "Take a screenshot and go home" --model gpt-4o-mini
  %(prog)s --task "Open Chrome and search for weather" --verbose
  %(prog)s --list-models

Common tasks:
  - "Open [app name]" (e.g., "Open YouTube", "Open Settings")  
  - "Go to the home screen"
  - "Take a screenshot"
  - "Press the back button"
  - "Open [app] and [action]" (e.g., "Open YouTube and play first video")
        """
    )
    
    # Task arguments
    parser.add_argument(
        "--task", "-t", 
        type=str,
        help="Natural language task to execute (e.g., 'Open YouTube and play first video')"
    )
    
    # Configuration arguments
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="gpt-4o",
        help="OpenAI model to use (default: gpt-4o-mini). Try gpt-4o for best quality"
    )
    
    parser.add_argument(
        "--max-steps", "-s",
        type=int,
        default=20,
        help="Maximum steps to execute (default: 20)"
    )
    
    parser.add_argument(
        "--screenshot-quality", "-q",
        type=float,
        default=0.5,
        help="Screenshot scale factor (default: 0.5). Lower = faster but less detail"
    )
    
    parser.add_argument(
        "--step-delay", "-d",
        type=float,
        default=2.0,
        help="Delay between actions in seconds (default: 2.0)"
    )
    
    parser.add_argument(
        "--adb-path",
        type=str,
        default="adb",
        help="Path to ADB executable (default: adb)"
    )
    
    # Utility arguments
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress output during execution"
    )
    
    parser.add_argument(
        "--screenshot-only",
        action="store_true",
        help="Just take a screenshot and exit"
    )
    
    parser.add_argument(
        "--device-info",
        action="store_true",
        help="Show connected device information and exit"
    )
    
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available OpenAI models and exit"
    )
    
    parser.add_argument(
        "--check-setup",
        action="store_true",
        help="Check prerequisites and setup, then exit"
    )
    
    parser.add_argument(
        "--cleanup-screenshots",
        action="store_true",
        help="Remove screenshots after execution (default: keep them)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    print("🤖 Mobile Agent CLI")
    print("=" * 50)
    
    # Handle utility commands
    if args.check_setup:
        success = check_prerequisites()
        sys.exit(0 if success else 1)
    
    if args.list_models:
        print("Available OpenAI Models:")
        print("- gpt-4o (latest, best quality, higher cost)")
        print("- gpt-4o-mini (90% cheaper, good quality)")
        print("- gpt-4-turbo (older model)")
        print("\nRecommendation: Start with gpt-4o-mini for testing")
        sys.exit(0)
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Run --check-setup for details.")
        sys.exit(1)
    
    try:
        # Create configuration
        config = MobileAgentConfig(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_model=args.model,
            max_iterations=args.max_steps,
            screenshot_quality=args.screenshot_quality,
            step_delay=args.step_delay,
            adb_path=args.adb_path
        )
        
        # Initialize agent
        agent = MobileAgent(config=config)
        exit_code = 1  # Default to error
        
        # Handle device info request
        if args.device_info:
            device_info = agent.get_device_info()
            print("📱 Connected Device Information:")
            print(f"   Model: {device_info.get('model', 'Unknown')}")
            print(f"   Android Version: {device_info.get('android_version', 'Unknown')}")
            print(f"   Screen Size: {device_info.get('screen_size', 'Unknown')}")
            exit_code = 0
        
        # Handle screenshot-only request
        elif args.screenshot_only:
            print("📸 Taking screenshot...")
            screenshot_path = agent.get_current_screenshot()
            print(f"✅ Screenshot saved: {screenshot_path}")
            exit_code = 0
        
        # Main task execution
        elif not args.task:
            print("❌ No task specified. Use --task 'your instruction here'")
            print("Examples:")
            print("  --task 'Open YouTube and play first video'")
            print("  --task 'Go to Settings'")
            print("  --task 'Take a screenshot'")
            exit_code = 1
        
        else:
            print(f"🎯 Task: {args.task}")
            print(f"🤖 Model: {args.model}")
            print(f"📊 Max Steps: {args.max_steps}")
            print()
            
            # Execute the task
            callback = None if args.no_progress else progress_callback
            
            result = agent.execute_instruction(
                instruction=args.task,
                max_steps=args.max_steps,
                progress_callback=callback
            )
            
            # Show results
            print("\n📊 EXECUTION SUMMARY")
            print("=" * 50)
            print(f"🎯 Task: {result['instruction']}")
            print(f"⏱️  Time: {result['execution_time']:.2f} seconds")
            print(f"👣 Steps: {result['steps_taken']}")
            print(f"🏁 Status: {result['final_status']}")
            print(f"✅ Completed: {result['completed']}")
            
            # Action breakdown
            if result['actions_summary']['action_breakdown']:
                print(f"\n📋 Actions Taken:")
                for action, count in result['actions_summary']['action_breakdown'].items():
                    print(f"   {action}: {count} times")
            
            # Exit with appropriate code
            if result['completed']:
                print("\n🎉 Task completed successfully!")
                exit_code = 0
            else:
                print(f"\n⚠️  Task ended with status: {result['final_status']}")
                exit_code = 1
        
        # Cleanup with screenshot preservation option
        keep_screenshots = not args.cleanup_screenshots
        agent.cleanup(keep_screenshots=keep_screenshots)
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        # Still do cleanup if agent was created
        if 'agent' in locals():
            keep_screenshots = not args.cleanup_screenshots
            agent.cleanup(keep_screenshots=keep_screenshots)
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        # Still do cleanup if agent was created
        if 'agent' in locals():
            keep_screenshots = not args.cleanup_screenshots
            agent.cleanup(keep_screenshots=keep_screenshots)
        sys.exit(1)
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main() 