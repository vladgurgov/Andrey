"""
YouTube Example - Open YouTube App and Play First Video

This example demonstrates the exact use case mentioned in the requirements:
"Open YouTube App and play first video"

Usage:
    python youtube_example.py                           # Interactive mode
    python youtube_example.py --task "Open YouTube"     # Command line mode
    python youtube_example.py --help                    # Show all options
"""

import argparse
import os
import sys
import time

# Add the parent directory to the path so we can import mobile_agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mobile_agent import MobileAgent
from mobile_agent.config import MobileAgentConfig


def main():
    """Main YouTube example."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="YouTube Mobile Agent Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Interactive mode
  %(prog)s --task "Open YouTube"              # Open YouTube app
  %(prog)s --task "Open YouTube and play first video" --model gpt-4o-mini
        """
    )
    
    parser.add_argument(
        "--task", "-t",
        type=str,
        help="Task to execute (default: 'Open YouTube app and play the first video')"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="gpt-4o",
        help="OpenAI model to use (default: gpt-4o)"
    )
    
    parser.add_argument(
        "--max-steps", "-s",
        type=int,
        default=15,
        help="Maximum steps to execute (default: 15)"
    )
    
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Skip interactive prompts and confirmations"
    )
    
    args = parser.parse_args()
    
    # Check for OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("❌ Please set your OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Determine task and mode
    task = args.task or "Open YouTube app and play the first video"
    interactive_mode = not args.no_interactive and not args.task
    
    if interactive_mode:
        print("🎥 Mobile Agent - YouTube Example")
        print("=" * 50)
        print("This example will:")
        print("1. Open the YouTube app")
        print("2. Find and play the first video")
        print("3. Show execution details")
        print()
    else:
        print("🎥 Mobile Agent - YouTube CLI")
        print("=" * 50)
        print(f"🎯 Task: {task}")
        print(f"🤖 Model: {args.model}")
        print(f"📊 Max Steps: {args.max_steps}")
        print()
    
    def progress_callback(step: int, action_plan: dict):
        """Track progress with detailed output."""
        action = action_plan.get("action", "UNKNOWN")
        reasoning = action_plan.get("reasoning", "No reasoning provided")
        confidence = action_plan.get("confidence", 0.0)
        
        print(f"📱 Step {step}: {action}")
        print(f"   💭 Reasoning: {reasoning}")
        print(f"   🎯 Confidence: {confidence:.2f}")
        print()
    
    try:
        # Create configuration
        config = MobileAgentConfig(
            openai_api_key=openai_api_key,
            openai_model=args.model,
            max_iterations=args.max_steps
        )
        
        # Initialize the mobile agent
        print("🤖 Initializing Mobile Agent...")
        with MobileAgent(config=config) as agent:
            
            # Get and display device info
            device_info = agent.get_device_info()
            print(f"📱 Connected Device: {device_info.get('model', 'Unknown')}")
            print(f"🤖 Android Version: {device_info.get('android_version', 'Unknown')}")
            print(f"📐 Screen Size: {device_info.get('screen_size', 'Unknown')}")
            print()
            
            if interactive_mode:
                # Interactive mode - detailed workflow
                # First, go to home screen to ensure clean start
                print("🏠 Going to home screen...")
                home_result = agent.execute_instruction(
                    "Go to the home screen",
                    max_steps=3,
                    progress_callback=progress_callback
                )
                print(f"✅ Home screen result: {home_result['final_status']}")
                print()
                
                # Take initial screenshot
                print("📸 Taking initial screenshot...")
                initial_screenshot = agent.get_current_screenshot()
                if initial_screenshot:
                    print(f"📁 Screenshot saved: {initial_screenshot}")
                print()
                
                # Main task: Open YouTube and play first video
                print("🎯 Main Task: Open YouTube App and Play First Video")
                print("-" * 50)
            
            # Execute the main instruction
            start_time = time.time()
            result = agent.execute_instruction(
                task,
                max_steps=args.max_steps,
                progress_callback=progress_callback
            )
            execution_time = time.time() - start_time
            
            # Display detailed results
            print("📊 EXECUTION SUMMARY")
            print("=" * 50)
            print(f"🎯 Instruction: {result['instruction']}")
            print(f"⏱️  Execution Time: {execution_time:.2f} seconds")
            print(f"👣 Steps Taken: {result['steps_taken']}")
            print(f"🎬 Screenshots: {result['screenshots_count']}")
            print(f"🏁 Final Status: {result['final_status']}")
            print(f"✅ Completed: {result['completed']}")
            print()
            
            # Show action breakdown
            actions_summary = result['actions_summary']
            print("📋 Action Breakdown:")
            for action, count in actions_summary['action_breakdown'].items():
                print(f"   {action}: {count} times")
            print()
            
            # Interpret the result
            if result['final_status'] == 'COMPLETED':
                print("🎉 SUCCESS! YouTube video should now be playing!")
                print("The agent successfully:")
                print("✅ Opened the YouTube app")
                print("✅ Found the first video")
                print("✅ Started playing it")
            elif result['final_status'] == 'MAX_STEPS_REACHED':
                print("⚠️  Task reached maximum steps but may have partially completed")
                print("The agent made progress but couldn't complete all steps within the limit")
            elif result['final_status'] == 'ERROR':
                print("❌ Task failed with an error")
                print("This could be due to:")
                print("- YouTube app not installed")
                print("- Network connectivity issues")
                print("- Unexpected UI changes")
            else:
                print(f"❓ Task ended with status: {result['final_status']}")
            
            print()
            
            if interactive_mode:
                # Interactive mode - offer final screenshot
                take_final = input("📸 Take final screenshot to see current state? (y/n): ").lower().strip()
                if take_final in ['y', 'yes']:
                    final_screenshot = agent.get_current_screenshot()
                    if final_screenshot:
                        print(f"📁 Final screenshot saved: {final_screenshot}")
                
                print("\n🧹 Cleaning up...")
                print("✅ Mobile Agent session completed!")
            else:
                # Command line mode - show success/failure
                if result['completed']:
                    print("\n🎉 Task completed successfully!")
                else:
                    print(f"\n⚠️  Task ended with status: {result['final_status']}")
                    if result['final_status'] == 'MAX_STEPS_REACHED':
                        print("Try increasing --max-steps if the task needs more time")
                    elif 'ERROR' in result['final_status']:
                        print("Check the logs above for error details")
            
    except KeyboardInterrupt:
        print("\n⚠️  Execution interrupted by user")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure your Android device is connected via USB")
        print("2. Verify USB debugging is enabled")
        print("3. Check that YouTube app is installed on your device")
        print("4. Make sure your OpenAI API key is valid")
        print("5. Verify ADB is working: run 'adb devices'")


def test_youtube_app():
    """Test if YouTube app is available on the device."""
    print("🔍 Checking if YouTube app is available...")
    
    try:
        import subprocess
        
        # Check if YouTube app is installed
        result = subprocess.run(
            ["adb", "shell", "pm", "list", "packages", "com.google.android.youtube"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0 and "com.google.android.youtube" in result.stdout:
            print("✅ YouTube app found on device")
            return True
        else:
            print("❌ YouTube app not found on device")
            print("Please install YouTube from Google Play Store")
            return False
            
    except Exception as e:
        print(f"⚠️  Could not check YouTube app status: {e}")
        return True  # Assume it's available


if __name__ == "__main__":
    print("YouTube Example - Mobile Agent Library")
    print("=" * 40)
    
    # Test YouTube app availability
    if test_youtube_app():
        print()
        main()
    else:
        print("\nPlease install YouTube app and try again.")
        print("You can install it from Google Play Store on your Android device.") 