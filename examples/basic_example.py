"""
Basic example of using Mobile Agent Library.

This example demonstrates how to:
1. Initialize the mobile agent
2. Execute simple instructions
3. Handle results and errors
"""

import os
import sys
import time

# Add the parent directory to the path so we can import mobile_agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mobile_agent import MobileAgent


def progress_callback(step: int, action_plan: dict):
    """Callback function to track progress during execution."""
    action = action_plan.get("action", "UNKNOWN")
    reasoning = action_plan.get("reasoning", "No reasoning")
    confidence = action_plan.get("confidence", 0.0)
    
    print(f"Step {step}: {action} (confidence: {confidence:.2f})")
    print(f"  Reasoning: {reasoning}")
    print("-" * 50)


def main():
    """Main example function."""
    # Set your OpenAI API key (or set OPENAI_API_KEY environment variable)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Please set your OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY='your-api-key-here'")
        return
    
    print("🤖 Mobile Agent Library - Basic Example")
    print("=" * 50)
    
    try:
        # Initialize the mobile agent
        print("Initializing Mobile Agent...")
        with MobileAgent(openai_api_key=openai_api_key) as agent:
            
            # Get device information
            device_info = agent.get_device_info()
            print(f"Connected to device: {device_info.get('model', 'Unknown')}")
            print(f"Android version: {device_info.get('android_version', 'Unknown')}")
            print(f"Screen size: {device_info.get('screen_size', 'Unknown')}")
            print()
            
            # Example 1: Take a screenshot
            print("📸 Taking initial screenshot...")
            screenshot_path = agent.get_current_screenshot()
            if screenshot_path:
                print(f"Screenshot saved to: {screenshot_path}")
            print()
            
            # Example 2: Simple instruction - go to home screen
            print("🏠 Example 1: Going to home screen...")
            result = agent.execute_instruction(
                "Go to the home screen",
                max_steps=3,
                progress_callback=progress_callback
            )
            print(f"Result: {result['final_status']}")
            print(f"Steps taken: {result['steps_taken']}")
            print()
            
            # Example 3: More complex instruction - open an app
            print("📱 Example 2: Opening Settings app...")
            result = agent.execute_instruction(
                "Open the Settings app",
                max_steps=5,
                progress_callback=progress_callback
            )
            print(f"Result: {result['final_status']}")
            print(f"Execution time: {result['execution_time']:.2f} seconds")
            print()
            
            # Example 4: Advanced instruction (if you want to try)
            user_instruction = input("Enter a custom instruction (or press Enter to skip): ").strip()
            if user_instruction:
                print(f"🎯 Executing custom instruction: '{user_instruction}'")
                result = agent.execute_instruction(
                    user_instruction,
                    max_steps=10,
                    progress_callback=progress_callback
                )
                print(f"Final result: {result}")
            
            print("✅ Example completed successfully!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure:")
        print("1. Your Android device is connected via ADB")
        print("2. USB debugging is enabled")
        print("3. You have authorized your computer")
        print("4. Your OpenAI API key is valid")


def test_adb_connection():
    """Test if ADB is working and device is connected."""
    print("🔍 Testing ADB connection...")
    
    try:
        import subprocess
        result = subprocess.run(["adb", "devices"], capture_output=True, text=True)
        if result.returncode == 0:
            devices = result.stdout.strip().split('\n')[1:]  # Skip header
            if devices and any(device.strip() for device in devices):
                print("✅ ADB is working and device(s) found:")
                for device in devices:
                    if device.strip():
                        print(f"  - {device}")
                return True
            else:
                print("❌ No devices found")
                return False
        else:
            print(f"❌ ADB command failed: {result.stderr}")
            return False
    except FileNotFoundError:
        print("❌ ADB not found. Please install Android SDK Platform Tools")
        return False
    except Exception as e:
        print(f"❌ Error testing ADB: {e}")
        return False


if __name__ == "__main__":
    print("Mobile Agent Library - Basic Example")
    print("=" * 40)
    
    # Test ADB connection first
    if test_adb_connection():
        print()
        main()
    else:
        print("\nPlease fix the ADB connection issues and try again.")
        print("\nSetup instructions:")
        print("1. Enable Developer Options on your Android device")
        print("2. Enable USB Debugging")
        print("3. Connect device via USB")
        print("4. Run 'adb devices' to verify connection")
        print("5. Install ADB if not available: https://developer.android.com/studio/command-line/adb") 