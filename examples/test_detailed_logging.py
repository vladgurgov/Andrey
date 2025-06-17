"""
Test Detailed ADB Logging

This demonstrates the new detailed logging that shows exactly what ADB commands
are being executed on your Android device.
"""

import os
import sys

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mobile_agent import MobileAgent
from mobile_agent.config import MobileAgentConfig


def main():
    """Test detailed ADB logging with mock vision analyzer."""
    
    print("🔧 Mobile Agent - Detailed ADB Command Logging Test")
    print("=" * 60)
    print("This will show you EXACTLY what ADB commands are being executed!")
    print()
    
    try:
        # Create config
        config = MobileAgentConfig(
            openai_api_key="mock-key",  # Not needed for this test
            max_iterations=3
        )
        
        # Initialize agent
        with MobileAgent(config=config) as agent:
            
            print("📱 Device Info:", agent.get_device_info())
            print()
            
            print("🔍 Test 1: Taking a screenshot")
            print("-" * 40)
            screenshot_path = agent.get_current_screenshot()
            print(f"Screenshot saved: {screenshot_path}")
            print()
            
            print("🔍 Test 2: Demonstrating device controls")
            print("-" * 40)
            
            # Test HOME button
            print("Testing HOME button...")
            agent.adb_controller.press_home()
            print()
            
            # Test TAP with coordinate scaling
            print("Testing TAP with coordinate scaling...")
            agent.adb_controller.tap(100, 200)  # Small coordinates
            print()
            
            # Test SWIPE
            print("Testing SWIPE...")
            agent.adb_controller.swipe(400, 800, 400, 400)  # Scroll up
            print()
            
            # Test TYPE
            print("Testing TYPE...")
            agent.adb_controller.type_text("Hello Mobile Agent!")
            print()
            
            # Test BACK button
            print("Testing BACK button...")
            agent.adb_controller.press_back()
            print()
            
            print("✅ ALL ADB COMMANDS EXECUTED SUCCESSFULLY!")
            print()
            print("📊 What you saw above:")
            print("- 🔧 Exact ADB commands with full paths")
            print("- 👆 Coordinate scaling (screenshot size → device size)")
            print("- ⌨️ Text processing (original → escaped)")
            print("- 📸 Screenshot workflow (capture → pull → process)")
            print("- ✅ Success confirmations for each action")
            print()
            print("This proves your mobile agent is executing real ADB commands!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 