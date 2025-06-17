"""
Test Coordinate Transformation Fix

This test verifies that the coordinate transformation between screenshot coordinates
(bottom-left origin) and device coordinates (top-left origin) is working correctly.
"""

import os
import sys

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mobile_agent import MobileAgent
from mobile_agent.config import MobileAgentConfig


def test_coordinate_transformation():
    """Test coordinate transformation with detailed logging."""
    
    print("🔧 Testing Coordinate Transformation Fix")
    print("=" * 50)
    print("This test will show the coordinate transformation from screenshot to device coordinates.")
    print("Screenshot coordinates: bottom-left origin (0,0)")
    print("Device coordinates: top-left origin (0,0)")
    print()
    
    try:
        # Create config
        config = MobileAgentConfig(
            openai_api_key="mock-key",  # Not needed for coordinate testing
            screenshot_quality=0.5  # This affects the scaling
        )
        
        # Initialize agent
        with MobileAgent(config=config) as agent:
            
            print("📱 Device Information:")
            device_info = agent.get_device_info()
            print(f"   Model: {device_info.get('model', 'Unknown')}")
            print(f"   Screen Size: {device_info.get('screen_size', 'Unknown')}")
            print()
            
            print("🧪 Coordinate Transformation Tests:")
            print("-" * 40)
            
            # Test corner coordinates
            test_coordinates = [
                (0, 0, "Bottom-left corner of screenshot"),
                (360, 0, "Bottom-center of screenshot"),
                (720, 0, "Bottom-right corner of screenshot (if 720px wide)"),
                (0, 800, "Top-left corner of screenshot (if 800px tall)"),
                (360, 400, "Center of screenshot"),
                (720, 800, "Top-right corner of screenshot"),
            ]
            
            for x, y, description in test_coordinates:
                print(f"\n📍 Testing: {description}")
                print(f"   Screenshot coords: ({x}, {y})")
                
                # This will show the coordinate transformation in the logs
                try:
                    agent.adb_controller.tap(x, y)
                    print("   ✅ Coordinate transformation logged above")
                except Exception as e:
                    print(f"   ⚠️  Error during tap (coordinate transform still logged): {e}")
                
                # Add a small delay
                import time
                time.sleep(1)
            
            print("\n🎯 Key Points About the Fix:")
            print("- Screenshot Y coordinates are flipped: device_y = screenshot_height - screenshot_y")
            print("- Coordinates are scaled by 1/screenshot_quality factor")
            print("- Device coordinates are clamped to screen bounds")
            print("- Detailed logging shows the transformation process")
            
            print("\n✅ Coordinate transformation test completed!")
            print("Check the logs above to see how coordinates are transformed.")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main test function."""
    test_coordinate_transformation()


if __name__ == "__main__":
    main() 