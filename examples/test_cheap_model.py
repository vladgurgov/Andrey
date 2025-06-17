"""
Test with cheaper OpenAI model (gpt-4o-mini)
"""

import os
import sys

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mobile_agent import MobileAgent
from mobile_agent.config import MobileAgentConfig


def main():
    """Test with cheaper model."""
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("❌ Please set your OPENAI_API_KEY environment variable")
        return
    
    print("🧪 Testing with cheaper gpt-4o-mini model...")
    print("This costs ~90% less than gpt-4o")
    print()
    
    # Create config with cheaper model
    config = MobileAgentConfig(
        openai_api_key=openai_api_key,
        openai_model="gpt-4o-mini",  # Much cheaper!
        max_iterations=3
    )
    
    try:
        with MobileAgent(config=config) as agent:
            print("📱 Taking screenshot and testing vision analysis...")
            
            # Simple test
            result = agent.execute_instruction("Go to the home screen", max_steps=1)
            
            if result['final_status'] != 'NO_ACTIONS_TAKEN':
                print("🎉 SUCCESS! gpt-4o-mini is working!")
                print(f"Result: {result['final_status']}")
                
                # If that worked, try YouTube
                print("\n🎥 Trying YouTube task with cheaper model...")
                youtube_result = agent.execute_instruction(
                    "Open YouTube app", 
                    max_steps=5
                )
                print(f"YouTube result: {youtube_result['final_status']}")
                
            else:
                print("❌ Still having quota issues even with cheaper model")
                print("Your OpenAI account may need:")
                print("1. Payment method added")
                print("2. Billing limits increased") 
                print("3. Wait for quota reset (if free tier)")
                
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main() 