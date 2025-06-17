"""
Model Selection Example - How to handle different OpenAI models

This example shows how to:
1. Try different OpenAI vision models
2. Handle model deprecation gracefully
3. Use fallback models if primary model fails
"""

import os
import sys

# Add the parent directory to the path so we can import mobile_agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mobile_agent import MobileAgent
from mobile_agent.config import MobileAgentConfig


def test_model(model_name: str, openai_api_key: str) -> bool:
    """Test if a specific OpenAI model works."""
    print(f"🧪 Testing model: {model_name}")
    
    try:
        # Create config with specific model
        config = MobileAgentConfig(
            openai_api_key=openai_api_key,
            openai_model=model_name,
            max_iterations=1  # Just test with 1 step
        )
        
        # Try to initialize and take a screenshot
        with MobileAgent(config=config) as agent:
            screenshot_path = agent.get_current_screenshot()
            if screenshot_path:
                print(f"✅ Model {model_name} works! Screenshot: {screenshot_path}")
                return True
            else:
                print(f"❌ Model {model_name} failed - no screenshot")
                return False
                
    except Exception as e:
        error_msg = str(e)
        if "model_not_found" in error_msg or "deprecated" in error_msg.lower():
            print(f"❌ Model {model_name} is deprecated or not found")
        elif "insufficient_quota" in error_msg or "billing" in error_msg.lower():
            print(f"⚠️  Model {model_name} requires billing/quota - check your OpenAI account")
        else:
            print(f"❌ Model {model_name} failed: {error_msg}")
        return False


def find_working_model(openai_api_key: str) -> str:
    """Find the best working OpenAI vision model."""
    
    # List of models to try in order of preference
    models_to_try = [
        "gpt-4o",              # Latest and best
        "gpt-4o-mini",         # Cheaper alternative
        "gpt-4-turbo",         # Older but might still work
        "gpt-4-vision-preview" # Deprecated but try anyway
    ]
    
    print("🔍 Searching for working OpenAI vision model...")
    print()
    
    for model in models_to_try:
        if test_model(model, openai_api_key):
            print(f"🎉 Found working model: {model}")
            return model
        print()
    
    print("❌ No working models found!")
    return None


def main():
    """Main example function."""
    
    # Check for OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("❌ Please set your OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY='your-api-key-here'")
        return
    
    print("🤖 Mobile Agent - Model Selection Example")
    print("=" * 50)
    
    # Find working model
    working_model = find_working_model(openai_api_key)
    
    if working_model:
        print(f"\n✅ Using model: {working_model}")
        
        # Create agent with working model
        config = MobileAgentConfig(
            openai_api_key=openai_api_key,
            openai_model=working_model,
            max_iterations=5
        )
        
        # Test simple instruction
        print("\n🧪 Testing simple instruction...")
        with MobileAgent(config=config) as agent:
            result = agent.execute_instruction("Go to the home screen")
            print(f"Result: {result['final_status']}")
            
            if result['final_status'] == 'COMPLETED':
                print(f"🎉 Success! Model {working_model} is working perfectly!")
            else:
                print(f"⚠️  Partial success with model {working_model}")
    
    else:
        print("\n❌ Could not find any working OpenAI vision model.")
        print("\nTroubleshooting:")
        print("1. Verify your OpenAI API key is correct")
        print("2. Check your OpenAI account has billing set up")
        print("3. Ensure you have access to vision models")
        print("4. Try the OpenAI playground to test your access")


if __name__ == "__main__":
    main() 