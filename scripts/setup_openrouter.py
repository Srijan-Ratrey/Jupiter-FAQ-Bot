#!/usr/bin/env python3
"""
OpenRouter Setup Script for Jupiter FAQ Bot
This script helps you configure and test OpenRouter with Mistral models.
"""

import os
import sys
import json
from pathlib import Path

def setup_openrouter():
    """Interactive setup for OpenRouter API key."""
    print("🚀 Jupiter FAQ Bot - OpenRouter/Mistral Setup")
    print("=" * 50)
    
    # Check if API key is already set
    existing_key = os.getenv("OPENROUTER_API_KEY")
    if existing_key:
        print(f"✅ OpenRouter API key already set: {existing_key[:10]}...")
        use_existing = input("Use existing key? (y/n): ").strip().lower()
        if use_existing == 'y':
            return existing_key
    
    print("\n📋 To get your OpenRouter API key:")
    print("1. Visit: https://openrouter.ai/")
    print("2. Sign up/Login")
    print("3. Go to 'Keys' section")
    print("4. Create a new API key")
    print("5. Copy the key starting with 'sk-or-...'")
    
    print("\n💡 OpenRouter provides free access to:")
    print("• Mistral 7B Instruct (mistralai/mistral-7b-instruct:free)")
    print("• Other free models available")
    
    # Get API key from user
    api_key = input("\n🔑 Enter your OpenRouter API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided. Exiting...")
        return None
    
    if not api_key.startswith("sk-or-"):
        print("⚠️  Warning: OpenRouter keys usually start with 'sk-or-'")
        continue_anyway = input("Continue anyway? (y/n): ").strip().lower()
        if continue_anyway != 'y':
            return None
    
    # Save to environment (for current session)
    os.environ["OPENROUTER_API_KEY"] = api_key
    
    # Optionally save to .env file
    save_to_file = input("\n💾 Save API key to .env file? (y/n): ").strip().lower()
    if save_to_file == 'y':
        env_file = Path(".env")
        with open(env_file, "a") as f:
            f.write(f"\nOPENROUTER_API_KEY={api_key}\n")
        print(f"✅ API key saved to {env_file}")
        
        # Update .gitignore to include .env
        gitignore_file = Path(".gitignore")
        if gitignore_file.exists():
            with open(gitignore_file, "r") as f:
                content = f.read()
            if ".env" not in content:
                with open(gitignore_file, "a") as f:
                    f.write("\n# Environment variables\n.env\n")
                print("✅ Added .env to .gitignore")
    
    print("\n🎯 For bash/zsh, you can also add to your profile:")
    print(f"export OPENROUTER_API_KEY={api_key}")
    
    return api_key

def test_openrouter_integration():
    """Test OpenRouter integration with the bot."""
    print("\n🧪 Testing OpenRouter Integration...")
    
    try:
        # Import bot modules
        sys.path.append('src')
        from bot.llm_service import setup_llm_service
        from bot.jupiter_faq_bot import create_jupiter_bot
        
        # Test LLM service
        print("1. Testing LLM service...")
        llm_service = setup_llm_service({"primary_provider": "openrouter"})
        
        # Test with mock FAQ
        mock_faqs = [
            {
                "question": "How can I open a Savings account?",
                "answer": "To open a free Savings or Salary Bank Account on Jupiter - powered by Federal Bank - in 3 minutes, simply install the Jupiter App.",
                "category": "Account Opening",
                "similarity_score": 0.85
            }
        ]
        
        print("2. Testing response generation...")
        response = llm_service.generate_faq_response(
            user_query="How do I create a new account?",
            relevant_faqs=mock_faqs,
            style="conversational"
        )
        
        print(f"✅ Provider: {response['provider']}")
        print(f"✅ Response: {response['response'][:100]}...")
        
        if response['provider'] == 'OpenRouterProvider':
            print("🎉 OpenRouter/Mistral integration successful!")
            return True
        else:
            print(f"⚠️  Using fallback provider: {response['provider']}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing integration: {e}")
        return False

def demo_with_openrouter():
    """Run a demo using OpenRouter/Mistral."""
    print("\n🎮 Running Demo with OpenRouter/Mistral...")
    
    try:
        sys.path.append('src')
        from bot.jupiter_faq_bot import create_jupiter_bot
        
        # Create bot with OpenRouter config
        config = {
            "primary_provider": "openrouter",
            "openrouter_api_key": os.getenv("OPENROUTER_API_KEY")
        }
        
        bot = create_jupiter_bot(config_file="config.json")
        
        # Test questions
        test_questions = [
            "How can I open a Jupiter account?",
            "What is Jupiter Pro?",
            "How to complete KYC verification?"
        ]
        
        print(f"\nTesting {len(test_questions)} questions with Mistral...")
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n--- Question {i}: {question} ---")
            
            response = bot.ask(question, style="conversational")
            
            print(f"Provider: {response['provider']}")
            print(f"Confidence: {response['confidence']:.1%}")
            print(f"Response: {response['response'][:200]}...")
            
            if response.get('sources'):
                print(f"Sources: {len(response['sources'])} relevant FAQs")
        
        print("\n✅ Demo completed!")
        
    except Exception as e:
        print(f"❌ Error running demo: {e}")

def main():
    """Main setup function."""
    print("Choose an option:")
    print("1. Setup OpenRouter API key")
    print("2. Test OpenRouter integration")
    print("3. Run demo with OpenRouter/Mistral")
    print("4. All of the above")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice in ["1", "4"]:
        api_key = setup_openrouter()
        if not api_key:
            print("❌ Setup failed. Exiting...")
            return
    
    if choice in ["2", "4"]:
        success = test_openrouter_integration()
        if not success:
            print("❌ Integration test failed")
            return
    
    if choice in ["3", "4"]:
        demo_with_openrouter()
    
    print("\n🎉 Setup complete! You can now use:")
    print("python demo_bot.py interactive")
    print("python demo_bot.py detailed")

if __name__ == "__main__":
    main() 