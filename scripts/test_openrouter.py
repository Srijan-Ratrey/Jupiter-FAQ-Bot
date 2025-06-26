#!/usr/bin/env python3
"""
Simple OpenRouter Test Script
Run this after setting your OPENROUTER_API_KEY environment variable.
"""

import os
import sys
sys.path.append('src')

def test_openrouter():
    """Test OpenRouter/Mistral integration."""
    print("🧪 Testing OpenRouter/Mistral Integration")
    print("=" * 45)
    
    # Check if API key is set
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENROUTER_API_KEY not found!")
        print("\n📋 To fix this:")
        print("1. Get your key from: https://openrouter.ai/keys")
        print("2. Run: export OPENROUTER_API_KEY='your-key-here'")
        print("3. Run this script again")
        return False
    
    print(f"✅ API Key found: {api_key[:10]}...")
    
    try:
        # Test LLM service
        from bot.llm_service import setup_llm_service
        
        print("🔧 Initializing LLM service...")
        llm_service = setup_llm_service({
            "primary_provider": "openrouter",
            "openrouter_api_key": api_key
        })
        
        # Test with mock FAQ
        mock_faqs = [
            {
                "question": "How can I open a Savings account?",
                "answer": "To open a free Savings or Salary Bank Account on Jupiter - powered by Federal Bank - in 3 minutes, simply install the Jupiter App. Follow the on-screen instructions to create your account.",
                "category": "Account Opening",
                "similarity_score": 0.85
            }
        ]
        
        print("🤖 Testing Mistral response generation...")
        response = llm_service.generate_faq_response(
            user_query="How do I create a new Jupiter account?",
            relevant_faqs=mock_faqs,
            style="conversational"
        )
        
        print(f"\n📊 Results:")
        print(f"   Provider: {response['provider']}")
        print(f"   Confidence: {response['confidence']:.1%}")
        print(f"   Response: {response['response'][:150]}...")
        
        if response['provider'] == 'OpenRouterProvider':
            print("\n🎉 SUCCESS! OpenRouter/Mistral is working!")
            return True
        else:
            print(f"\n⚠️  Using fallback provider: {response['provider']}")
            print("Check your API key and internet connection.")
            return False
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

def demo_questions():
    """Run a quick demo with multiple questions."""
    print("\n🎮 Running Quick Demo with Mistral...")
    
    try:
        from bot.jupiter_faq_bot import create_jupiter_bot
        
        bot = create_jupiter_bot(config_file="config.json")
        
        questions = [
            "How can I open a Jupiter account?",
            "What is Jupiter Pro?",
            "How do I complete KYC?"
        ]
        
        for i, question in enumerate(questions, 1):
            print(f"\n--- Q{i}: {question} ---")
            response = bot.ask(question)
            print(f"🤖 {response['response'][:120]}...")
            print(f"📊 Confidence: {response['confidence']:.1%} | Provider: {response['provider']}")
            
    except Exception as e:
        print(f"❌ Demo error: {e}")

if __name__ == "__main__":
    success = test_openrouter()
    
    if success:
        run_demo = input("\n🎮 Run demo with multiple questions? (y/n): ").strip().lower()
        if run_demo == 'y':
            demo_questions()
    
    print("\n✅ Test complete!")
    if success:
        print("🚀 Your bot is ready! Try: python demo_bot.py interactive") 