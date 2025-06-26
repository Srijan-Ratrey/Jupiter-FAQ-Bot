#!/usr/bin/env python3
"""
Jupiter FAQ Bot Demo Script
A comprehensive demonstration of the bot's capabilities.
"""

import sys
import os
sys.path.append('src')

from bot.jupiter_faq_bot import create_jupiter_bot

def detailed_demo():
    """Run a detailed demo showing bot capabilities."""
    print("🚀 Jupiter FAQ Bot - Detailed Demo")
    print("=" * 50)
    
    # Create bot with configuration
    bot = create_jupiter_bot(config_file="config.json")
    
    # Show bot info
    info = bot.get_bot_info()
    print(f"📋 Bot Info:")
    print(f"   • Name: {info['name']} v{info['version']}")
    print(f"   • Total FAQs: {info['total_faqs']}")
    print(f"   • Categories: {info['categories']}")
    print(f"   • Embedding Model: {info['config']['embedding_model']}")
    
    # Show categories
    categories = bot.rag_pipeline.get_faq_categories()
    print(f"\n📂 Available Categories ({len(categories)}):")
    for i, cat in enumerate(categories, 1):
        print(f"   {i}. {cat}")
    
    # Demo questions with different styles
    demo_tests = [
        {
            "question": "How can I open a Jupiter account?",
            "style": "conversational",
            "description": "Account opening (conversational style)"
        },
        {
            "question": "What is Jupiter Pro?",
            "style": "brief",
            "description": "Product information (brief style)"
        },
        {
            "question": "How to complete KYC verification?",
            "style": "formal",
            "description": "Process inquiry (formal style)"
        },
        {
            "question": "Customer support contact details",
            "style": "conversational",
            "description": "Support information"
        },
        {
            "question": "How to apply for credit card?",
            "style": "conversational",
            "description": "Credit card application"
        }
    ]
    
    print(f"\n🎯 Testing {len(demo_tests)} scenarios...\n")
    
    for i, test in enumerate(demo_tests, 1):
        print(f"--- Test {i}: {test['description']} ---")
        print(f"Question: {test['question']}")
        print(f"Style: {test['style']}")
        
        response = bot.ask(test['question'], style=test['style'])
        
        print(f"Confidence: {response['confidence']:.1%}")
        print(f"Provider: {response['provider']}")
        print(f"Response: {response['response'][:200]}...")
        
        if response.get('sources'):
            print(f"Sources ({len(response['sources'])}):")
            for j, source in enumerate(response['sources'][:2], 1):
                print(f"  {j}. {source['question']} (sim: {source['similarity']:.2f})")
        
        print("-" * 50)
    
    # Category exploration
    print("\n📂 Category Exploration:")
    if categories:
        sample_category = categories[0]
        category_faqs = bot.get_faq_by_category(sample_category, limit=3)
        print(f"\nSample FAQs from '{sample_category}' category:")
        for i, faq in enumerate(category_faqs, 1):
            print(f"{i}. Q: {faq['question']}")
            print(f"   A: {faq['answer'][:100]}...")
    
    # Final statistics
    final_metrics = bot.get_bot_info()['metrics']
    print(f"\n📊 Demo Statistics:")
    print(f"   • Total queries: {final_metrics['total_queries']}")
    print(f"   • Success rate: {final_metrics['success_rate']:.1%}")
    print(f"   • Average confidence: {final_metrics['avg_confidence']:.1%}")
    
    for provider, count in final_metrics['provider_usage'].items():
        print(f"   • {provider}: {count} queries")
    
    print(f"\n✅ Demo completed successfully!")

def interactive_demo():
    """Run an interactive demo."""
    print("🚀 Jupiter FAQ Bot - Interactive Demo")
    print("=" * 50)
    
    bot = create_jupiter_bot(config_file="config.json")
    
    print("💡 Try asking questions like:")
    print("   • How do I open an account?")
    print("   • What is Jupiter Pro?")
    print("   • How to apply for credit card?")
    print("   • What are the KYC requirements?")
    print("   • Customer support options")
    print("\n(Type 'quit' to exit, 'demo' for guided demo)")
    print("=" * 50)
    
    while True:
        try:
            question = input("\n💬 Ask me anything about Jupiter: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', 'bye']:
                print("👋 Thank you for trying Jupiter FAQ Bot!")
                break
            
            if question.lower() == 'demo':
                detailed_demo()
                continue
            
            # Process question
            response = bot.ask(question)
            
            print(f"\n🤖 Jupiter Assistant: {response['response']}")
            print(f"📊 Confidence: {response['confidence']:.1%}")
            
            if response.get('sources'):
                print(f"📚 Based on {len(response['sources'])} relevant FAQs:")
                for source in response['sources'][:2]:
                    print(f"   • {source['question']} (similarity: {source['similarity']:.2f})")
            
        except KeyboardInterrupt:
            print("\n\n👋 Thanks for using Jupiter FAQ Bot!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "interactive":
            interactive_demo()
        elif sys.argv[1] == "detailed":
            detailed_demo()
        else:
            print("Usage: python demo_bot.py [interactive|detailed]")
    else:
        # Default to interactive
        interactive_demo() 