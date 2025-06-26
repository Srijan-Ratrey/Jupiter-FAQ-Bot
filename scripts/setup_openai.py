#!/usr/bin/env python3
"""
OpenAI API Setup Script for Jupiter FAQ Bot
This script helps configure your OpenAI API key securely.
"""

import os
import sys
import getpass
from pathlib import Path

def setup_openai_api_key():
    """Setup OpenAI API key for the Jupiter FAQ Bot."""
    
    print("🤖 Jupiter FAQ Bot - OpenAI Setup")
    print("=" * 40)
    print()
    print("This script will help you configure your OpenAI API key.")
    print("Your API key will be stored securely in a .env file.")
    print()
    
    # Get API key from user
    while True:
        api_key = getpass.getpass("Enter your OpenAI API key: ").strip()
        
        if not api_key:
            print("❌ API key cannot be empty. Please try again.")
            continue
            
        if not api_key.startswith("sk-"):
            print("⚠️  OpenAI API keys typically start with 'sk-'. Are you sure this is correct?")
            confirm = input("Continue anyway? (y/N): ").strip().lower()
            if confirm != 'y':
                continue
        
        break
    
    # Create .env file
    env_file = Path(".env")
    env_content = f"OPENAI_API_KEY={api_key}\n"
    
    # If .env exists, update it
    if env_file.exists():
        with open(env_file, 'r') as f:
            lines = f.readlines()
        
        # Remove existing OPENAI_API_KEY lines
        lines = [line for line in lines if not line.startswith('OPENAI_API_KEY=')]
        
        # Add new key
        lines.append(env_content)
        
        with open(env_file, 'w') as f:
            f.writelines(lines)
    else:
        # Create new .env file
        with open(env_file, 'w') as f:
            f.write(env_content)
    
    print()
    print("✅ OpenAI API key configured successfully!")
    print(f"📁 Stored in: {env_file.absolute()}")
    print()
    print("🔧 Configuration Summary:")
    print("  • Primary Provider: OpenAI")
    print("  • Model: gpt-3.5-turbo")
    print("  • API Key: Configured ✓")
    print()
    print("🚀 You can now test the bot with:")
    print("   python scripts/demo_bot.py interactive")
    print()

def test_openai_connection():
    """Test the OpenAI connection."""
    try:
        import openai
        from openai import OpenAI
        
        # Load API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ No API key found. Please run setup first.")
            return False
        
        print("🧪 Testing OpenAI connection...")
        
        client = OpenAI(api_key=api_key)
        
        # Test with a simple request
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'Hello from Jupiter FAQ Bot!'"}
            ],
            max_tokens=50
        )
        
        print("✅ OpenAI connection successful!")
        print(f"📋 Response: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def main():
    """Main function."""
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv()
        test_openai_connection()
    else:
        setup_openai_api_key()

if __name__ == "__main__":
    main() 