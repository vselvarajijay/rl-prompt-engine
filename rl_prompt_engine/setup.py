#!/usr/bin/env python3
"""
Setup script for RL Prompt Engine

This script helps you set up the environment for the RL Prompt Engine package.
"""

import os
import shutil
from pathlib import Path

def setup_environment():
    """Set up the environment for RL Prompt Engine"""
    print("🚀 Setting up RL Prompt Engine...")
    print("=" * 40)
    
    # Check if .env already exists
    env_file = Path(".env")
    template_file = Path("env.template")
    
    if env_file.exists():
        print("✅ .env file already exists")
        return
    
    if not template_file.exists():
        print("❌ env.template file not found")
        return
    
    # Copy template to .env
    try:
        shutil.copy(template_file, env_file)
        print("✅ Created .env file from template")
        print()
        print("📝 Next steps:")
        print("1. Edit .env file and add your OpenAI API key:")
        print("   OPENAI_API_KEY=your-openai-api-key-here")
        print()
        print("2. Get your API key from: https://platform.openai.com/api-keys")
        print()
        print("3. Run the examples:")
        print("   poetry run python usage_examples.py")
        
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")

def check_dependencies():
    """Check if all dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    try:
        import gymnasium
        import stable_baselines3
        import numpy
        import openai
        import dotenv
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Run: poetry install")
        return False

def main():
    """Main setup function"""
    print("🎯 RL Prompt Engine Setup")
    print("=" * 30)
    print()
    
    # Check dependencies
    if not check_dependencies():
        return
    
    print()
    
    # Setup environment
    setup_environment()
    
    print()
    print("🎉 Setup complete!")

if __name__ == "__main__":
    main()
