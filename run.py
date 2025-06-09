#!/usr/bin/env python3
"""
Simple launcher for Local AI Demo Stack
"""

import sys
import subprocess
import os

def check_requirements():
    """Check if requirements are installed"""
    try:
        import gradio
        import torch
        import transformers
        import whisper
        return True
    except ImportError:
        return False

def install_requirements():
    """Install requirements if needed"""
    print("📦 Installing requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def main():
    """Main launcher"""
    print("🤖 Local AI Demo Stack Launcher")
    print("=" * 40)
    
    # Check if requirements are installed
    if not check_requirements():
        print("⚠️ Requirements not found. Installing...")
        try:
            install_requirements()
            print("✅ Requirements installed!")
        except Exception as e:
            print(f"❌ Failed to install requirements: {e}")
            print("Please run: pip install -r requirements.txt")
            sys.exit(1)
    
    # Launch the application
    print("🚀 Starting Local AI Demo Stack...")
    try:
        from app import main as app_main
        app_main()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        print("Please check the logs above for details.")

if __name__ == "__main__":
    main()