"""
Setup script for Local AI Demo Stack
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def download_models():
    """Download required models"""
    print("🤖 Downloading AI models...")
    
    # Whisper models are downloaded automatically on first use
    print("✅ Whisper: Will download on first use")
    
    # Transformers models are downloaded automatically
    print("✅ LLM & Vision models: Will download on first use")
    
    print("📝 Note: Models will be cached locally after first download")

def setup_ollama():
    """Setup instructions for Ollama (optional)"""
    print("\n🦙 Optional: Setup Ollama for better LLM performance")
    print("1. Install Ollama: curl -fsSL https://ollama.ai/install.sh | sh")
    print("2. Pull a model: ollama pull llama2:7b-chat")
    print("3. Ollama will be used automatically if available")

def main():
    """Main setup function"""
    print("🚀 Setting up Local AI Demo Stack...")
    
    try:
        install_requirements()
        download_models()
        setup_ollama()
        
        print("\n✅ Setup complete!")
        print("\n🎯 To run the application:")
        print("   python app.py")
        print(f"\n🌐 Then open: http://localhost:7860")
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()