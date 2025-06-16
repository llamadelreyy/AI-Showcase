#!/bin/bash

# Local AI Demo Stack - Quick Start Script
echo "🤖 Local AI Demo Stack - Quick Start"
echo "===================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    echo "Please install Python 3.8+ and try again."
    exit 1
fi

echo "✅ Python found: $(python3 --version)"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is required but not installed."
    echo "Please install pip and try again."
    exit 1
fi

echo "✅ pip found"

# Kill any existing processes on port 7860
echo "🔄 Checking for existing processes on port 7860..."
if lsof -ti:7860 >/dev/null 2>&1; then
    echo "⚠️ Found existing process on port 7860, stopping it..."
    kill -9 $(lsof -ti:7860) 2>/dev/null || true
    sleep 2
fi

# Create virtual environment (optional but recommended)
echo "📦 Setting up virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📥 Installing optimized requirements (faster installation)..."
if [ -f "requirements_optimized.txt" ]; then
    echo "Using optimized requirements for faster setup..."
    pip install -r requirements_optimized.txt
else
    echo "Using standard requirements..."
    pip install -r requirements.txt
fi

# Install enhanced TTS separately to avoid conflicts
echo "🔊 Installing enhanced TTS for better voice quality..."
pip install edge-tts gTTS --no-deps || echo "⚠️ Some TTS packages may need manual installation"

# Check if Ollama is available
echo "🦙 Checking for Ollama..."
if command -v ollama &> /dev/null; then
    echo "✅ Ollama found! Checking for models..."
    
    # Check for LLM models
    if ollama list | grep -q "llama2"; then
        echo "✅ Llama2 LLM model found"
    else
        echo "⚠️ No Llama2 model found. You can install it with:"
        echo "   ollama pull llama2:7b-chat"
    fi
    
    # Check for vision models
    if ollama list | grep -q "llava"; then
        echo "✅ LLaVA vision model found"
    else
        echo "🎯 Installing LLaVA vision model for advanced image analysis..."
        echo "   This will enable features like 'Is this a ghost?' questions"
        ollama pull llava:7b
        if [ $? -eq 0 ]; then
            echo "✅ LLaVA vision model installed successfully!"
        else
            echo "⚠️ Failed to install LLaVA. You can install it manually:"
            echo "   ollama pull llava:7b"
        fi
    fi
else
    echo "⚠️ Ollama not found. The app will use local transformers."
    echo "   For better performance and vision features, install Ollama:"
    echo "   curl -fsSL https://ollama.ai/install.sh | sh"
    echo "   ollama pull llama2:7b-chat"
    echo "   ollama pull llava:7b"
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "🚀 Choose your experience:"
echo "1) Interactive AI Demo Tour (Recommended for new users)"
echo "2) Original AI Demo Stack (Free-play mode)"
echo ""
read -p "Enter your choice (1 or 2): " choice

case $choice in
    2)
        echo ""
        echo "🚀 Starting Original AI Demo Stack..."
        echo "📝 Note: Models will download automatically on first use"
        echo "   This may take a few minutes initially."
        echo ""
        echo "🌐 The app will open in your browser automatically"
        echo "   If port 7860 is busy, it will find another free port"
        echo ""
        echo "🎯 Vision Features:"
        echo "   - With Ollama LLaVA: Advanced question answering"
        echo "   - Ask specific questions like 'Is this a ghost?'"
        echo "   - Detailed scene analysis and object recognition"
        echo ""
        echo "Press Ctrl+C to stop the application"
        echo "===================================="
        
        # Start the original application
        python3 app.py
        ;;
    *)
        echo ""
        echo "🚀 Starting Interactive AI Demo Tour..."
echo "📝 Note: Models will download automatically on first use"
echo "   This may take a few minutes initially."
echo ""
echo "🌐 The interactive tour will open in your browser automatically"
echo "   If port 7860 is busy, it will find another free port"
echo ""
echo "🎯 Interactive Tour Features:"
echo "   ✅ Step-by-step guided experience through AI capabilities"
echo "   ✅ Email registration and completion certificates"
echo "   ✅ Hands-on testing of LLM Chat, Vision AI, Speech-to-Text, and TTS"
echo "   ✅ Knowledge assessment quiz with scoring"
echo "   ✅ Back/Next navigation with progress tracking"
echo "   ✅ Modern UI with consistent design and fonts"
echo ""
echo "🎓 Tour Modules:"
echo "   1. Welcome & Introduction"
echo "   2. About Local AI Benefits"
echo "   3. Email Registration"
echo "   4. LLM Chat Experience"
echo "   5. Vision AI Image Analysis"
echo "   6. Speech-to-Text (Whisper)"
echo "   7. Text-to-Speech Demo"
echo "   8. Knowledge Quiz (10 questions)"
echo "   9. Completion & Certificate"
echo ""
echo "⏱️ Duration: 15-20 minutes"
echo "🏆 Completion: Quiz results and participation certificate"
echo ""
echo "Press Ctrl+C to stop the application"
echo "===================================="

        # Start the enhanced interactive tour application
        python3 interactive_tour_enhanced.py
        ;;
esac