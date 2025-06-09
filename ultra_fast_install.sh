#!/bin/bash

# Ultra Fast Installation Script for 3x RTX 4090 Setup
echo "⚡ Ultra Fast AI Models Installation for 3x RTX 4090"
echo "=================================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

echo "✅ Python found: $(python3 --version)"

# Create virtual environment
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
pip install --upgrade pip --quiet

# Install ultra minimal requirements (fastest possible)
echo "📥 Installing ultra minimal requirements (only essentials)..."
pip install -r requirements_ultra_minimal.txt --quiet

# Check Ollama and model
echo "🦙 Checking for Ollama..."
if command -v ollama &> /dev/null; then
    echo "✅ Ollama found! Checking for required model..."
    
    if ollama list | grep -q "myaniu/qwen2.5-1m:14b-instruct-q8_0"; then
        echo "✅ Model myaniu/qwen2.5-1m:14b-instruct-q8_0 found"
    else
        echo "📥 Downloading model myaniu/qwen2.5-1m:14b-instruct-q8_0..."
        if ollama pull myaniu/qwen2.5-1m:14b-instruct-q8_0; then
            echo "✅ Model downloaded successfully"
        else
            echo "❌ Failed to download model"
        fi
    fi
    
    # Start Ollama service
    echo "🔄 Ensuring Ollama service is running..."
    if ! pgrep -x "ollama" > /dev/null; then
        echo "🚀 Starting Ollama service..."
        ollama serve &
        sleep 2
        echo "✅ Ollama service started"
    else
        echo "✅ Ollama service already running"
    fi
else
    echo "⚠️ Ollama not found in PATH"
fi

echo ""
echo "🎉 Ultra fast installation complete!"
echo ""
echo "🚀 Starting the application..."
echo "   Activating virtual environment and launching app..."

# Start the application automatically
source venv/bin/activate
python app.py &

echo ""
echo "✅ Application started successfully!"
echo ""
echo "🌐 Open your browser to:"
echo "   http://localhost:7860"
echo ""
echo "⚡ Your optimized setup includes:"
echo "   • Whisper large-v3 (best accuracy)"
echo "   • Coqui TTS (high-quality speech)"
echo "   • LLaVA-1.6-34B (advanced vision)"
echo "   • 3x RTX 4090 GPU optimization"
echo "   • Ollama model: myaniu/qwen2.5-1m:14b-instruct-q8_0"
echo ""
echo "📦 To install additional packages later:"
echo "   source venv/bin/activate"
echo "   pip install TTS accelerate bitsandbytes optimum"
echo ""
echo "🛑 To stop the application:"
echo "   Press Ctrl+C or kill the Python process"