#!/bin/bash

# Optional Performance Packages Installation Script
echo "🚀 Installing Optional Performance Packages"
echo "=========================================="

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "🔄 Activating virtual environment..."
    source venv/bin/activate
fi

echo "📦 Installing performance optimization packages..."

# Install flash-attn (optional, for faster attention)
echo "⚡ Installing flash-attn (this may take a while)..."
if pip install flash-attn>=2.5.0; then
    echo "✅ flash-attn installed successfully"
else
    echo "⚠️ flash-attn installation failed - this is optional and the system will work without it"
fi

# Install deepspeed (optional, for large model optimization)
echo "🔧 Installing deepspeed..."
if pip install deepspeed>=0.12.0; then
    echo "✅ deepspeed installed successfully"
else
    echo "⚠️ deepspeed installation failed - this is optional and the system will work without it"
fi

echo ""
echo "🎉 Performance package installation complete!"
echo ""
echo "📝 Note: These packages are optional optimizations."
echo "   Your AI models will work perfectly without them."
echo "   They just provide additional speed improvements."