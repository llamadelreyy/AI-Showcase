<<<<<<< HEAD
# AI-Showcase
=======
# 🤖 Local AI Demo Stack

**Complete AI application with Voice, Text, and Vision - 100% Local & Free**

## ✨ Features

- 🎤 **Voice Chat**: Speak → AI transcribes → Generates response → Speaks back
- 💬 **Text Chat**: Type messages with optional voice output  
- 👁️ **Vision Analysis**: Upload images for AI description and analysis
- 🔒 **100% Local**: No external APIs, complete privacy
- 💰 **Free**: No usage costs, open-source models only

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Clone or download this project
cd local-ai-demo

# Install Python packages
pip install -r requirements.txt

# Or run setup script
python setup.py
```

### 2. Run Application
```bash
python app.py
```

### 3. Open Interface
Open your browser to: **http://localhost:7860**

## 🛠️ System Requirements

### Minimum
- **RAM**: 8GB
- **VRAM**: 4GB (GPU recommended)
- **Storage**: 10GB for models
- **Python**: 3.8+

### Recommended
- **RAM**: 16GB+
- **VRAM**: 8GB+ (NVIDIA GPU)
- **Storage**: 20GB
- **OS**: Linux/Windows/macOS

## 🎯 How It Works

### Voice Chat Flow
```
🎙️ Record Audio → 🗣️ Whisper STT → 🧠 Local LLM → 🔊 TTS → 🎵 Audio Response
```

### Text Chat Flow
```
⌨️ Type Message → 🧠 Local LLM → 📝 Text Response (+ Optional 🔊 TTS)
```

### Vision Analysis Flow
```
📷 Upload Image → 👁️ BLIP-2 VLM → 📋 Description & Analysis
```

## 🤖 AI Models Used

| Component | Model | Size | Purpose |
|-----------|-------|------|---------|
| **Speech-to-Text** | OpenAI Whisper | ~1GB | Voice transcription |
| **Language Model** | DialoGPT-medium | ~1.5GB | Text generation |
| **Text-to-Speech** | pyttsx3 | <1MB | Voice synthesis |
| **Vision** | BLIP-2-2.7B | ~3GB | Image analysis |

### Alternative: Ollama Integration
For better LLM performance, install [Ollama](https://ollama.ai):

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull llama2:7b-chat

# App will automatically use Ollama if available
```

## 📱 Interface Overview

### 🎤 Voice Chat Tab
- Record audio with microphone
- Real-time speech-to-text transcription
- AI generates contextual responses
- Text-to-speech audio output
- Conversation history display

### 💬 Text Chat Tab
- Standard chat interface
- Optional voice output toggle
- Conversation memory
- Clear chat functionality

### 👁️ Vision Analysis Tab
- Image upload or camera capture
- Custom question input
- AI-powered image description
- Detailed visual analysis

### ⚙️ Settings Tab
- System information display
- Model configuration details
- Performance tips
- Privacy information

## 🔒 Privacy & Security

- ✅ **100% Local Processing**: All AI runs on your machine
- ✅ **No External APIs**: No data sent to cloud services
- ✅ **Offline Capable**: Works without internet connection
- ✅ **No Data Collection**: Conversations stay private
- ✅ **Open Source**: Fully transparent implementation

## 🎛️ Configuration

Edit [`config.py`](config.py) to customize:

```python
# Model Selection
LLM_MODEL = "microsoft/DialoGPT-medium"
WHISPER_MODEL = "base"  # tiny, base, small, medium
VLM_MODEL = "Salesforce/blip2-opt-2.7b"

# Performance Settings
DEVICE = "cuda"  # or "cpu"
MAX_TOKENS = 512
TEMPERATURE = 0.7

# UI Settings
GRADIO_PORT = 7860
```

## 🚨 Troubleshooting

### Common Issues

**Models downloading slowly?**
- Models download automatically on first use
- Subsequent runs are much faster
- Check internet connection for initial setup

**Out of memory errors?**
- Reduce model sizes in config.py
- Use CPU instead of GPU: `DEVICE = "cpu"`
- Close other applications

**Audio not working?**
- Check microphone permissions
- Install audio drivers
- Try different audio input device

**Ollama connection failed?**
- Install Ollama: `curl -fsSL https://ollama.ai/install.sh | sh`
- Start Ollama service: `ollama serve`
- Pull model: `ollama pull llama2:7b-chat`

### Performance Tips

1. **Use GPU**: Significantly faster inference
2. **Smaller Models**: Trade quality for speed
3. **Close Apps**: Free up RAM and VRAM
4. **SSD Storage**: Faster model loading

## 🛣️ Roadmap

- [ ] Real-time streaming responses
- [ ] Multi-language support
- [ ] Custom model fine-tuning
- [ ] Mobile app version
- [ ] Advanced conversation memory
- [ ] Plugin system for extensions

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make improvements
4. Test thoroughly
5. Submit pull request

## 📄 License

MIT License - Feel free to use and modify!

## 🙏 Acknowledgments

- **OpenAI Whisper**: Speech recognition
- **Hugging Face**: Model hosting and transformers
- **Gradio**: Web interface framework
- **Ollama**: Local LLM serving
- **Open Source Community**: Making AI accessible

---

**🚀 Built with ❤️ for local AI enthusiasts**

*No cloud required • No API costs • Complete control*
