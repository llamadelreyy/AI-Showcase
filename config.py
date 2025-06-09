"""
Configuration for Local AI Demo Stack
"""

import os
import torch
import socket

class Config:
    # Model Configuration
    LLM_MODEL = "microsoft/DialoGPT-medium"  # Lightweight local model
    WHISPER_MODEL = "base"  # Good balance of speed/accuracy
    VLM_MODEL = "llava:7b"  # Ollama vision model (preferred)
    
    # Ollama Configuration (if using Ollama)
    OLLAMA_BASE_URL = "http://localhost:11434"
    OLLAMA_MODEL = "llama2:7b-chat"
    OLLAMA_VISION_MODELS = ["llava:7b", "llava:13b", "bakllava", "llava-llama3"]
    
    # Audio Settings
    SAMPLE_RATE = 16000
    CHUNK_LENGTH = 30  # seconds
    
    # Device Configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # UI Configuration
    GRADIO_PORT = 7860
    GRADIO_HOST = "0.0.0.0"  # Bind to all interfaces for network access
    GRADIO_SHARE = False  # Keep local but allow network access
    
    # Conversation Settings
    MAX_HISTORY = 10
    MAX_TOKENS = 512
    TEMPERATURE = 0.7
    
    @classmethod
    def find_free_port(cls, start_port=7860):
        """Find a free port starting from the given port"""
        for port in range(start_port, start_port + 100):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('0.0.0.0', port))  # Bind to all interfaces
                    return port
            except OSError:
                continue
        return start_port  # fallback to original port
    
    @classmethod
    def get_local_ip(cls):
        """Get the local IP address"""
        try:
            # Connect to a remote address to determine local IP
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]
                return local_ip
        except Exception:
            return "localhost"