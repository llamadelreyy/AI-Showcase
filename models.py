"""
Local AI Model Handlers
"""

import whisper
import torch
import pyttsx3
import numpy as np
from PIL import Image
import requests
import tempfile
import os
import base64
import io
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import Config

class WhisperSTT:
    """Local Whisper Speech-to-Text"""
    
    def __init__(self):
        print("Loading Whisper model...")
        self.model = whisper.load_model(Config.WHISPER_MODEL)
        print("Whisper model loaded!")
    
    def transcribe(self, audio_path):
        """Transcribe audio file to text"""
        try:
            result = self.model.transcribe(audio_path)
            return result["text"].strip()
        except Exception as e:
            return f"Error transcribing audio: {str(e)}"

class LocalTTS:
    """Enhanced Local Text-to-Speech with multiple high-quality options"""
    
    def __init__(self):
        self.tts_engine = None
        self.engine_type = None
        
        # Try to initialize the best available TTS engine
        self._initialize_best_tts()
    
    def _initialize_best_tts(self):
        """Initialize the best available TTS engine"""
        
        # Option 1: Try edge-tts (Microsoft Edge TTS - High Quality)
        try:
            import edge_tts
            self.tts_engine = edge_tts
            self.engine_type = "edge_tts"
            print("✅ Using Microsoft Edge TTS (High Quality)")
            return
        except ImportError:
            print("⚠️ edge-tts not available, trying next option...")
        
        # Option 2: Try Coqui TTS (Neural TTS - Very High Quality)
        try:
            from TTS.api import TTS
            # Use a fast, good quality model
            model_name = "tts_models/en/ljspeech/tacotron2-DDC"
            self.tts_engine = TTS(model_name=model_name, progress_bar=False)
            self.engine_type = "coqui_tts"
            print("✅ Using Coqui TTS (Neural - Very High Quality)")
            return
        except ImportError:
            print("⚠️ Coqui TTS not available, trying next option...")
        except Exception as e:
            print(f"⚠️ Coqui TTS failed to load: {e}")
        
        # Option 3: Try gTTS (Google TTS - Good Quality, requires internet)
        try:
            from gtts import gTTS
            self.tts_engine = gTTS
            self.engine_type = "gtts"
            print("✅ Using Google TTS (Good Quality - requires internet)")
            return
        except ImportError:
            print("⚠️ gTTS not available, falling back to pyttsx3...")
        
        # Fallback: pyttsx3 (Basic quality but always available)
        try:
            import pyttsx3
            self.tts_engine = pyttsx3.init()
            # Configure voice settings for better quality
            voices = self.tts_engine.getProperty('voices')
            if voices:
                # Try to find a better voice (prefer female voices as they often sound better)
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
                else:
                    self.tts_engine.setProperty('voice', voices[0].id)
            
            self.tts_engine.setProperty('rate', 180)  # Slightly faster
            self.tts_engine.setProperty('volume', 0.9)
            self.engine_type = "pyttsx3"
            print("✅ Using pyttsx3 TTS (Basic Quality)")
        except Exception as e:
            print(f"❌ All TTS engines failed: {e}")
            self.tts_engine = None
            self.engine_type = None
    
    def speak(self, text):
        """Convert text to speech and save as audio file"""
        if not self.tts_engine:
            return None
        
        try:
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_path = temp_file.name
            temp_file.close()
            
            if self.engine_type == "edge_tts":
                return self._speak_edge_tts(text, temp_path)
            elif self.engine_type == "coqui_tts":
                return self._speak_coqui_tts(text, temp_path)
            elif self.engine_type == "gtts":
                return self._speak_gtts(text, temp_path)
            elif self.engine_type == "pyttsx3":
                return self._speak_pyttsx3(text, temp_path)
            
        except Exception as e:
            print(f"TTS Error: {e}")
            return None
    
    def _speak_edge_tts(self, text, output_path):
        """Use Microsoft Edge TTS (High Quality)"""
        try:
            import asyncio
            import edge_tts
            
            async def generate_speech():
                # Use a high-quality voice
                voice = "en-US-AriaNeural"  # Natural sounding female voice
                communicate = edge_tts.Communicate(text, voice)
                await communicate.save(output_path)
            
            # Run the async function
            asyncio.run(generate_speech())
            return output_path
            
        except Exception as e:
            print(f"Edge TTS error: {e}")
            return None
    
    def _speak_coqui_tts(self, text, output_path):
        """Use Coqui TTS (Neural - Very High Quality)"""
        try:
            self.tts_engine.tts_to_file(text=text, file_path=output_path)
            return output_path
        except Exception as e:
            print(f"Coqui TTS error: {e}")
            return None
    
    def _speak_gtts(self, text, output_path):
        """Use Google TTS (Good Quality)"""
        try:
            from gtts import gTTS
            import pygame
            
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(output_path)
            return output_path
            
        except Exception as e:
            print(f"gTTS error: {e}")
            return None
    
    def _speak_pyttsx3(self, text, output_path):
        """Use pyttsx3 TTS (Basic Quality - Fallback)"""
        try:
            self.tts_engine.save_to_file(text, output_path)
            self.tts_engine.runAndWait()
            return output_path
        except Exception as e:
            print(f"pyttsx3 error: {e}")
            return None

class LocalLLM:
    """Local Language Model using Transformers"""
    
    def __init__(self):
        print("Loading LLM model...")
        self.tokenizer = AutoTokenizer.from_pretrained(Config.LLM_MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(
            Config.LLM_MODEL,
            torch_dtype=torch.float16 if Config.DEVICE == "cuda" else torch.float32,
            device_map="auto" if Config.DEVICE == "cuda" else None
        )
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("LLM model loaded!")
    
    def generate_response(self, prompt, conversation_history=None):
        """Generate response from text prompt"""
        try:
            # Build conversation context
            if conversation_history:
                context = "\n".join([f"Human: {h['human']}\nAssistant: {h['assistant']}" 
                                   for h in conversation_history[-Config.MAX_HISTORY:]])
                full_prompt = f"{context}\nHuman: {prompt}\nAssistant:"
            else:
                full_prompt = f"Human: {prompt}\nAssistant:"
            
            # Tokenize input
            inputs = self.tokenizer.encode(full_prompt, return_tensors="pt")
            if Config.DEVICE == "cuda":
                inputs = inputs.to("cuda")
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=Config.MAX_TOKENS,
                    temperature=Config.TEMPERATURE,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    attention_mask=torch.ones_like(inputs)
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the new response
            response = response.split("Assistant:")[-1].strip()
            
            return response
            
        except Exception as e:
            return f"Error generating response: {str(e)}"

class OllamaVLM:
    """Ollama Vision Language Model - Efficient and Fast"""
    
    def __init__(self):
        self.base_url = Config.OLLAMA_BASE_URL
        self.vision_models = [
            "llava:7b",
            "llava:13b", 
            "llava:34b",
            "bakllava",
            "llava-llama3",
            "llava-phi3"
        ]
        self.model = None
        self.model_type = "ollama_vision"
        
        # Test connection and find available vision model
        available_model = self._find_vision_model()
        if available_model:
            self.model = available_model
            print(f"✅ Ollama Vision model loaded: {self.model}")
        else:
            raise Exception("No Ollama vision models available")
    
    def _find_vision_model(self):
        """Find available vision model in Ollama"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                return None
            
            available_models = [model['name'] for model in response.json().get('models', [])]
            
            # Check for vision models in order of preference
            for vision_model in self.vision_models:
                for available in available_models:
                    if vision_model in available.lower():
                        return available
            
            return None
            
        except Exception as e:
            print(f"Error checking Ollama models: {e}")
            return None
    
    def _image_to_base64(self, image):
        """Convert PIL image to base64 string"""
        if isinstance(image, str):
            # If it's a file path
            with open(image, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode('utf-8')
        else:
            # If it's a PIL Image
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def analyze_image(self, image, question="What do you see in this image?"):
        """Analyze image using Ollama vision model"""
        try:
            if not self.model:
                return "No vision model available in Ollama"
            
            # Convert image to base64
            image_b64 = self._image_to_base64(image)
            
            # Prepare the request
            payload = {
                "model": self.model,
                "prompt": question,
                "images": [image_b64],
                "stream": False
            }
            
            # Make request to Ollama
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "No response from vision model")
            else:
                return f"Ollama vision error: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Error with Ollama vision: {str(e)}"

class LocalVLM:
    """Vision Language Model with Ollama priority"""
    
    def __init__(self):
        print("Loading Vision model...")
        
        # Try Ollama vision first
        try:
            self.vlm = OllamaVLM()
            self.model_type = "ollama_vision"
            print("✅ Using Ollama Vision model!")
            return
        except Exception as e:
            print(f"Ollama vision not available: {e}")
        
        # Fallback to lightweight transformers model
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            print("Loading lightweight BLIP model as fallback...")
            
            model_name = "Salesforce/blip-image-captioning-base"
            self.processor = BlipProcessor.from_pretrained(model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if Config.DEVICE == "cuda" else torch.float32,
                device_map="auto" if Config.DEVICE == "cuda" else None
            )
            self.vlm = None
            self.model_type = "blip_basic"
            print("✅ Lightweight BLIP vision model loaded!")
            
        except Exception as e:
            print(f"All vision models failed: {e}")
            raise e
    
    def analyze_image(self, image_path, question="What do you see in this image?"):
        """Analyze image with best available model"""
        try:
            if self.model_type == "ollama_vision":
                # Use Ollama vision
                return self.vlm.analyze_image(image_path, question)
            
            else:
                # Use BLIP fallback
                if isinstance(image_path, str):
                    image = Image.open(image_path).convert('RGB')
                else:
                    image = image_path.convert('RGB')
                
                inputs = self.processor(image, question, return_tensors="pt")
                if Config.DEVICE == "cuda":
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, max_new_tokens=100)
                
                description = self.processor.decode(outputs[0], skip_special_tokens=True)
                return description
            
        except Exception as e:
            return f"Error analyzing image: {str(e)}"

class OllamaLLM:
    """Enhanced Ollama LLM integration with smart model detection"""
    
    def __init__(self):
        self.base_url = Config.OLLAMA_BASE_URL
        self.model = None
        
    def test_connection(self):
        """Test if Ollama is available and find best model"""
        try:
            # Test basic connection
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                return False, "Ollama API not responding"
            
            # Get available models
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            
            if not model_names:
                return False, "No models available in Ollama"
            
            # Prioritized model selection
            preferred_models = [
                # High-quality instruction models
                "myaniu/qwen2.5-1m:7b-instruct-q8_0", 
                "command-a:latest",
                # Standard models
                "llama2:7b-chat",
                "llama2:13b-chat",
                "gemma3:27b-it-fp16",
                "gemma3:4b",
                # Any other available model
            ]
            
            # Find the best available model
            for preferred in preferred_models:
                if preferred in model_names:
                    self.model = preferred
                    return True, f"Using high-quality model: {preferred}"
            
            # If no preferred model found, use the first available
            self.model = model_names[0]
            return True, f"Using available model: {self.model}"
            
        except requests.exceptions.RequestException as e:
            return False, f"Connection error: {e}"
        except Exception as e:
            return False, f"Unexpected error: {e}"
    
    def generate_response(self, prompt, conversation_history=None):
        """Generate response using Ollama API"""
        try:
            if not self.model:
                return "No model selected"
            
            # Build conversation context
            messages = []
            if conversation_history:
                for h in conversation_history[-Config.MAX_HISTORY:]:
                    messages.append({"role": "user", "content": h['human']})
                    messages.append({"role": "assistant", "content": h['assistant']})
            
            messages.append({"role": "user", "content": prompt})
            
            # Call Ollama API
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()["message"]["content"]
            else:
                return f"Ollama API error: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Error connecting to Ollama: {str(e)}"