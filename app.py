"""
Local AI Demo Stack - Complete Application
Voice + Text + Vision AI - 100% Local & Free
"""

import gradio as gr
import numpy as np
import tempfile
import os
from PIL import Image
import threading
import time
import cv2
import base64
import io

from models import WhisperSTT, LocalTTS, LocalLLM, LocalVLM, OllamaLLM
from config import Config

# Global model instances
whisper_model = None
tts_model = None
llm_model = None
vlm_model = None

# Conversation history
conversation_history = []

def initialize_models():
    """Initialize all AI models"""
    global whisper_model, tts_model, llm_model, vlm_model
    
    print("🚀 Initializing Local AI Models...")
    
    try:
        # Initialize models
        whisper_model = WhisperSTT()
        tts_model = LocalTTS()
        
        # Try Ollama first with proper connection testing
        try:
            print("Testing Ollama connection...")
            ollama_llm = OllamaLLM()
            is_available, message = ollama_llm.test_connection()
            
            if is_available:
                llm_model = ollama_llm
                print(f"✅ Using Ollama LLM: {message}")
            else:
                print(f"⚠️ Ollama not available: {message}")
                raise Exception("Ollama not available")
        except:
            print("⚠️ Falling back to local transformers...")
            llm_model = LocalLLM()
        
        # Initialize Vision Model (Ollama priority)
        try:
            vlm_model = LocalVLM()
        except Exception as e:
            print(f"⚠️ Vision model failed to load: {e}")
            print("Vision features will be disabled.")
            vlm_model = None
        
        print("✅ All available models initialized!")
        
    except Exception as e:
        print(f"❌ Error initializing models: {e}")
        print("Some features may not work properly.")

def voice_chat(audio_input):
    """Handle voice-to-voice conversation"""
    if audio_input is None:
        return "Please record some audio first.", None, conversation_history
    
    if whisper_model is None or llm_model is None or tts_model is None:
        return "Voice chat not available - models not loaded.", None, conversation_history
    
    try:
        # Save audio to temporary file
        sample_rate, audio_data = audio_input
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        
        # Convert audio data to proper format
        if audio_data.dtype != np.int16:
            audio_data = (audio_data * 32767).astype(np.int16)
        
        import soundfile as sf
        sf.write(temp_audio.name, audio_data, sample_rate)
        
        # Speech to Text
        user_text = whisper_model.transcribe(temp_audio.name)
        
        # Generate LLM response
        ai_response = llm_model.generate_response(user_text, conversation_history)
        
        # Text to Speech
        response_audio_path = tts_model.speak(ai_response)
        
        # Update conversation history
        conversation_history.append({
            'human': user_text,
            'assistant': ai_response
        })
        
        # Cleanup
        os.unlink(temp_audio.name)
        
        # Format conversation display
        conversation_display = format_conversation_history()
        
        return conversation_display, response_audio_path, conversation_history
        
    except Exception as e:
        return f"Error in voice chat: {str(e)}", None, conversation_history

def text_chat(user_input, history):
    """Handle text-only conversation"""
    if not user_input.strip():
        return history, ""
    
    if llm_model is None:
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": "Text chat not available - LLM model not loaded."})
        return history, ""
    
    try:
        # Generate response
        ai_response = llm_model.generate_response(user_input, conversation_history)
        
        # Update conversation history
        conversation_history.append({
            'human': user_input,
            'assistant': ai_response
        })
        
        # Update gradio chat history
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": ai_response})
        
        return history, ""
        
    except Exception as e:
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": f"Error: {str(e)}"})
        return history, ""

def text_chat_with_tts(user_input, history, enable_tts):
    """Text chat with optional TTS output"""
    history, _ = text_chat(user_input, history)
    
    if enable_tts and len(history) > 0 and tts_model is not None:
        last_response = history[-1]["content"]
        try:
            audio_path = tts_model.speak(last_response)
            return history, "", audio_path
        except:
            return history, "", None
    
    return history, "", None

def analyze_camera_image(image, question):
    """Analyze image from camera using vision model"""
    if image is None:
        return "No image captured. Please take a photo first."
    
    if vlm_model is None:
        return "Vision model not available. Please check model loading."
    
    try:
        # Use custom question or default
        if not question.strip():
            question = "What do you see in this image? Describe it in detail."
        
        # Analyze image with vision model
        analysis = vlm_model.analyze_image(image, question)
        
        return f"🤖 **AI Analysis:**\n\n{analysis}"
        
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

def clear_conversation():
    """Clear conversation history"""
    global conversation_history
    conversation_history = []
    return [], ""

def format_conversation_history():
    """Format conversation history for display"""
    if not conversation_history:
        return "No conversation yet."
    
    formatted = []
    for i, conv in enumerate(conversation_history[-5:], 1):  # Show last 5
        formatted.append(f"**Turn {i}:**")
        formatted.append(f"🗣️ **You:** {conv['human']}")
        formatted.append(f"🤖 **AI:** {conv['assistant']}")
        formatted.append("---")
    
    return "\n".join(formatted)

def create_interface():
    """Create Gradio interface"""
    
    # Custom CSS for better styling and font
    custom_css = """
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global font settings - fallback to system fonts */
    * {
        font-family: 'Helvetica Neue', 'Helvetica', 'DejaVu Sans', 'Arial', sans-serif !important;
    }
    
    /* Main container styling */
    .gradio-container {
        max-width: 1400px !important;
        margin: 0 auto !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        min-height: 100vh;
    }
    
    /* Header styling */
    .main-header {
        background: rgba(255, 255, 255, 0.95) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 15px !important;
        padding: 25px !important;
        margin: 20px 0 !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
    
    /* Tab styling */
    .tab-nav {
        background: rgba(255, 255, 255, 0.9) !important;
        border-radius: 12px !important;
        padding: 8px !important;
        margin: 10px 0 !important;
    }
    
    .tab-nav button {
        border-radius: 8px !important;
        padding: 12px 20px !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
        border: none !important;
        margin: 2px !important;
    }
    
    .tab-nav button.selected {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
    }
    
    /* Card styling for content areas */
    .content-card {
        background: rgba(255, 255, 255, 0.95) !important;
        border-radius: 15px !important;
        padding: 25px !important;
        margin: 15px 0 !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        backdrop-filter: blur(10px) !important;
    }
    
    /* Button styling */
    .btn-primary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
    }
    
    .btn-primary:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
    }
    
    .btn-secondary {
        background: rgba(255, 255, 255, 0.9) !important;
        border: 2px solid #667eea !important;
        color: #667eea !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }
    
    .btn-secondary:hover {
        background: #667eea !important;
        color: white !important;
        transform: translateY(-1px) !important;
    }
    
    /* Input styling */
    .input-field {
        border-radius: 10px !important;
        border: 2px solid rgba(102, 126, 234, 0.2) !important;
        padding: 12px 16px !important;
        font-size: 14px !important;
        transition: all 0.3s ease !important;
        background: rgba(255, 255, 255, 0.9) !important;
    }
    
    .input-field:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
        outline: none !important;
    }
    
    /* Chat styling */
    .chatbot {
        border-radius: 15px !important;
        border: 2px solid rgba(102, 126, 234, 0.2) !important;
        background: rgba(255, 255, 255, 0.95) !important;
    }
    
    /* Audio component styling */
    .audio-component {
        border-radius: 12px !important;
        border: 2px solid rgba(102, 126, 234, 0.2) !important;
        background: rgba(255, 255, 255, 0.9) !important;
        padding: 15px !important;
    }
    
    /* Image component styling */
    .image-component {
        border-radius: 12px !important;
        border: 2px solid rgba(102, 126, 234, 0.2) !important;
        overflow: hidden !important;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-online {
        background: #10b981;
        box-shadow: 0 0 10px rgba(16, 185, 129, 0.5);
    }
    
    .status-offline {
        background: #ef4444;
        box-shadow: 0 0 10px rgba(239, 68, 68, 0.5);
    }
    
    .status-warning {
        background: #f59e0b;
        box-shadow: 0 0 10px rgba(245, 158, 11, 0.5);
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .gradio-container {
            padding: 10px !important;
        }
        
        .content-card {
            padding: 15px !important;
            margin: 10px 0 !important;
        }
        
        .main-header {
            padding: 15px !important;
        }
    }
    
    /* Smooth animations */
    * {
        transition: all 0.3s ease !important;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(102, 126, 234, 0.5);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(102, 126, 234, 0.7);
    }
    """
    
    with gr.Blocks(
        title="🤖 Local AI Demo Stack - Enhanced Interface",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="purple",
            neutral_hue="slate",
            font=["Helvetica Neue", "Helvetica", "DejaVu Sans", "Arial", "sans-serif"]
        ),
        css=custom_css
    ) as demo:
        
        # Enhanced header with better styling
        gr.HTML("""
        <div class="main-header">
            <div style="text-align: center;">
                <h1 style="font-size: 2.5rem; margin: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 700;">
                    🤖 Local AI Demo Stack
                </h1>
                <p style="font-size: 1.2rem; margin: 10px 0; color: #666; font-weight: 500;">
                    Enhanced Interface - 100% Free & Local AI
                </p>
            </div>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-top: 25px;">
                <div style="text-align: center; padding: 15px; background: rgba(102, 126, 234, 0.1); border-radius: 10px;">
                    <div style="font-size: 1.5rem; margin-bottom: 8px;">🔒</div>
                    <div style="font-weight: 600; color: #333;">Complete Privacy</div>
                    <div style="font-size: 0.9rem; color: #666;">All processing on your machine</div>
                </div>
                
                <div style="text-align: center; padding: 15px; background: rgba(102, 126, 234, 0.1); border-radius: 10px;">
                    <div style="font-size: 1.5rem; margin-bottom: 8px;">💰</div>
                    <div style="font-weight: 600; color: #333;">Zero Cost</div>
                    <div style="font-size: 0.9rem; color: #666;">No API fees or subscriptions</div>
                </div>
                
                <div style="text-align: center; padding: 15px; background: rgba(102, 126, 234, 0.1); border-radius: 10px;">
                    <div style="font-size: 1.5rem; margin-bottom: 8px;">⚡</div>
                    <div style="font-weight: 600; color: #333;">Lightning Fast</div>
                    <div style="font-size: 0.9rem; color: #666;">No internet required</div>
                </div>
                
                <div style="text-align: center; padding: 15px; background: rgba(102, 126, 234, 0.1); border-radius: 10px;">
                    <div style="font-size: 1.5rem; margin-bottom: 8px;">🌐</div>
                    <div style="font-weight: 600; color: #333;">Network Ready</div>
                    <div style="font-size: 0.9rem; color: #666;">Access from any device</div>
                </div>
            </div>
        </div>
        """)
        
        with gr.Tabs():
            
            # Voice Chat Tab
            with gr.TabItem("🎤 Voice Chat"):
                gr.HTML("""
                <div class="content-card">
                    <h3 style="margin-top: 0; color: #333; font-weight: 600;">🎤 Voice-to-Voice Conversation</h3>
                    <p style="color: #666; margin-bottom: 20px; font-size: 1.1rem;">
                        <strong>How it works:</strong> Record your voice → AI transcribes → Generates intelligent response → Speaks back to you
                    </p>
                    <div style="background: rgba(102, 126, 234, 0.1); padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                        <strong>💡 Pro Tips:</strong>
                        <ul style="margin: 10px 0; padding-left: 20px;">
                            <li>Speak clearly and at normal pace</li>
                            <li>Wait for the recording to finish before speaking</li>
                            <li>Use headphones to prevent audio feedback</li>
                        </ul>
                    </div>
                </div>
                """)
                
                with gr.Row():
                    with gr.Column():
                        # Enhanced audio input for network compatibility
                        audio_input = gr.Audio(
                            sources=["microphone"],
                            type="numpy",
                            label="🎙️ Record your voice",
                            streaming=False,
                            show_download_button=False,
                            interactive=True,
                            elem_classes=["audio-component"]
                        )
                        voice_btn = gr.Button("💬 Process Voice", variant="primary", size="lg", elem_classes=["btn-primary"])
                    
                    with gr.Column():
                        conversation_display = gr.Markdown(
                            value="No conversation yet.",
                            label="📝 Conversation"
                        )
                        audio_output = gr.Audio(
                            label="🔊 AI Response",
                            autoplay=True,
                            show_download_button=True,
                            elem_classes=["audio-component"]
                        )
                
                voice_btn.click(
                    voice_chat,
                    inputs=[audio_input],
                    outputs=[conversation_display, audio_output, gr.State()]
                )
            
            # Text Chat Tab
            with gr.TabItem("💬 Text Chat"):
                gr.HTML("""
                <div class="content-card">
                    <h3 style="margin-top: 0; color: #333; font-weight: 600;">💬 Smart Text Conversation</h3>
                    <p style="color: #666; margin-bottom: 20px; font-size: 1.1rem;">
                        <strong>Chat with AI:</strong> Type your questions or messages and get intelligent responses. Enable voice output to hear the AI speak!
                    </p>
                    <div style="background: rgba(102, 126, 234, 0.1); padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                        <strong>✨ Features:</strong>
                        <ul style="margin: 10px 0; padding-left: 20px;">
                            <li><strong>Smart Responses:</strong> Context-aware AI conversations</li>
                            <li><strong>Voice Output:</strong> Toggle to hear responses spoken aloud</li>
                            <li><strong>Chat History:</strong> Maintains conversation context</li>
                            <li><strong>Quick Send:</strong> Press Enter to send messages</li>
                        </ul>
                    </div>
                </div>
                """)
                
                with gr.Row():
                    with gr.Column(scale=4):
                        chatbot = gr.Chatbot(
                            label="💭 Conversation",
                            height=400,
                            type="messages",
                            elem_classes=["chatbot"]
                        )
                        
                        with gr.Row():
                            msg = gr.Textbox(
                                placeholder="Type your message here...",
                                label="✍️ Your message",
                                scale=4,
                                elem_classes=["input-field"]
                            )
                            send_btn = gr.Button("📤 Send", variant="primary", elem_classes=["btn-primary"])
                    
                    with gr.Column(scale=1):
                        enable_tts = gr.Checkbox(
                            label="🔊 Enable voice output",
                            value=False
                        )
                        text_audio_output = gr.Audio(
                            label="🎵 Voice Response",
                            autoplay=True
                        )
                        clear_btn = gr.Button("🗑️ Clear Chat", elem_classes=["btn-secondary"])
                
                # Event handlers
                send_btn.click(
                    text_chat_with_tts,
                    inputs=[msg, chatbot, enable_tts],
                    outputs=[chatbot, msg, text_audio_output]
                )
                
                msg.submit(
                    text_chat_with_tts,
                    inputs=[msg, chatbot, enable_tts],
                    outputs=[chatbot, msg, text_audio_output]
                )
                
                clear_btn.click(
                    clear_conversation,
                    outputs=[chatbot, msg]
                )
            
            # Camera Vision Tab
            with gr.TabItem("📹 Camera Vision"):
                gr.HTML("""
                <div class="content-card">
                    <h3 style="margin-top: 0; color: #333; font-weight: 600;">📹 AI Vision Analysis</h3>
                    <p style="color: #666; margin-bottom: 20px; font-size: 1.1rem;">
                        <strong>See through AI eyes:</strong> Take photos with your camera and get detailed AI analysis of what's in the image
                    </p>
                    <div style="background: rgba(102, 126, 234, 0.1); padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                        <strong>🔍 What AI can see:</strong>
                        <ul style="margin: 10px 0; padding-left: 20px;">
                            <li><strong>Objects & People:</strong> Identify and describe what's in the scene</li>
                            <li><strong>Colors & Details:</strong> Analyze visual elements and composition</li>
                            <li><strong>Text Reading:</strong> Read and transcribe text in images</li>
                            <li><strong>Custom Questions:</strong> Ask specific questions about the photo</li>
                        </ul>
                    </div>
                </div>
                """)
                
                # Show model status
                if vlm_model:
                    model_type = getattr(vlm_model, 'model_type', 'unknown')
                    if model_type == "ollama_vision":
                        status_msg = "🚀 **Ollama LLaVA** - Advanced vision analysis ready"
                    else:
                        status_msg = "✅ **Basic Vision** - Image analysis available"
                else:
                    status_msg = "❌ **Vision not available** - Install with: `ollama pull llava:7b`"
                
                gr.Markdown(f"**Status:** {status_msg}")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        # Camera input - this will open device camera
                        camera_input = gr.Image(
                            sources=["webcam"],  # This enables camera access
                            type="pil",
                            label="📹 Camera (Click camera icon to take photo)",
                            height=400,
                            interactive=True,
                            elem_classes=["image-component"]
                        )
                        
                        # Analyze button
                        analyze_btn = gr.Button("🔍 Analyze Photo", variant="primary", size="lg", elem_classes=["btn-primary"])
                        
                        # Instructions
                        gr.Markdown("""
                        **📹 How to use:**
                        1. **Click the camera icon** in the image box above
                        2. **Allow camera access** when prompted by browser
                        3. **Take a photo** by clicking the capture button
                        4. **Ask your question** in the text box →
                        5. **Click 'Analyze Photo'** to get AI analysis
                        
                        **💡 Camera Tips:**
                        - For network access, use: `https://localhost:7860` (if available)
                        - Or access locally: `http://localhost:7860`
                        - Camera works best on local access
                        """)
                    
                    with gr.Column(scale=1):
                        # Question input
                        camera_question = gr.Textbox(
                            placeholder="Ask about the photo: 'Is this a ghost?', 'What colors do you see?', 'Describe the scene in detail'...",
                            label="❓ Your question about the photo",
                            value="What do you see in this image? Describe it in detail.",
                            lines=4
                        )
                        
                        # Analysis output
                        camera_analysis = gr.Textbox(
                            label="🤖 AI Vision Analysis",
                            lines=15,
                            placeholder="Take a photo and click 'Analyze Photo' to see AI analysis here...",
                            max_lines=25
                        )
                        
                        # Camera troubleshooting
                        gr.Markdown("""
                        **🔧 Camera Troubleshooting:**
                        
                        **If camera doesn't work:**
                        - **Local access**: Use `http://localhost:7860`
                        - **HTTPS needed**: For network access, camera requires HTTPS
                        - **Browser permissions**: Allow camera access when prompted
                        - **Close other apps**: Make sure camera isn't used elsewhere
                        
                        **Alternative:**
                        - Use **"Upload"** option to analyze existing photos
                        - Take photo with phone, then upload here
                        """)
                
                # Event handler for analysis
                analyze_btn.click(
                    analyze_camera_image,
                    inputs=[camera_input, camera_question],
                    outputs=[camera_analysis]
                )
            
            # Settings Tab
            with gr.TabItem("⚙️ Settings"):
                gr.HTML("""
                <div class="content-card">
                    <h3 style="margin-top: 0; color: #333; font-weight: 600;">⚙️ System Status & Information</h3>
                    <p style="color: #666; margin-bottom: 20px; font-size: 1.1rem;">
                        <strong>Monitor your AI system:</strong> Check which models are loaded and get troubleshooting tips
                    </p>
                </div>
                """)
                
                with gr.Row():
                    with gr.Column():
                        # Determine model statuses
                        llm_status = "✅ Ollama" if isinstance(llm_model, OllamaLLM) else "✅ Local Transformers" if llm_model else "❌ Not Loaded"
                        
                        if vlm_model:
                            model_type = getattr(vlm_model, 'model_type', 'unknown')
                            if model_type == "ollama_vision":
                                vision_status = "🚀 Ollama LLaVA"
                            else:
                                vision_status = "✅ Basic BLIP"
                        else:
                            vision_status = "❌ Not Available"
                        
                        gr.Markdown(f"""
                        **Configuration:**
                        - Device: `{Config.DEVICE}`
                        - LLM Status: `{llm_status}`
                        - Whisper Model: `{Config.WHISPER_MODEL}`
                        - Vision Status: `{vision_status}`
                        - Network: `Accessible on local network`
                        - Camera: `Browser-based camera access`
                        
                        **Features:**
                        - ✅ Speech-to-Text (Whisper)
                        - {llm_status.split()[0]} Text Generation
                        - ✅ Text-to-Speech (pyttsx3)
                        - {vision_status.split()[0]} Camera Vision
                        - ✅ 100% Local Processing
                        - 🌐 Network Accessible
                        - 📹 Browser Camera Integration
                        """)
                    
                    with gr.Column():
                        gr.Markdown("""
                        **Camera Access Solutions:**
                        
                        **🏠 Local Access (Recommended):**
                        - URL: `http://localhost:7860`
                        - Camera: ✅ Works perfectly
                        - Voice: ✅ Works perfectly
                        
                        **🌐 Network Access:**
                        - URL: `http://[YOUR-IP]:7860`
                        - Camera: ⚠️ May need HTTPS for camera
                        - Voice: ✅ Works with enhanced compatibility
                        
                        **📱 Mobile Access:**
                        - Camera: ✅ Works on mobile browsers
                        - Voice: ✅ Works on mobile
                        - Upload: ✅ Alternative to camera
                        
                        **🔧 HTTPS Workaround:**
                        - Use localhost for camera access
                        - Or upload photos instead of live camera
                        - Mobile browsers often allow camera over HTTP
                        """)
        
        # Enhanced footer
        gr.HTML("""
        <div style="margin-top: 40px; padding: 25px; background: rgba(255, 255, 255, 0.95); border-radius: 15px; text-align: center; backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.2);">
            <h3 style="margin: 0 0 15px 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 600;">
                🚀 Local AI Demo Stack - Enhanced Edition
            </h3>
            <div style="display: flex; justify-content: center; gap: 30px; flex-wrap: wrap; margin-top: 15px;">
                <span style="color: #666; font-weight: 500;">🔒 Complete Privacy</span>
                <span style="color: #666; font-weight: 500;">💰 Zero API Costs</span>
                <span style="color: #666; font-weight: 500;">⚡ Lightning Fast</span>
                <span style="color: #666; font-weight: 500;">🌐 Network Ready</span>
                <span style="color: #666; font-weight: 500;">📹 Camera & Voice</span>
            </div>
            <p style="margin: 15px 0 0 0; color: #888; font-size: 0.9rem;">
                Enhanced with modern UI, improved fonts, and better user experience
            </p>
        </div>
        """)
    
    return demo

def main():
    """Main application entry point"""
    print("🤖 Starting Local AI Demo Stack...")
    
    # Initialize models in background
    init_thread = threading.Thread(target=initialize_models)
    init_thread.start()
    
    # Create and launch interface
    demo = create_interface()
    
    # Wait for models to load
    init_thread.join()
    
    # Find free port
    free_port = Config.find_free_port(Config.GRADIO_PORT)
    if free_port != Config.GRADIO_PORT:
        print(f"⚠️ Port {Config.GRADIO_PORT} is busy, using port {free_port}")
    
    # Get local IP for network access
    local_ip = Config.get_local_ip()
    
    print(f"🌐 Launching interface...")
    print(f"   Local:   http://localhost:{free_port}")
    print(f"   Network: http://{local_ip}:{free_port}")
    print(f"   Mobile:  http://{local_ip}:{free_port}")
    print(f"")
    print(f"📹 Camera Access:")
    print(f"   ✅ Local: Camera works perfectly at localhost")
    print(f"   ⚠️ Network: Camera may need HTTPS for network access")
    print(f"   💡 Tip: Use localhost for best camera experience")
    print(f"")
    print(f"🎙️ Voice: Network-compatible recording enabled")
    
    demo.launch(
        server_name=Config.GRADIO_HOST,  # Bind to all interfaces
        server_port=free_port,
        share=Config.GRADIO_SHARE,
        inbrowser=True
    )

if __name__ == "__main__":
    main()