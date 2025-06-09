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
    
    with gr.Blocks(title="Local AI Demo Stack", theme=gr.themes.Soft()) as demo:
        
        gr.Markdown("""
        # 🤖 Local AI Demo Stack
        **100% Free & Local** - Voice, Text & Vision AI
        
        🔒 **Privacy**: All processing happens on your machine  
        💰 **Cost**: Completely free, no API fees  
        ⚡ **Speed**: No internet required after setup  
        🌐 **Network Access**: Available on your local network
        """)
        
        with gr.Tabs():
            
            # Voice Chat Tab
            with gr.TabItem("🎤 Voice Chat"):
                gr.Markdown("### Voice-to-Voice Conversation")
                gr.Markdown("Record your voice → AI transcribes → Generates response → Speaks back")
                
                with gr.Row():
                    with gr.Column():
                        # Enhanced audio input for network compatibility
                        audio_input = gr.Audio(
                            sources=["microphone"], 
                            type="numpy",
                            label="🎙️ Record your voice",
                            streaming=False,
                            show_download_button=False,
                            interactive=True
                        )
                        voice_btn = gr.Button("💬 Process Voice", variant="primary", size="lg")
                    
                    with gr.Column():
                        conversation_display = gr.Markdown(
                            value="No conversation yet.",
                            label="📝 Conversation"
                        )
                        audio_output = gr.Audio(
                            label="🔊 AI Response",
                            autoplay=True,
                            show_download_button=True
                        )
                
                voice_btn.click(
                    voice_chat,
                    inputs=[audio_input],
                    outputs=[conversation_display, audio_output, gr.State()]
                )
            
            # Text Chat Tab
            with gr.TabItem("💬 Text Chat"):
                gr.Markdown("### Text Conversation with Optional Voice Output")
                
                with gr.Row():
                    with gr.Column(scale=4):
                        chatbot = gr.Chatbot(
                            label="💭 Conversation",
                            height=400,
                            type="messages"
                        )
                        
                        with gr.Row():
                            msg = gr.Textbox(
                                placeholder="Type your message here...",
                                label="✍️ Your message",
                                scale=4
                            )
                            send_btn = gr.Button("📤 Send", variant="primary")
                    
                    with gr.Column(scale=1):
                        enable_tts = gr.Checkbox(
                            label="🔊 Enable voice output",
                            value=False
                        )
                        text_audio_output = gr.Audio(
                            label="🎵 Voice Response",
                            autoplay=True
                        )
                        clear_btn = gr.Button("🗑️ Clear Chat")
                
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
                gr.Markdown("### Live Camera Analysis with AI Vision")
                gr.Markdown("📸 **Click camera button → Take photo → AI analyzes what it sees**")
                
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
                            interactive=True
                        )
                        
                        # Analyze button
                        analyze_btn = gr.Button("🔍 Analyze Photo", variant="primary", size="lg")
                        
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
                gr.Markdown("### System Information")
                
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
        
        gr.Markdown("""
        ---
        **🚀 Local AI Demo Stack** - Browser camera integration + network compatibility  
        No internet required • No API costs • Complete privacy • Camera & voice support
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