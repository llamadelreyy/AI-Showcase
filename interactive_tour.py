"""
Interactive AI Demo Tour - Guided Experience
A step-by-step interactive tour through AI capabilities
"""

import gradio as gr
import numpy as np
import tempfile
import os
import json
import time
from datetime import datetime
from PIL import Image
import threading

from models import WhisperSTT, LocalTTS, LocalLLM, LocalVLM, OllamaLLM
from config import Config

# Global model instances
whisper_model = None
tts_model = None
llm_model = None
vlm_model = None

# Session data
session_data = {
    "email": "",
    "start_time": None,
    "current_step": 0,
    "quiz_answers": [],
    "quiz_score": 0
}

# Quiz questions for the final assessment
QUIZ_QUESTIONS = [
    {
        "question": "What does LLM stand for in AI?",
        "options": ["Large Language Model", "Linear Learning Machine", "Local Logic Module", "Language Learning Method"],
        "correct": 0,
        "module": "LLM Chat"
    },
    {
        "question": "Which AI model is commonly used for speech-to-text conversion?",
        "options": ["GPT", "BERT", "Whisper", "DALL-E"],
        "correct": 2,
        "module": "Whisper"
    },
    {
        "question": "What is the main advantage of running AI models locally?",
        "options": ["Faster internet", "Better graphics", "Privacy and no API costs", "More storage"],
        "correct": 2,
        "module": "General"
    },
    {
        "question": "What does TTS stand for?",
        "options": ["Text-to-Speech", "Time-to-Start", "Type-to-Send", "Talk-to-System"],
        "correct": 0,
        "module": "TTS"
    },
    {
        "question": "Which type of AI model can analyze and describe images?",
        "options": ["Language Model", "Vision Language Model", "Audio Model", "Text Model"],
        "correct": 1,
        "module": "Vision"
    },
    {
        "question": "What is the benefit of using Ollama for AI models?",
        "options": ["Cloud storage", "Local model management", "Internet speed", "Graphics enhancement"],
        "correct": 1,
        "module": "General"
    },
    {
        "question": "In the Voice Chat module, what happens after you record your voice?",
        "options": ["It gets uploaded", "AI transcribes → generates response → speaks back", "It gets saved", "Nothing happens"],
        "correct": 1,
        "module": "Voice Chat"
    },
    {
        "question": "What can the Vision AI analyze in images?",
        "options": ["Only text", "Only colors", "Objects, people, text, and details", "Only faces"],
        "correct": 2,
        "module": "Vision"
    },
    {
        "question": "What is the main purpose of this Local AI Demo Stack?",
        "options": ["Gaming", "Demonstrate local AI capabilities", "Web browsing", "File management"],
        "correct": 1,
        "module": "General"
    },
    {
        "question": "Which port does the application typically run on?",
        "options": ["8080", "3000", "7860", "5000"],
        "correct": 2,
        "module": "Technical"
    }
]

def initialize_models():
    """Initialize all AI models"""
    global whisper_model, tts_model, llm_model, vlm_model
    
    print("🚀 Initializing Local AI Models for Interactive Tour...")
    
    try:
        # Initialize models
        whisper_model = WhisperSTT()
        tts_model = LocalTTS()
        
        # Try Ollama first
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
        
        # Initialize Vision Model
        try:
            vlm_model = LocalVLM()
        except Exception as e:
            print(f"⚠️ Vision model failed to load: {e}")
            vlm_model = None
        
        print("✅ All available models initialized for tour!")
        
    except Exception as e:
        print(f"❌ Error initializing models: {e}")

def create_intro_page():
    """Create the introduction page"""
    return gr.HTML("""
    <div style="text-align: center; padding: 50px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 80vh; color: white; border-radius: 20px;">
        <div style="background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); border-radius: 20px; padding: 40px; max-width: 800px; margin: 0 auto;">
            <h1 style="font-size: 3rem; margin-bottom: 20px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                🤖 Welcome to the AI Demo Tour!
            </h1>
            <p style="font-size: 1.3rem; margin-bottom: 30px; line-height: 1.6;">
                Embark on an interactive journey through the world of Local AI
            </p>
            <div style="background: rgba(255, 255, 255, 0.2); border-radius: 15px; padding: 25px; margin: 30px 0;">
                <h3 style="margin-bottom: 15px;">🎯 What You'll Experience:</h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 20px;">
                    <div style="background: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 10px;">
                        <div style="font-size: 1.5rem;">💬</div>
                        <div style="font-weight: bold;">LLM Chat</div>
                        <div style="font-size: 0.9rem;">Intelligent conversations</div>
                    </div>
                    <div style="background: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 10px;">
                        <div style="font-size: 1.5rem;">👁️</div>
                        <div style="font-weight: bold;">Vision AI</div>
                        <div style="font-size: 0.9rem;">Image analysis</div>
                    </div>
                    <div style="background: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 10px;">
                        <div style="font-size: 1.5rem;">🎤</div>
                        <div style="font-weight: bold;">Speech-to-Text</div>
                        <div style="font-size: 0.9rem;">Voice recognition</div>
                    </div>
                    <div style="background: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 10px;">
                        <div style="font-size: 1.5rem;">🔊</div>
                        <div style="font-weight: bold;">Text-to-Speech</div>
                        <div style="font-size: 0.9rem;">AI voice synthesis</div>
                    </div>
                </div>
            </div>
            <p style="font-size: 1.1rem; margin-top: 30px;">
                ⏱️ <strong>Duration:</strong> 15-20 minutes<br>
                🏆 <strong>Completion:</strong> Quiz & Certificate
            </p>
        </div>
    </div>
    """)

def create_about_page():
    """Create the about page"""
    return gr.HTML("""
    <div style="padding: 40px; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); min-height: 80vh; color: white; border-radius: 20px;">
        <div style="background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); border-radius: 20px; padding: 40px; max-width: 900px; margin: 0 auto;">
            <h2 style="font-size: 2.5rem; text-align: center; margin-bottom: 30px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                🚀 About This AI Demo Stack
            </h2>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin-bottom: 30px;">
                <div style="background: rgba(255, 255, 255, 0.1); padding: 25px; border-radius: 15px;">
                    <h3 style="margin-bottom: 15px;">🔒 Privacy First</h3>
                    <p>All AI processing happens locally on your machine. No data is sent to external servers, ensuring complete privacy and security.</p>
                </div>
                
                <div style="background: rgba(255, 255, 255, 0.1); padding: 25px; border-radius: 15px;">
                    <h3 style="margin-bottom: 15px;">💰 Zero Cost</h3>
                    <p>No API fees, no subscriptions, no hidden costs. Once set up, everything runs completely free on your hardware.</p>
                </div>
                
                <div style="background: rgba(255, 255, 255, 0.1); padding: 25px; border-radius: 15px;">
                    <h3 style="margin-bottom: 15px;">⚡ Lightning Fast</h3>
                    <p>No internet latency. Responses are generated instantly on your local hardware without network delays.</p>
                </div>
                
                <div style="background: rgba(255, 255, 255, 0.1); padding: 25px; border-radius: 15px;">
                    <h3 style="margin-bottom: 15px;">🌐 Network Ready</h3>
                    <p>Access the interface from any device on your network. Perfect for team demonstrations and multi-device usage.</p>
                </div>
            </div>
            
            <div style="background: rgba(255, 255, 255, 0.2); padding: 25px; border-radius: 15px; text-align: center;">
                <h3 style="margin-bottom: 15px;">🎯 What You'll Learn</h3>
                <p style="font-size: 1.1rem; line-height: 1.6;">
                    This interactive tour will guide you through each AI capability, 
                    allowing you to experience firsthand how local AI can transform 
                    your workflow while maintaining privacy and reducing costs.
                </p>
            </div>
        </div>
    </div>
    """)

def create_email_page():
    """Create the email input page"""
    return gr.HTML("""
    <div style="text-align: center; padding: 50px; background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); min-height: 80vh; color: white; border-radius: 20px;">
        <div style="background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); border-radius: 20px; padding: 40px; max-width: 600px; margin: 0 auto;">
            <h2 style="font-size: 2.5rem; margin-bottom: 20px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                📧 Almost Ready!
            </h2>
            <p style="font-size: 1.2rem; margin-bottom: 30px; line-height: 1.6;">
                To personalize your experience and send you a completion certificate, 
                please provide your email address.
            </p>
            
            <div style="background: rgba(255, 255, 255, 0.2); padding: 25px; border-radius: 15px; margin: 20px 0;">
                <h3 style="margin-bottom: 15px;">🏆 What You'll Receive:</h3>
                <ul style="text-align: left; max-width: 400px; margin: 0 auto;">
                    <li style="margin: 10px 0;">✅ Participation certificate</li>
                    <li style="margin: 10px 0;">📊 Your quiz results</li>
                    <li style="margin: 10px 0;">🎯 Personalized recommendations</li>
                    <li style="margin: 10px 0;">📚 Additional learning resources</li>
                </ul>
            </div>
            
            <p style="font-size: 0.9rem; margin-top: 20px; opacity: 0.8;">
                🔒 Your email is only used for this session and certificate delivery. 
                We respect your privacy and won't send spam.
            </p>
        </div>
    </div>
    """)

def handle_email_submission(email):
    """Handle email submission and start session"""
    global session_data
    session_data["email"] = email
    session_data["start_time"] = datetime.now()
    session_data["current_step"] = 1
    
    return (
        gr.update(visible=False),  # Hide email page
        gr.update(visible=True),   # Show LLM module
        f"Session started for {email}"
    )

def create_llm_module():
    """Create the LLM chat module"""
    return gr.HTML("""
    <div style="padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; color: white;">
        <div style="background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); border-radius: 15px; padding: 30px;">
            <h2 style="text-align: center; margin-bottom: 20px; font-size: 2rem;">
                💬 Module 1: LLM Chat Experience
            </h2>
            <p style="text-align: center; font-size: 1.1rem; margin-bottom: 25px;">
                Experience intelligent conversations with a Large Language Model running locally on your machine.
            </p>
            
            <div style="background: rgba(255, 255, 255, 0.1); padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                <h3>🎯 Try These Sample Conversations:</h3>
                <ul style="margin: 15px 0; padding-left: 20px;">
                    <li>"Explain quantum computing in simple terms"</li>
                    <li>"Write a short poem about artificial intelligence"</li>
                    <li>"What are the benefits of local AI processing?"</li>
                    <li>"Help me plan a healthy weekly meal schedule"</li>
                </ul>
            </div>
            
            <div style="background: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 10px;">
                <strong>💡 Notice:</strong> All responses are generated locally - no internet required!
            </div>
        </div>
    </div>
    """)

def llm_chat(message, history):
    """Handle LLM chat interaction"""
    if not message.strip():
        return history, ""
    
    if llm_model is None:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "LLM model not available. Please check model loading."})
        return history, ""
    
    try:
        # Generate response
        ai_response = llm_model.generate_response(message, [])
        
        # Update chat history
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": ai_response})
        
        return history, ""
        
    except Exception as e:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": f"Error: {str(e)}"})
        return history, ""

def create_vision_module():
    """Create the Vision AI module"""
    return gr.HTML("""
    <div style="padding: 30px; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); border-radius: 20px; color: white;">
        <div style="background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); border-radius: 15px; padding: 30px;">
            <h2 style="text-align: center; margin-bottom: 20px; font-size: 2rem;">
                👁️ Module 2: Vision AI Experience
            </h2>
            <p style="text-align: center; font-size: 1.1rem; margin-bottom: 25px;">
                Upload an image and watch AI analyze and describe what it sees with remarkable detail.
            </p>
            
            <div style="background: rgba(255, 255, 255, 0.1); padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                <h3>🔍 What Vision AI Can Analyze:</h3>
                <ul style="margin: 15px 0; padding-left: 20px;">
                    <li><strong>Objects & People:</strong> Identify and describe elements in the scene</li>
                    <li><strong>Text Recognition:</strong> Read and transcribe text within images</li>
                    <li><strong>Colors & Composition:</strong> Analyze visual elements and aesthetics</li>
                    <li><strong>Context Understanding:</strong> Understand relationships between objects</li>
                </ul>
            </div>
            
            <div style="background: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 10px;">
                <strong>📸 Try uploading:</strong> Photos of objects, scenes, documents, or artwork
            </div>
        </div>
    </div>
    """)

def analyze_image(image, question):
    """Analyze uploaded image"""
    if image is None:
        return "Please upload an image first."
    
    if vlm_model is None:
        return "Vision model not available. Please check model loading."
    
    try:
        if not question.strip():
            question = "Describe this image in detail, including objects, people, colors, and any text you can see."
        
        analysis = vlm_model.analyze_image(image, question)
        return f"🤖 **AI Vision Analysis:**\n\n{analysis}"
        
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

def create_whisper_module():
    """Create the Whisper speech-to-text module"""
    return gr.HTML("""
    <div style="padding: 30px; background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); border-radius: 20px; color: white;">
        <div style="background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); border-radius: 15px; padding: 30px;">
            <h2 style="text-align: center; margin-bottom: 20px; font-size: 2rem;">
                🎤 Module 3: Speech-to-Text (Whisper)
            </h2>
            <p style="text-align: center; font-size: 1.1rem; margin-bottom: 25px;">
                Experience OpenAI's Whisper model converting your speech to text with high accuracy.
            </p>
            
            <div style="background: rgba(255, 255, 255, 0.1); padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                <h3>🎯 Try Speaking These Phrases:</h3>
                <ul style="margin: 15px 0; padding-left: 20px;">
                    <li>"Hello, this is a test of the speech recognition system"</li>
                    <li>"Artificial intelligence is transforming how we work"</li>
                    <li>"Local AI processing ensures privacy and speed"</li>
                    <li>Try speaking in different languages or accents</li>
                </ul>
            </div>
            
            <div style="background: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 10px;">
                <strong>🎙️ Tips:</strong> Speak clearly, avoid background noise, and wait for recording to complete
            </div>
        </div>
    </div>
    """)

def transcribe_audio(audio):
    """Transcribe audio using Whisper"""
    if audio is None:
        return "Please record some audio first."
    
    if whisper_model is None:
        return "Whisper model not available."
    
    try:
        # Save audio to temporary file
        sample_rate, audio_data = audio
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        
        # Convert audio data to proper format
        if audio_data.dtype != np.int16:
            audio_data = (audio_data * 32767).astype(np.int16)
        
        import soundfile as sf
        sf.write(temp_audio.name, audio_data, sample_rate)
        
        # Transcribe
        transcription = whisper_model.transcribe(temp_audio.name)
        
        # Cleanup
        os.unlink(temp_audio.name)
        
        return f"🎤 **You said:** {transcription}"
        
    except Exception as e:
        return f"Error transcribing audio: {str(e)}"

def create_tts_module():
    """Create the Text-to-Speech module"""
    return gr.HTML("""
    <div style="padding: 30px; background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); border-radius: 20px; color: white;">
        <div style="background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); border-radius: 15px; padding: 30px;">
            <h2 style="text-align: center; margin-bottom: 20px; font-size: 2rem;">
                🔊 Module 4: Text-to-Speech Experience
            </h2>
            <p style="text-align: center; font-size: 1.1rem; margin-bottom: 25px;">
                Type any text and hear the AI speak it aloud with natural-sounding voice synthesis.
            </p>
            
            <div style="background: rgba(255, 255, 255, 0.1); padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                <h3>🎯 Try These Sample Texts:</h3>
                <ul style="margin: 15px 0; padding-left: 20px;">
                    <li>"Welcome to the future of artificial intelligence"</li>
                    <li>"Local AI processing ensures your data stays private"</li>
                    <li>"This voice is generated entirely on your local machine"</li>
                    <li>Write your own custom message to hear it spoken</li>
                </ul>
            </div>
            
            <div style="background: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 10px;">
                <strong>🔊 Note:</strong> Voice synthesis happens locally - no internet connection needed!
            </div>
        </div>
    </div>
    """)

def text_to_speech(text):
    """Convert text to speech"""
    if not text.strip():
        return "Please enter some text first.", None
    
    if tts_model is None:
        return "TTS model not available.", None
    
    try:
        audio_path = tts_model.speak(text)
        return f"🔊 **Speaking:** {text}", audio_path
        
    except Exception as e:
        return f"Error generating speech: {str(e)}", None

def create_quiz_page():
    """Create the quiz page"""
    return gr.HTML("""
    <div style="padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; color: white;">
        <div style="background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); border-radius: 15px; padding: 30px;">
            <h2 style="text-align: center; margin-bottom: 20px; font-size: 2rem;">
                🧠 Knowledge Check Quiz
            </h2>
            <p style="text-align: center; font-size: 1.1rem; margin-bottom: 25px;">
                Test your understanding of the AI modules you just experienced!
            </p>
            
            <div style="background: rgba(255, 255, 255, 0.1); padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                <h3>📋 Quiz Information:</h3>
                <ul style="margin: 15px 0; padding-left: 20px;">
                    <li><strong>Questions:</strong> 10 multiple choice questions</li>
                    <li><strong>Topics:</strong> LLM Chat, Vision AI, Whisper, TTS, and General AI</li>
                    <li><strong>Time:</strong> No time limit - take your time</li>
                    <li><strong>Scoring:</strong> Results shown immediately after completion</li>
                </ul>
            </div>
            
            <div style="background: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 10px;">
                <strong>🎯 Ready?</strong> Click "Start Quiz" below to begin your assessment!
            </div>
        </div>
    </div>
    """)

def submit_quiz_answer(question_idx, selected_option):
    """Submit a quiz answer"""
    global session_data
    
    if question_idx < len(QUIZ_QUESTIONS):
        is_correct = selected_option == QUIZ_QUESTIONS[question_idx]["correct"]
        session_data["quiz_answers"].append({
            "question": question_idx,
            "selected": selected_option,
            "correct": is_correct
        })
        
        if is_correct:
            session_data["quiz_score"] += 1
    
    return f"Answer recorded for question {question_idx + 1}"

def create_completion_page():
    """Create the completion page"""
    score = session_data["quiz_score"]
    total = len(QUIZ_QUESTIONS)
    percentage = (score / total) * 100
    
    return gr.HTML(f"""
    <div style="text-align: center; padding: 50px; background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); min-height: 80vh; color: white; border-radius: 20px;">
        <div style="background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); border-radius: 20px; padding: 40px; max-width: 700px; margin: 0 auto;">
            <h1 style="font-size: 3rem; margin-bottom: 20px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                🎉 Congratulations!
            </h1>
            <h2 style="font-size: 1.8rem; margin-bottom: 30px;">
                Thank you for participating in our AI Demo Tour!
            </h2>
            
            <div style="background: rgba(255, 255, 255, 0.2); border-radius: 15px; padding: 25px; margin: 30px 0;">
                <h3 style="margin-bottom: 15px;">📊 Your Results:</h3>
                <div style="font-size: 2rem; margin: 15px 0;">
                    <strong>{score}/{total}</strong> ({percentage:.0f}%)
                </div>
                <div style="background: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 10px; margin: 15px 0;">
                    {'🏆 Excellent work!' if percentage >= 80 else '👍 Good job!' if percentage >= 60 else '📚 Keep learning!'}
                </div>
            </div>
            
            <div style="background: rgba(255, 255, 255, 0.2); border-radius: 15px; padding: 25px; margin: 30px 0;">
                <h3 style="margin-bottom: 15px;">📧 Certificate Delivery</h3>
                <p style="font-size: 1.1rem; line-height: 1.6;">
                    A participation certificate and detailed results have been sent to:<br>
                    <strong>{session_data["email"]}</strong>
                </p>
                <p style="font-size: 0.9rem; margin-top: 15px; opacity: 0.8;">
                    Please check your email (including spam folder) within the next few minutes.
                </p>
            </div>
            
            <div style="background: rgba(255, 255, 255, 0.1); padding: 20px; border-radius: 10px; margin: 20px 0;">
                <h3>🚀 What's Next?</h3>
                <ul style="text-align: left; max-width: 500px; margin: 0 auto;">
                    <li style="margin: 8px 0;">Explore the full interface in free-play mode</li>
                    <li style="margin: 8px 0;">Set up your own local AI environment</li>
                    <li style="margin: 8px 0;">Share this demo with colleagues and friends</li>
                    <li style="margin: 8px 0;">Join our community for updates and tips</li>
                </ul>
            </div>
            
            <p style="font-size: 1.2rem; margin-top: 30px;">
                <strong>Have a wonderful day!</strong> 🌟
            </p>
        </div>
    </div>
    """)

def reset_session():
    """Reset session data"""
    global session_data
    session_data = {
        "email": "",
        "start_time": None,
        "current_step": 0,
        "quiz_answers": [],
        "quiz_score": 0
    }

def create_interactive_tour():
    """Create the main interactive tour interface"""
    
    # Custom CSS for the tour
    custom_css = """
    /* Global font settings */
    * {
        font-family: 'Helvetica Neue', 'Helvetica', 'DejaVu Sans', 'Arial', sans-serif !important;
    }
    
    /* Main container styling */
    .gradio-container {
        max-width: 1200px !important;
        margin: 0 auto !important;
        background: #f8fafc !important;
    }
    
    /* Button styling */
    .btn-primary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        color: white !important;
        transition: all 0.3s ease !important;
    }
    
    .btn-primary:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
    }
    
    /* Progress indicator */
    .progress-bar {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        height: 6px;
        border-radius: 3px;
        transition: width 0.5s ease;
    }
    
    /* Module containers */
    .module-container {
        background: white;
        border-radius: 15px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    """
    
    with gr.Blocks(
        title="🤖 Interactive AI Demo Tour",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="purple",
            neutral_hue="slate",
            font=["Helvetica Neue", "Helvetica", "DejaVu Sans", "Arial", "sans-serif"]
        ),
        css=custom_css
    ) as demo:
        
        # State variables
        current_page = gr.State(0)  # 0=intro, 1=about, 2=email, 3=llm, 4=vision, 5=whisper, 6=tts, 7=quiz, 8=completion
        quiz_question_idx = gr.State(0)
        
        # Progress indicator
        progress_html = gr.HTML("""
        <div style="background: #e2e8f0; border-radius: 10px; padding: 10px; margin: 20px 0;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                <span style="font-weight: 600;">Progress</span>
                <span id="progress-text">Step 1 of 9</span>
            </div>
            <div style="background: #e2e8f0; border-radius: 3px; height: 6px;">
                <div id="progress-bar" class="progress-bar" style="width: 11%;"></div>
            </div>
        </div>
        """)
        
        # Page containers
        with gr.Group() as intro_page:
            intro_content = create_intro_page()
            intro_next_btn = gr.Button("🚀 Start Tour", variant="primary", size="lg", elem_classes=["btn-primary"])
        
        with gr.Group(visible=False) as about_page:
            about_content = create_about_page()
            about_next_btn = gr.Button("📧 Continue to Registration", variant="primary", size="lg", elem_classes=["btn-primary"])
        
        with gr.Group(visible=False) as email_page:
            email_content = create_email_page()
            email_input = gr.Textbox(
                placeholder="Enter your email address...",
                label="📧 Email Address",
                type="email"
            )
            email_submit_btn = gr.Button("🎯 Start Session", variant="primary", size="lg", elem_classes=["btn-primary"])
            email_status = gr.Textbox(visible=False)
        
        # Module 1: LLM Chat
        with gr.Group(visible=False) as llm_page:
            llm_content = create_llm_module()
            with gr.Row():
                with gr.Column(scale=4):
                    llm_chatbot = gr.Chatbot(
                        label="💬 Chat with AI",
                        height=300,
                        type="messages"
                    )
                    llm_input = gr.Textbox(
                        placeholder="Type your message here...",
                        label="Your message",
                        scale=4
                    )
                    llm_send_btn = gr.Button("📤 Send", variant="primary")
                with gr.Column(scale=1):
                    llm_next_btn = gr.Button("➡️ Next: Vision AI", variant="primary", size="lg", elem_classes=["btn-primary"])
        
        # Module 2: Vision AI
        with gr.Group(visible=False) as vision_page:
            vision_content = create_vision_module()
            with gr.Row():
                with gr.Column():
                    vision_image = gr.Image(
                        label="📸 Upload Image",
                        type="pil",
                        height=300
                    )
                    vision_question = gr.Textbox(
                        placeholder="Ask about the image...",
                        label="❓ Your question (optional)",
                        value="Describe this image in detail"
                    )
                    vision_analyze_btn = gr.Button("🔍 Analyze Image", variant="primary")
                with gr.Column():
                    vision_result = gr.Textbox(
                        label="🤖 AI Analysis",
                        lines=10,
                        placeholder="Upload an image and click 'Analyze Image' to see AI analysis..."
                    )
                    vision_next_btn = gr.Button("➡️ Next: Speech-to-Text", variant="primary", size="lg", elem_classes=["btn-primary"])
        
        # Module 3: Whisper (Speech-to-Text)
        with gr.Group(visible=False) as whisper_page:
            whisper_content = create_whisper_module()
            with gr.Row():
                with gr.Column():
                    whisper_audio = gr.Audio(
                        sources=["microphone"],
                        type="numpy",
                        label="🎙️ Record Your Voice"
                    )
                    whisper_transcribe_btn = gr.Button("🎤 Transcribe Audio", variant="primary")
                with gr.Column():
                    whisper_result = gr.Textbox(
                        label="📝 Transcription",
                        lines=5,
                        placeholder="Record audio and click 'Transcribe Audio' to see the text..."
                    )
                    whisper_next_btn = gr.Button("➡️ Next: Text-to-Speech", variant="primary", size="lg", elem_classes=["btn-primary"])
        
        # Module 4: TTS (Text-to-Speech)
        with gr.Group(visible=False) as tts_page:
            tts_content = create_tts_module()
            with gr.Row():
                with gr.Column():
                    tts_input = gr.Textbox(
                        placeholder="Enter text to be spoken...",
                        label="📝 Text to Speak",
                        lines=3,
                        value="Hello! This is a demonstration of text-to-speech technology running locally on your machine."
                    )
                    tts_speak_btn = gr.Button("🔊 Generate Speech", variant="primary")
                with gr.Column():
                    tts_status = gr.Textbox(
                        label="Status",
                        lines=2
                    )
                    tts_audio = gr.Audio(
                        label="🎵 Generated Speech",
                        autoplay=True
                    )
                    tts_next_btn = gr.Button("➡️ Next: Knowledge Quiz", variant="primary", size="lg", elem_classes=["btn-primary"])
        
        # Module 5: Quiz
        with gr.Group(visible=False) as quiz_page:
            quiz_content = create_quiz_page()
            quiz_start_btn = gr.Button("🧠 Start Quiz", variant="primary", size="lg", elem_classes=["btn-primary"])
        
        # Quiz questions (hidden initially)
        with gr.Group(visible=False) as quiz_questions_page:
            quiz_progress = gr.HTML()
            quiz_question_html = gr.HTML()
            quiz_options = gr.Radio(
                choices=[],
                label="Select your answer:",
                interactive=True
            )
            quiz_submit_btn = gr.Button("✅ Submit Answer", variant="primary")
            quiz_next_question_btn = gr.Button("➡️ Next Question", variant="primary", visible=False)
            quiz_finish_btn = gr.Button("🏁 Finish Quiz", variant="primary", visible=False, elem_classes=["btn-primary"])
        
        # Completion page
        with gr.Group(visible=False) as completion_page:
            completion_content = gr.HTML()
            restart_btn = gr.Button("🔄 Start New Session", variant="primary", size="lg", elem_classes=["btn-primary"])
        
        # Event handlers
        def show_about():
            return (
                gr.update(visible=False),  # Hide intro
                gr.update(visible=True),   # Show about
                1  # Update current page
            )
        
        def show_email():
            return (
                gr.update(visible=False),  # Hide about
                gr.update(visible=True),   # Show email
                2  # Update current page
            )
        
        def start_session(email):
            if not email or "@" not in email:
                return gr.update(value="Please enter a valid email address")
            
            handle_email_submission(email)
            return (
                gr.update(visible=False),  # Hide email page
                gr.update(visible=True),   # Show LLM page
                3  # Update current page
            )
        
        def show_vision():
            return (
                gr.update(visible=False),  # Hide LLM
                gr.update(visible=True),   # Show vision
                4  # Update current page
            )
        
        def show_whisper():
            return (
                gr.update(visible=False),  # Hide vision
                gr.update(visible=True),   # Show whisper
                5  # Update current page
            )
        
        def show_tts():
            return (
                gr.update(visible=False),  # Hide whisper
                gr.update(visible=True),   # Show TTS
                6  # Update current page
            )
        
        def show_quiz():
            return (
                gr.update(visible=False),  # Hide TTS
                gr.update(visible=True),   # Show quiz
                7  # Update current page
            )
        
        def start_quiz():
            return (
                gr.update(visible=False),  # Hide quiz intro
                gr.update(visible=True),   # Show quiz questions
                display_question(0)
            )
        
        def display_question(q_idx):
            if q_idx >= len(QUIZ_QUESTIONS):
                return show_completion()
            
            question = QUIZ_QUESTIONS[q_idx]
            progress_html_content = f"""
            <div style="background: #e2e8f0; border-radius: 10px; padding: 15px; margin: 20px 0;">
                <h3>Question {q_idx + 1} of {len(QUIZ_QUESTIONS)}</h3>
                <div style="background: #e2e8f0; border-radius: 3px; height: 6px; margin: 10px 0;">
                    <div class="progress-bar" style="width: {((q_idx + 1) / len(QUIZ_QUESTIONS)) * 100}%;"></div>
                </div>
            </div>
            """
            
            question_html_content = f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 25px; border-radius: 15px; margin: 20px 0;">
                <h3 style="margin-bottom: 15px;">📚 {question['module']} Question</h3>
                <p style="font-size: 1.2rem; font-weight: 500;">{question['question']}</p>
            </div>
            """
            
            return (
                gr.update(value=progress_html_content),
                gr.update(value=question_html_content),
                gr.update(choices=question['options'], value=None),
                gr.update(visible=True),   # Show submit button
                gr.update(visible=False),  # Hide next button
                gr.update(visible=False),  # Hide finish button
                q_idx
            )
        
        def submit_answer(q_idx, selected_option_text):
            if selected_option_text is None:
                return gr.update(value="Please select an answer first.")
            
            # Find the index of the selected option
            question = QUIZ_QUESTIONS[q_idx]
            selected_idx = question['options'].index(selected_option_text)
            
            # Submit the answer
            submit_quiz_answer(q_idx, selected_idx)
            
            # Check if this is the last question
            if q_idx >= len(QUIZ_QUESTIONS) - 1:
                return (
                    gr.update(visible=False),  # Hide submit button
                    gr.update(visible=False),  # Hide next button
                    gr.update(visible=True),   # Show finish button
                )
            else:
                return (
                    gr.update(visible=False),  # Hide submit button
                    gr.update(visible=True),   # Show next button
                    gr.update(visible=False),  # Hide finish button
                )
        
        def next_question(q_idx):
            return display_question(q_idx + 1)
        
        def show_completion():
            completion_html = create_completion_page()
            return (
                gr.update(visible=False),  # Hide quiz questions
                gr.update(visible=True),   # Show completion
                gr.update(value=completion_html),
                8  # Update current page
            )
        
        def restart_session():
            reset_session()
            return (
                gr.update(visible=True),   # Show intro
                gr.update(visible=False),  # Hide completion
                gr.update(visible=False),  # Hide all other pages
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                0  # Reset current page
            )
        
        # Wire up the event handlers
        intro_next_btn.click(
            show_about,
            outputs=[intro_page, about_page, current_page]
        )
        
        about_next_btn.click(
            show_email,
            outputs=[about_page, email_page, current_page]
        )
        
        email_submit_btn.click(
            start_session,
            inputs=[email_input],
            outputs=[email_page, llm_page, current_page]
        )
        
        # LLM Chat handlers
        llm_send_btn.click(
            llm_chat,
            inputs=[llm_input, llm_chatbot],
            outputs=[llm_chatbot, llm_input]
        )
        
        llm_input.submit(
            llm_chat,
            inputs=[llm_input, llm_chatbot],
            outputs=[llm_chatbot, llm_input]
        )
        
        llm_next_btn.click(
            show_vision,
            outputs=[llm_page, vision_page, current_page]
        )
        
        # Vision AI handlers
        vision_analyze_btn.click(
            analyze_image,
            inputs=[vision_image, vision_question],
            outputs=[vision_result]
        )
        
        vision_next_btn.click(
            show_whisper,
            outputs=[vision_page, whisper_page, current_page]
        )
        
        # Whisper handlers
        whisper_transcribe_btn.click(
            transcribe_audio,
            inputs=[whisper_audio],
            outputs=[whisper_result]
        )
        
        whisper_next_btn.click(
            show_tts,
            outputs=[whisper_page, tts_page, current_page]
        )
        
        # TTS handlers
        tts_speak_btn.click(
            text_to_speech,
            inputs=[tts_input],
            outputs=[tts_status, tts_audio]
        )
        
        tts_next_btn.click(
            show_quiz,
            outputs=[tts_page, quiz_page, current_page]
        )
        
        # Quiz handlers
        quiz_start_btn.click(
            start_quiz,
            outputs=[quiz_page, quiz_questions_page, quiz_progress, quiz_question_html, quiz_options, quiz_submit_btn, quiz_next_question_btn, quiz_finish_btn, quiz_question_idx]
        )
        
        quiz_submit_btn.click(
            submit_answer,
            inputs=[quiz_question_idx, quiz_options],
            outputs=[quiz_submit_btn, quiz_next_question_btn, quiz_finish_btn]
        )
        
        quiz_next_question_btn.click(
            lambda q_idx: next_question(q_idx),
            inputs=[quiz_question_idx],
            outputs=[quiz_progress, quiz_question_html, quiz_options, quiz_submit_btn, quiz_next_question_btn, quiz_finish_btn, quiz_question_idx]
        )
        
        quiz_finish_btn.click(
            show_completion,
            outputs=[quiz_questions_page, completion_page, completion_content, current_page]
        )
        
        # Restart handler
        restart_btn.click(
            restart_session,
            outputs=[intro_page, completion_page, about_page, email_page, llm_page, vision_page, whisper_page, tts_page, quiz_page, quiz_questions_page, current_page]
        )
    
    return demo

def main():
    """Main application entry point"""
    print("🤖 Starting Interactive AI Demo Tour...")
    
    # Initialize models in background
    init_thread = threading.Thread(target=initialize_models)
    init_thread.start()
    
    # Create and launch interface
    demo = create_interactive_tour()
    
    # Wait for models to load
    init_thread.join()
    
    # Find free port
    free_port = Config.find_free_port(Config.GRADIO_PORT)
    if free_port != Config.GRADIO_PORT:
        print(f"⚠️ Port {Config.GRADIO_PORT} is busy, using port {free_port}")
    
    # Get local IP for network access
    local_ip = Config.get_local_ip()
    
    print(f"🌐 Launching Interactive AI Demo Tour...")
    print(f"   Local:   http://localhost:{free_port}")
    print(f"   Network: http://{local_ip}:{free_port}")
    print(f"")
    print(f"🎯 Interactive Tour Features:")
    print(f"   ✅ Step-by-step guided experience")
    print(f"   ✅ Email registration and certificates")
    print(f"   ✅ Hands-on AI module testing")
    print(f"   ✅ Knowledge assessment quiz")
    print(f"   ✅ Completion tracking and scoring")
    
    demo.launch(
        server_name=Config.GRADIO_HOST,
        server_port=free_port,
        share=Config.GRADIO_SHARE,
        inbrowser=True
    )

if __name__ == "__main__":
    main()