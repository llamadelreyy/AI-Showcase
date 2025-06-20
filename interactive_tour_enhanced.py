"""
Interactive AI Demo Tour - Enhanced Version with Complete Step-by-Step Structure
Includes all 9 steps: Intro → About → Name → LLM → Vision → Whisper → TTS → Quiz → Complete
Enhanced with: Camera capture, Microphone recording, QR code certificates
"""

import gradio as gr
import numpy as np
import tempfile
import os
import json
import time
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import threading
import qrcode
import base64
import io

from models import WhisperSTT, LocalTTS, LocalLLM, LocalVLM, OllamaLLM
from config import Config

# Global model instances
whisper_model = None
tts_model = None
llm_model = None
vlm_model = None

# Session data
session_data = {
    "name": "",
    "start_time": None,
    "current_step": 0,
    "quiz_answers": [],
    "quiz_score": 0,
    "current_question": 0
}

# Enhanced Quiz questions for the final assessment (10 questions)
QUIZ_QUESTIONS = [
    {
        "question": "What does LLM stand for in AI?",
        "options": ["Large Language Model", "Linear Learning Machine", "Local Logic Module", "Language Learning Method"],
        "correct": 0,
        "module": "LLM Chat",
        "explanation": "LLM stands for Large Language Model - AI systems trained on vast amounts of text data."
    },
    {
        "question": "Which AI model is commonly used for speech-to-text conversion?",
        "options": ["GPT-4", "BERT", "Whisper", "DALL-E"],
        "correct": 2,
        "module": "Speech-to-Text",
        "explanation": "Whisper is OpenAI's automatic speech recognition system for converting speech to text."
    },
    {
        "question": "What is the main advantage of running AI models locally?",
        "options": ["Faster internet", "Better graphics", "Privacy and no API costs", "More storage"],
        "correct": 2,
        "module": "General",
        "explanation": "Local AI ensures complete privacy and eliminates ongoing API costs."
    },
    {
        "question": "What does TTS stand for?",
        "options": ["Text-to-Speech", "Time-to-Start", "Type-to-Send", "Talk-to-System"],
        "correct": 0,
        "module": "Text-to-Speech",
        "explanation": "TTS stands for Text-to-Speech - technology that converts written text into spoken words."
    },
    {
        "question": "Which type of AI model can analyze and describe images?",
        "options": ["Language Model only", "Vision Language Model (VLM)", "Audio Model only", "Text Model only"],
        "correct": 1,
        "module": "Vision AI",
        "explanation": "Vision Language Models combine computer vision with language understanding to analyze images."
    },
    {
        "question": "What is the benefit of using Ollama for AI models?",
        "options": ["Cloud storage", "Local model management and optimization", "Internet speed", "Graphics enhancement"],
        "correct": 1,
        "module": "General",
        "explanation": "Ollama provides easy local model management, allowing you to run AI models efficiently on your machine."
    },
    {
        "question": "Which neural TTS engine provides the highest quality voices?",
        "options": ["pyttsx3", "Microsoft Edge TTS", "Basic system TTS", "Command line TTS"],
        "correct": 1,
        "module": "Text-to-Speech",
        "explanation": "Microsoft Edge TTS uses neural networks to generate very natural-sounding voices."
    },
    {
        "question": "What can Vision AI analyze in uploaded images?",
        "options": ["Only text (OCR)", "Only colors and shapes", "Objects, people, text, scenes, and details", "Only facial expressions"],
        "correct": 2,
        "module": "Vision AI",
        "explanation": "Modern Vision AI can comprehensively analyze images including objects, people, text, scenes, and contextual details."
    },
    {
        "question": "How many languages can Whisper speech recognition handle?",
        "options": ["Only English", "About 10 languages", "99+ languages", "Only European languages"],
        "correct": 2,
        "module": "Speech-to-Text",
        "explanation": "Whisper supports 99+ languages and can handle various accents and background noise."
    },
    {
        "question": "What makes this AI Demo Stack special compared to cloud AI services?",
        "options": ["It's slower but cheaper", "100% local, private, and free after setup", "It requires internet", "It only works on servers"],
        "correct": 1,
        "module": "General",
        "explanation": "This demo runs entirely locally, ensuring privacy, eliminating ongoing costs, and working without internet."
    }
]

def generate_certificate_qr(name, score, total):
    """Generate a QR code certificate for the participant"""
    try:
        # Create certificate data
        certificate_data = {
            "participant": name,
            "course": "AI Demo Tour",
            "score": f"{score}/{total}",
            "percentage": f"{(score/total)*100:.0f}%",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "issuer": "Local AI Demo Stack",
            "certificate_id": f"AIDT-{datetime.now().strftime('%Y%m%d')}-{hash(name) % 10000:04d}"
        }
        
        # Create certificate text for QR code
        certificate_text = f"""
🎓 CERTIFICATE OF COMPLETION 🎓

Participant: {name}
Course: AI Demo Tour
Score: {score}/{total} ({(score/total)*100:.0f}%)
Date: {datetime.now().strftime("%B %d, %Y")}
Certificate ID: {certificate_data['certificate_id']}

This certifies that {name} has successfully completed the Interactive AI Demo Tour, demonstrating understanding of Local AI technologies including LLM Chat, Vision AI, Speech-to-Text, and Text-to-Speech systems.

Issued by: Local AI Demo Stack
Verification: This certificate can be verified using the QR code data.
        """.strip()
        
        # Generate QR code
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(certificate_text)
        qr.make(fit=True)
        
        # Create QR code image
        qr_img = qr.make_image(fill_color="black", back_color="white")
        
        # Create a more elaborate certificate image
        cert_width, cert_height = 800, 600
        cert_img = Image.new('RGB', (cert_width, cert_height), 'white')
        draw = ImageDraw.Draw(cert_img)
        
        # Try to use a nice font, fallback to default
        try:
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
            header_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            text_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
            small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except:
            title_font = ImageFont.load_default()
            header_font = ImageFont.load_default()
            text_font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        # Draw certificate border
        border_color = "#667eea"
        draw.rectangle([10, 10, cert_width-10, cert_height-10], outline=border_color, width=3)
        draw.rectangle([20, 20, cert_width-20, cert_height-20], outline=border_color, width=1)
        
        # Draw title
        title_text = "🎓 CERTIFICATE OF COMPLETION 🎓"
        title_bbox = draw.textbbox((0, 0), title_text, font=title_font)
        title_width = title_bbox[2] - title_bbox[0]
        draw.text(((cert_width - title_width) // 2, 40), title_text, fill=border_color, font=title_font)
        
        # Draw participant name
        name_text = f"This certifies that {name}"
        name_bbox = draw.textbbox((0, 0), name_text, font=header_font)
        name_width = name_bbox[2] - name_bbox[0]
        draw.text(((cert_width - name_width) // 2, 120), name_text, fill="black", font=header_font)
        
        # Draw course info
        course_text = "has successfully completed the"
        course_bbox = draw.textbbox((0, 0), course_text, font=text_font)
        course_width = course_bbox[2] - course_bbox[0]
        draw.text(((cert_width - course_width) // 2, 160), course_text, fill="black", font=text_font)
        
        course_name = "Interactive AI Demo Tour"
        course_name_bbox = draw.textbbox((0, 0), course_name, font=header_font)
        course_name_width = course_name_bbox[2] - course_name_bbox[0]
        draw.text(((cert_width - course_name_width) // 2, 190), course_name, fill=border_color, font=header_font)
        
        # Draw score
        score_text = f"Score: {score}/{total} ({(score/total)*100:.0f}%)"
        score_bbox = draw.textbbox((0, 0), score_text, font=text_font)
        score_width = score_bbox[2] - score_bbox[0]
        draw.text(((cert_width - score_width) // 2, 240), score_text, fill="black", font=text_font)
        
        # Draw date
        date_text = f"Date: {datetime.now().strftime('%B %d, %Y')}"
        date_bbox = draw.textbbox((0, 0), date_text, font=text_font)
        date_width = date_bbox[2] - date_bbox[0]
        draw.text(((cert_width - date_width) // 2, 280), date_text, fill="black", font=text_font)
        
        # Draw certificate ID
        id_text = f"Certificate ID: {certificate_data['certificate_id']}"
        id_bbox = draw.textbbox((0, 0), id_text, font=small_font)
        id_width = id_bbox[2] - id_bbox[0]
        draw.text(((cert_width - id_width) // 2, 320), id_text, fill="gray", font=small_font)
        
        # Add QR code to certificate
        qr_size = 120
        qr_resized = qr_img.resize((qr_size, qr_size))
        qr_x = cert_width - qr_size - 40
        qr_y = cert_height - qr_size - 40
        cert_img.paste(qr_resized, (qr_x, qr_y))
        
        # Add QR code label
        qr_label = "Scan for verification"
        qr_label_bbox = draw.textbbox((0, 0), qr_label, font=small_font)
        qr_label_width = qr_label_bbox[2] - qr_label_bbox[0]
        draw.text((qr_x + (qr_size - qr_label_width) // 2, qr_y + qr_size + 5), qr_label, fill="gray", font=small_font)
        
        # Add issuer
        issuer_text = "Issued by: Local AI Demo Stack"
        draw.text((40, cert_height - 60), issuer_text, fill="gray", font=small_font)
        
        # Save certificate to temporary file
        cert_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        cert_img.save(cert_temp.name, 'PNG')
        cert_temp.close()
        
        return cert_temp.name, certificate_data
        
    except Exception as e:
        print(f"Error generating certificate: {e}")
        return None, None

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

def get_progress_html(step, total_steps=10):
    """Generate progress bar HTML"""
    # Ensure completion page shows 100%
    if step >= 9:  # Completion page
        percentage = 100
    else:
        percentage = ((step + 1) / total_steps) * 100
    
    step_names = ["Intro", "About", "Name", "LLM Chat", "Vision AI", "Speech-to-Text", "Text-to-Speech", "Quiz Intro", "Quiz", "Complete"]
    current_step_name = step_names[min(step, len(step_names)-1)]
    
    return f"""
    <div style="background: #e2e8f0; border-radius: 10px; padding: 15px; margin: 20px 0;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-weight: 600; color: #333;">Progress: {current_step_name}</span>
            <span style="color: #666;">Step {step + 1} of {total_steps}</span>
        </div>
        <div style="background: #cbd5e1; border-radius: 3px; height: 8px;">
            <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); height: 8px; border-radius: 3px; width: {percentage}%; transition: width 0.5s ease;"></div>
        </div>
    </div>
    """

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

def analyze_image(image, question):
    """Analyze uploaded or captured image"""
    if image is None:
        return "Please upload an image or capture one with the camera first."
    
    if vlm_model is None:
        return "Vision model not available. Please check model loading."
    
    try:
        if not question.strip():
            question = "Describe this image in detail, including objects, people, colors, and any text you can see."
        
        analysis = vlm_model.analyze_image(image, question)
        return f"🤖 **AI Vision Analysis:**\n\n{analysis}"
        
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

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

def submit_quiz_answer(selected_option, current_q):
    """Submit a quiz answer and move to next question"""
    global session_data
    
    if selected_option is None:
        return "Please select an answer first.", current_q, gr.update(), gr.update(), gr.update()
    
    # Find the index of the selected option
    question = QUIZ_QUESTIONS[current_q]
    selected_idx = question['options'].index(selected_option)
    
    # Check if correct
    is_correct = selected_idx == question['correct']
    if is_correct:
        session_data["quiz_score"] += 1
    
    # Store answer
    session_data["quiz_answers"].append({
        "question": current_q,
        "selected": selected_idx,
        "correct": is_correct
    })
    
    # Move to next question
    next_q = current_q + 1
    
    if next_q >= len(QUIZ_QUESTIONS):
        # Quiz completed - Generate certificate and detailed results
        score = session_data["quiz_score"]
        total = len(QUIZ_QUESTIONS)
        name = session_data.get("name", "Participant")
        
        # Generate QR code certificate
        cert_path, cert_data = generate_certificate_qr(name, score, total)
        
        # Navigate to completion step (step 9)
        session_data["current_step"] = 9
        session_data["certificate_path"] = cert_path
        session_data["certificate_data"] = cert_data
        
        return (
            next_q,
            gr.update(value=""),  # Clear quiz question
            gr.update(choices=[], value=None),  # Clear quiz options
            gr.update(),  # Keep feedback hidden
            gr.update(value=cert_path, visible=True) if cert_path else gr.update()  # Show certificate
        )
    else:
        # Show next question
        next_question = QUIZ_QUESTIONS[next_q]
        question_html = f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 25px; border-radius: 15px; margin: 20px 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 15px;">
                <h3>📚 {next_question['module']} Question</h3>
                <span>Question {next_q + 1} of {len(QUIZ_QUESTIONS)}</span>
            </div>
            <p style="font-size: 1.2rem; font-weight: 500; margin-bottom: 20px;">{next_question['question']}</p>
        </div>
        """
        
        return (
            next_q,
            gr.update(value=question_html),
            gr.update(choices=next_question['options'], value=None),
            gr.update(),  # Keep feedback hidden
            gr.update()  # Keep certificate hidden
        )

def reset_session():
    """Reset session data"""
    global session_data
    session_data = {
        "name": "",
        "start_time": None,
        "current_step": 0,
        "quiz_answers": [],
        "quiz_score": 0,
        "current_question": 0,
        "certificate_path": None,
        "certificate_data": None
    }

def create_interactive_tour():
    """Create the main interactive tour interface with complete step-by-step structure"""
    
    # Custom CSS
    custom_css = """
    * {
        font-family: 'Helvetica Neue', 'Helvetica', 'DejaVu Sans', 'Arial', sans-serif !important;
    }
    
    .gradio-container {
        max-width: 1200px !important;
        margin: 0 auto !important;
        background: #f8fafc !important;
    }
    
    .btn-primary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        color: white !important;
        transition: all 0.3s ease !important;
    }
    
    .btn-secondary {
        background: rgba(255, 255, 255, 0.9) !important;
        border: 2px solid #667eea !important;
        color: #667eea !important;
        border-radius: 10px !important;
        padding: 10px 20px !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }
    
    .btn-primary:hover, .btn-secondary:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
    }
    """
    
    with gr.Blocks(
        title="🤖 Interactive AI Demo Tour - Enhanced",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="purple",
            neutral_hue="slate",
            font=["Helvetica Neue", "Helvetica", "DejaVu Sans", "Arial", "sans-serif"]
        ),
        css=custom_css
    ) as demo:
        
        # State variables
        current_step = gr.State(0)
        quiz_question_idx = gr.State(0)
        
        # Progress indicator
        progress_display = gr.HTML(get_progress_html(0))
        
        # Navigation buttons (always visible)
        with gr.Row():
            back_btn = gr.Button("⬅️ Back", variant="secondary", visible=False, elem_classes=["btn-secondary"])
            next_btn = gr.Button("➡️ Next", variant="primary", elem_classes=["btn-primary"])
        
        # Direct module navigation buttons
        with gr.Row():
            gr.HTML("<h3 style='text-align: center; margin: 20px 0; color: #667eea;'>🚀 Quick Access - Jump to Any Module:</h3>")
        with gr.Row():
            home_direct_btn = gr.Button("🏠 Home", variant="secondary", elem_classes=["btn-secondary"])
            llm_direct_btn = gr.Button("💬 LLM Chat", variant="secondary", elem_classes=["btn-secondary"])
            vision_direct_btn = gr.Button("👁️ Vision AI", variant="secondary", elem_classes=["btn-secondary"])
            whisper_direct_btn = gr.Button("🎤 Speech-to-Text", variant="secondary", elem_classes=["btn-secondary"])
            tts_direct_btn = gr.Button("🔊 Text-to-Speech", variant="secondary", elem_classes=["btn-secondary"])
        
        # Main content area
        main_content = gr.HTML("""
        <div style="text-align: center; padding: 50px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 70vh; color: white; border-radius: 20px;">
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
        
        # Interactive components (hidden initially)
        with gr.Group(visible=False) as name_input_group:
            name_input = gr.Textbox(
                placeholder="Enter your full name...",
                label="👤 Your Name",
                type="text"
            )
        
        with gr.Group(visible=False) as llm_group:
            llm_chatbot = gr.Chatbot(
                label="💬 Chat with AI",
                height=300,
                type="messages"
            )
            llm_input = gr.Textbox(
                placeholder="Type your message here...",
                label="Your message"
            )
            llm_send_btn = gr.Button("📤 Send", variant="primary")
        
        with gr.Group(visible=False) as vision_group:
            with gr.Row():
                with gr.Column():
                    vision_image = gr.Image(
                        sources=["webcam", "upload"],  # Enable both camera and upload
                        label="📸 Capture or Upload Image",
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
                        lines=15,
                        placeholder="Capture/upload an image and click 'Analyze Image' to see AI analysis..."
                    )
        
        with gr.Group(visible=False) as whisper_group:
            with gr.Row():
                whisper_audio = gr.Audio(
                    sources=["microphone"],
                    type="numpy",
                    label="🎙️ Record Your Voice"
                )
                whisper_result = gr.Textbox(
                    label="📝 Transcription",
                    lines=5,
                    placeholder="Record audio to see transcription..."
                )
            whisper_transcribe_btn = gr.Button("🎤 Transcribe Audio", variant="primary")
        
        with gr.Group(visible=False) as tts_group:
            with gr.Row():
                with gr.Column():
                    tts_input = gr.Textbox(
                        placeholder="Enter text to be spoken...",
                        label="📝 Text to Speak",
                        lines=3,
                        value="Hello! This is a demonstration of text-to-speech technology."
                    )
                    tts_speak_btn = gr.Button("🔊 Generate Speech", variant="primary")
                with gr.Column():
                    tts_status = gr.Textbox(label="Status", lines=2)
                    tts_audio = gr.Audio(label="🎵 Generated Speech", autoplay=True)
        
        with gr.Group(visible=False) as quiz_group:
            quiz_question_display = gr.HTML()
            quiz_options = gr.Radio(
                choices=[],
                label="Select your answer:",
                interactive=True
            )
            quiz_submit_btn = gr.Button("✅ Submit Answer", variant="primary")
            quiz_feedback = gr.HTML(visible=False)  # Hidden feedback for results
        
        # Certificate display
        with gr.Group(visible=False) as certificate_group:
            certificate_image = gr.Image(
                label="🎓 Your Certificate",
                type="filepath",
                height=400,
                show_download_button=True
            )
        
        # Completion display
        completion_display = gr.HTML(visible=False)
        
        # Define page content
        page_contents = [
            # Page 0: Intro
            """
            <div style="text-align: center; padding: 50px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 70vh; color: white; border-radius: 20px;">
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
            """,
            # Page 1: About
            """
            <div style="padding: 40px; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); min-height: 70vh; color: white; border-radius: 20px;">
                <div style="background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); border-radius: 20px; padding: 40px; max-width: 800px; margin: 0 auto;">
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
            """,
            # Page 2: Name Collection
            """
            <div style="text-align: center; padding: 50px; background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); min-height: 70vh; color: white; border-radius: 20px;">
                <div style="background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); border-radius: 20px; padding: 40px; max-width: 800px; margin: 0 auto;">
                    <h2 style="font-size: 2.5rem; margin-bottom: 20px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                        👤 Almost Ready!
                    </h2>
                    <p style="font-size: 1.2rem; margin-bottom: 30px; line-height: 1.6;">
                        To personalize your experience and generate your completion certificate,
                        please provide your full name.
                    </p>
                    
                    <div style="background: rgba(255, 255, 255, 0.2); padding: 25px; border-radius: 15px; margin: 20px 0;">
                        <h3 style="margin-bottom: 15px;">🏆 What You'll Receive:</h3>
                        <ul style="text-align: left; max-width: 400px; margin: 0 auto;">
                            <li style="margin: 10px 0;">🎓 Digital certificate with QR code</li>
                            <li style="margin: 10px 0;">📊 Your personalized quiz results</li>
                            <li style="margin: 10px 0;">🔍 Scannable verification code</li>
                            <li style="margin: 10px 0;">📱 Mobile-friendly certificate</li>
                        </ul>
                    </div>
                    
                    <div style="background: rgba(255, 255, 255, 0.2); padding: 20px; border-radius: 15px; margin: 20px 0;">
                        <h3 style="margin-bottom: 10px;">📱 QR Code Certificate</h3>
                        <p style="font-size: 1rem; line-height: 1.5;">
                            Your certificate will include a QR code that contains all your completion details.
                            Anyone can scan it to verify your achievement!
                        </p>
                    </div>
                    
                    <p style="font-size: 0.9rem; margin-top: 20px; opacity: 0.8;">
                        🔒 Your name is only used for certificate generation.
                        No personal data is stored or transmitted.
                    </p>
                </div>
            </div>
            """,
            # Page 3: LLM Chat Content
            """
            <div style="padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 70vh; color: white; border-radius: 20px;">
                <div style="background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); border-radius: 20px; padding: 40px; max-width: 800px; margin: 0 auto;">
                    <h1 style="font-size: 2.5rem; text-align: center; margin-bottom: 30px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                        💬 Large Language Model (LLM) Chat
                    </h1>
                    
                    <div style="background: rgba(255, 255, 255, 0.2); border-radius: 15px; padding: 25px; margin: 30px 0;">
                        <h3 style="margin-bottom: 15px;">🧠 What is an LLM?</h3>
                        <p style="font-size: 1.1rem; line-height: 1.6;">
                            A Large Language Model is an AI system trained on vast amounts of text data to understand and generate human-like responses.
                            It can engage in conversations, answer questions, help with writing, coding, analysis, and much more.
                        </p>
                    </div>
                    
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 30px 0;">
                        <div style="background: rgba(255, 255, 255, 0.1); padding: 20px; border-radius: 15px;">
                            <h4 style="margin-bottom: 10px;">🎯 Capabilities</h4>
                            <ul style="text-align: left; margin: 0; padding-left: 20px;">
                                <li>Natural conversations</li>
                                <li>Question answering</li>
                                <li>Text generation</li>
                                <li>Code assistance</li>
                                <li>Creative writing</li>
                                <li>Problem solving</li>
                            </ul>
                        </div>
                        
                        <div style="background: rgba(255, 255, 255, 0.1); padding: 20px; border-radius: 15px;">
                            <h4 style="margin-bottom: 10px;">⚡ Local Benefits</h4>
                            <ul style="text-align: left; margin: 0; padding-left: 20px;">
                                <li>Complete privacy</li>
                                <li>No internet required</li>
                                <li>Instant responses</li>
                                <li>No usage limits</li>
                                <li>Zero API costs</li>
                                <li>Full control</li>
                            </ul>
                        </div>
                    </div>
                    
                    <div style="background: rgba(255, 255, 255, 0.2); border-radius: 15px; padding: 25px; margin: 30px 0;">
                        <h3 style="margin-bottom: 15px;">💡 Try These Examples:</h3>
                        <div style="text-align: left;">
                            <p style="margin: 10px 0; padding: 10px; background: rgba(255, 255, 255, 0.1); border-radius: 8px;">
                                "Explain quantum computing in simple terms"
                            </p>
                            <p style="margin: 10px 0; padding: 10px; background: rgba(255, 255, 255, 0.1); border-radius: 8px;">
                                "Write a short poem about artificial intelligence"
                            </p>
                            <p style="margin: 10px 0; padding: 10px; background: rgba(255, 255, 255, 0.1); border-radius: 8px;">
                                "Help me plan a healthy weekly meal schedule"
                            </p>
                            <p style="margin: 10px 0; padding: 10px; background: rgba(255, 255, 255, 0.1); border-radius: 8px;">
                                "What are the benefits of local AI processing?"
                            </p>
                        </div>
                    </div>
                </div>
            </div>
            """,
            # Page 4: Vision AI Content
            """
            <div style="padding: 40px; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); min-height: 70vh; color: white; border-radius: 20px;">
                <div style="background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); border-radius: 20px; padding: 40px; max-width: 800px; margin: 0 auto;">
                    <h1 style="font-size: 2.5rem; text-align: center; margin-bottom: 30px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                        👁️ Vision AI - Image Analysis
                    </h1>
                    
                    <div style="background: rgba(255, 255, 255, 0.2); border-radius: 15px; padding: 25px; margin: 30px 0;">
                        <h3 style="margin-bottom: 15px;">🔍 What is Vision AI?</h3>
                        <p style="font-size: 1.1rem; line-height: 1.6;">
                            Vision AI combines computer vision with language models to understand and describe images.
                            It can identify objects, read text, analyze scenes, and answer specific questions about visual content.
                        </p>
                    </div>
                    
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 30px 0;">
                        <div style="background: rgba(255, 255, 255, 0.1); padding: 20px; border-radius: 15px;">
                            <h4 style="margin-bottom: 10px;">🎯 What It Can See</h4>
                            <ul style="text-align: left; margin: 0; padding-left: 20px;">
                                <li>Objects & people</li>
                                <li>Text in images (OCR)</li>
                                <li>Colors & composition</li>
                                <li>Facial expressions</li>
                                <li>Scene context</li>
                                <li>Artistic elements</li>
                            </ul>
                        </div>
                        
                        <div style="background: rgba(255, 255, 255, 0.1); padding: 20px; border-radius: 15px;">
                            <h4 style="margin-bottom: 10px;">🚀 Use Cases</h4>
                            <ul style="text-align: left; margin: 0; padding-left: 20px;">
                                <li>Document analysis</li>
                                <li>Photo organization</li>
                                <li>Accessibility tools</li>
                                <li>Content moderation</li>
                                <li>Quality inspection</li>
                                <li>Medical imaging</li>
                            </ul>
                        </div>
                    </div>
                    
                    <div style="background: rgba(255, 255, 255, 0.2); border-radius: 15px; padding: 25px; margin: 30px 0;">
                        <h3 style="margin-bottom: 15px;">📸 Try These Examples:</h3>
                        <div style="text-align: left;">
                            <p style="margin: 10px 0; padding: 10px; background: rgba(255, 255, 255, 0.1); border-radius: 8px;">
                                📷 <strong>Use Camera:</strong> Click the camera icon to take a live photo
                            </p>
                            <p style="margin: 10px 0; padding: 10px; background: rgba(255, 255, 255, 0.1); border-radius: 8px;">
                                📁 <strong>Upload Photo:</strong> Or upload an existing image
                            </p>
                            <p style="margin: 10px 0; padding: 10px; background: rgba(255, 255, 255, 0.1); border-radius: 8px;">
                                💬 <strong>Ask Questions:</strong> "What objects do you see?" or "Read any text"
                            </p>
                            <p style="margin: 10px 0; padding: 10px; background: rgba(255, 255, 255, 0.1); border-radius: 8px;">
                                🎨 <strong>Analyze Details:</strong> "Describe colors, mood, and atmosphere"
                            </p>
                        </div>
                    </div>
                </div>
            </div>
            """,
            # Page 5: Speech-to-Text Content
            """
            <div style="padding: 40px; background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); min-height: 70vh; color: white; border-radius: 20px;">
                <div style="background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); border-radius: 20px; padding: 40px; max-width: 800px; margin: 0 auto;">
                    <h1 style="font-size: 2.5rem; text-align: center; margin-bottom: 30px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                        🎤 Whisper - Speech to Text
                    </h1>
                    
                    <div style="background: rgba(255, 255, 255, 0.2); border-radius: 15px; padding: 25px; margin: 30px 0;">
                        <h3 style="margin-bottom: 15px;">🎵 What is Whisper?</h3>
                        <p style="font-size: 1.1rem; line-height: 1.6;">
                            Whisper is OpenAI's automatic speech recognition (ASR) system that converts spoken language into text.
                            It's trained on diverse audio data and can handle multiple languages, accents, and background noise.
                        </p>
                    </div>
                    
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 30px 0;">
                        <div style="background: rgba(255, 255, 255, 0.1); padding: 20px; border-radius: 15px;">
                            <h4 style="margin-bottom: 10px;">🎯 Capabilities</h4>
                            <ul style="text-align: left; margin: 0; padding-left: 20px;">
                                <li>99+ languages</li>
                                <li>Noise robustness</li>
                                <li>Accent handling</li>
                                <li>Real-time processing</li>
                                <li>High accuracy</li>
                                <li>Punctuation</li>
                            </ul>
                        </div>
                        
                        <div style="background: rgba(255, 255, 255, 0.1); padding: 20px; border-radius: 15px;">
                            <h4 style="margin-bottom: 10px;">🚀 Applications</h4>
                            <ul style="text-align: left; margin: 0; padding-left: 20px;">
                                <li>Voice assistants</li>
                                <li>Meeting transcription</li>
                                <li>Accessibility tools</li>
                                <li>Content creation</li>
                                <li>Language learning</li>
                                <li>Voice commands</li>
                            </ul>
                        </div>
                    </div>
                    
                    <div style="background: rgba(255, 255, 255, 0.2); border-radius: 15px; padding: 25px; margin: 30px 0;">
                        <h3 style="margin-bottom: 15px;">🎙️ Try Recording:</h3>
                        <div style="text-align: left;">
                            <p style="margin: 10px 0; padding: 10px; background: rgba(255, 255, 255, 0.1); border-radius: 8px;">
                                "Hello, this is a test of speech recognition technology"
                            </p>
                            <p style="margin: 10px 0; padding: 10px; background: rgba(255, 255, 255, 0.1); border-radius: 8px;">
                                Try speaking in different languages or accents
                            </p>
                            <p style="margin: 10px 0; padding: 10px; background: rgba(255, 255, 255, 0.1); border-radius: 8px;">
                                Record a longer passage or story
                            </p>
                            <p style="margin: 10px 0; padding: 10px; background: rgba(255, 255, 255, 0.1); border-radius: 8px;">
                                Test with background noise or music
                            </p>
                        </div>
                    </div>
                </div>
            </div>
            """,
            # Page 6: Text-to-Speech Content
            """
            <div style="padding: 40px; background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); min-height: 70vh; color: white; border-radius: 20px;">
                <div style="background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); border-radius: 20px; padding: 40px; max-width: 800px; margin: 0 auto;">
                    <h1 style="font-size: 2.5rem; text-align: center; margin-bottom: 30px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                        🔊 Text-to-Speech (TTS)
                    </h1>
                    
                    <div style="background: rgba(255, 255, 255, 0.2); border-radius: 15px; padding: 25px; margin: 30px 0;">
                        <h3 style="margin-bottom: 15px;">🗣️ What is TTS?</h3>
                        <p style="font-size: 1.1rem; line-height: 1.6;">
                            Text-to-Speech technology converts written text into natural-sounding spoken words.
                            Modern TTS systems use neural networks to create human-like voices with proper intonation and emotion.
                        </p>
                    </div>
                    
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 30px 0;">
                        <div style="background: rgba(255, 255, 255, 0.1); padding: 20px; border-radius: 15px;">
                            <h4 style="margin-bottom: 10px;">🎯 Features</h4>
                            <ul style="text-align: left; margin: 0; padding-left: 20px;">
                                <li>Natural voices</li>
                                <li>Multiple languages</li>
                                <li>Emotion control</li>
                                <li>Speed adjustment</li>
                                <li>Pronunciation</li>
                                <li>SSML support</li>
                            </ul>
                        </div>
                        
                        <div style="background: rgba(255, 255, 255, 0.1); padding: 20px; border-radius: 15px;">
                            <h4 style="margin-bottom: 10px;">🚀 Use Cases</h4>
                            <ul style="text-align: left; margin: 0; padding-left: 20px;">
                                <li>Accessibility</li>
                                <li>Audio books</li>
                                <li>Voice assistants</li>
                                <li>E-learning</li>
                                <li>Announcements</li>
                                <li>Content creation</li>
                            </ul>
                        </div>
                    </div>
                    
                    <div style="background: rgba(255, 255, 255, 0.2); border-radius: 15px; padding: 25px; margin: 30px 0;">
                        <h3 style="margin-bottom: 15px;">📝 Try These Texts:</h3>
                        <div style="text-align: left;">
                            <p style="margin: 10px 0; padding: 10px; background: rgba(255, 255, 255, 0.1); border-radius: 8px;">
                                "Welcome to the future of artificial intelligence!"
                            </p>
                            <p style="margin: 10px 0; padding: 10px; background: rgba(255, 255, 255, 0.1); border-radius: 8px;">
                                "The quick brown fox jumps over the lazy dog."
                            </p>
                            <p style="margin: 10px 0; padding: 10px; background: rgba(255, 255, 255, 0.1); border-radius: 8px;">
                                "Local AI processing ensures your privacy and data security."
                            </p>
                            <p style="margin: 10px 0; padding: 10px; background: rgba(255, 255, 255, 0.1); border-radius: 8px;">
                                Try longer texts, numbers, or special characters!
                            </p>
                        </div>
                    </div>
                </div>
            </div>
            """
        ]
        
        # Navigation functions
        def navigate_to_step(step):
            """Navigate to a specific step"""
            global session_data
            session_data["current_step"] = step
            
            # Update progress
            progress_html = get_progress_html(step)
            
            # Update navigation buttons
            back_visible = step > 0 and step != 9  # Show back button on all steps except step 0 and step 9
            
            # Determine next button text and visibility
            if step == 7:  # Quiz introduction page
                next_text = "🧠 Start Quiz"
                next_visible = True
            elif step == 8:  # During quiz
                next_text = "➡️ Next"
                next_visible = True  # Show next button to go to completion
            elif step == 9:  # Completion page
                next_text = "🔄 Start Again"
                next_visible = True
            else:
                next_text = "➡️ Next"
                next_visible = True
            
            # Update main content based on step
            if step == 9:  # Completion page
                score = session_data.get("quiz_score", 0)
                total = len(QUIZ_QUESTIONS)
                percentage = (score / total) * 100 if total > 0 else 0
                name = session_data.get("name", "Participant")
                cert_data = session_data.get("certificate_data", {})
                cert_id = cert_data.get("certificate_id", "N/A") if cert_data else "N/A"
                
                content = f"""
                <div style="text-align: center; padding: 50px; background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); min-height: 70vh; color: white; border-radius: 20px;">
                    <div style="background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); border-radius: 20px; padding: 40px; max-width: 800px; margin: 0 auto;">
                        <h1 style="font-size: 3rem; margin-bottom: 20px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                            🎉 Congratulations, {name}!
                        </h1>
                        <h2 style="font-size: 1.8rem; margin-bottom: 30px;">
                            You've completed the AI Demo Tour!
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
                            <h3 style="margin-bottom: 15px;">🎓 Your Digital Certificate</h3>
                            <p style="font-size: 1.1rem; line-height: 1.6;">
                                Your personalized certificate with QR code has been generated!<br>
                                <strong>Certificate ID:</strong> {cert_id}
                            </p>
                            <p style="font-size: 0.9rem; margin-top: 15px; opacity: 0.8;">
                                📱 The QR code contains all your completion details and can be scanned for verification.
                            </p>
                        </div>
                        
                        <div style="background: rgba(255, 255, 255, 0.1); padding: 20px; border-radius: 10px; margin: 20px 0;">
                            <h3>🚀 What's Next?</h3>
                            <ul style="text-align: left; max-width: 500px; margin: 0 auto;">
                                <li style="margin: 8px 0;">📱 Download your certificate using the button below</li>
                                <li style="margin: 8px 0;">🔍 Share the QR code for verification</li>
                                <li style="margin: 8px 0;">🤖 Explore the full AI interface</li>
                                <li style="margin: 8px 0;">🌟 Set up your own local AI environment</li>
                            </ul>
                        </div>
                        
                        <p style="font-size: 1.2rem; margin-top: 30px;">
                            <strong>Thank you for participating!</strong> 🌟
                        </p>
                    </div>
                </div>
                """
            elif step < len(page_contents):
                content = page_contents[step]
            else:
                content = page_contents[0]  # Fallback to intro
            
            # Show/hide interactive components
            name_visible = step == 2
            llm_visible = step == 3
            vision_visible = step == 4
            whisper_visible = step == 5
            tts_visible = step == 6
            quiz_visible = step == 8
            certificate_visible = step == 9 and session_data.get("certificate_path")
            completion_visible = step == 9
            
            # Update content for quiz step
            if step == 7:
                content = """
                <div style="text-align: center; padding: 50px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 70vh; color: white; border-radius: 20px;">
                    <div style="background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); border-radius: 20px; padding: 40px; max-width: 800px; margin: 0 auto;">
                        <h1 style="font-size: 2.5rem; margin-bottom: 20px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                            🧠 Knowledge Assessment Quiz
                        </h1>
                        <p style="font-size: 1.3rem; margin-bottom: 30px; line-height: 1.6;">
                            Test your understanding of the AI modules you've explored!
                        </p>
                        
                        <div style="background: rgba(255, 255, 255, 0.2); border-radius: 15px; padding: 25px; margin: 30px 0;">
                            <h3 style="margin-bottom: 15px;">📋 Quiz Details:</h3>
                            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 20px;">
                                <div style="background: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 10px;">
                                    <div style="font-size: 1.5rem;">📝</div>
                                    <div style="font-weight: bold;">10 Questions</div>
                                    <div style="font-size: 0.9rem;">Multiple choice format</div>
                                </div>
                                <div style="background: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 10px;">
                                    <div style="font-size: 1.5rem;">⏱️</div>
                                    <div style="font-weight: bold;">No Time Limit</div>
                                    <div style="font-size: 0.9rem;">Take your time</div>
                                </div>
                                <div style="background: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 10px;">
                                    <div style="font-size: 1.5rem;">🎯</div>
                                    <div style="font-weight: bold;">All Modules</div>
                                    <div style="font-size: 0.9rem;">LLM, Vision, Speech, TTS</div>
                                </div>
                                <div style="background: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 10px;">
                                    <div style="font-size: 1.5rem;">📊</div>
                                    <div style="font-weight: bold;">Detailed Results</div>
                                    <div style="font-size: 0.9rem;">See your score breakdown</div>
                                </div>
                            </div>
                        </div>
                        
                        <p style="font-size: 1.1rem; margin-top: 30px;">
                            Click <strong>"🧠 Start Quiz"</strong> below to begin!
                        </p>
                    </div>
                </div>
                """
            
            return (
                progress_html,
                gr.update(visible=back_visible),
                gr.update(value=next_text, visible=next_visible),
                gr.update(value=content),
                gr.update(visible=name_visible),
                gr.update(visible=llm_visible),
                gr.update(visible=vision_visible),
                gr.update(visible=whisper_visible),
                gr.update(visible=tts_visible),
                gr.update(visible=quiz_visible),
                gr.update(visible=certificate_visible, value=session_data.get("certificate_path")),
                gr.update(visible=completion_visible),
                step
            )
        
        def go_next(step, name_value):
            """Go to next step"""
            if step == 2 and name_value and name_value.strip():
                session_data["name"] = name_value.strip()
                session_data["start_time"] = datetime.now()
            
            # Handle "Start Again" from completion page
            if step == 9:  # Completion page - Start Again
                reset_session()
                return navigate_to_step(0) + (gr.update(), gr.update(), gr.update())
            
            if step == 7:  # Quiz introduction - Start Quiz
                # Reset quiz data
                session_data["quiz_answers"] = []
                session_data["quiz_score"] = 0
                session_data["current_question"] = 0
                
                # Initialize first question
                question = QUIZ_QUESTIONS[0]
                question_html = f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 25px; border-radius: 15px; margin: 20px 0;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 15px;">
                        <h3>📚 {question['module']} Question</h3>
                        <span>Question 1 of {len(QUIZ_QUESTIONS)}</span>
                    </div>
                    <p style="font-size: 1.2rem; font-weight: 500; margin-bottom: 20px;">{question['question']}</p>
                </div>
                """
                
                result = navigate_to_step(step + 1)
                return result + (
                    gr.update(value=question_html, visible=True),
                    gr.update(choices=question['options'], value=None, visible=True),
                    0  # Reset quiz question index
                )
            
            if step == 8:  # During quiz - this shouldn't be called as we use submit button
                return navigate_to_step(step + 1) + (gr.update(), gr.update(), gr.update())
            return navigate_to_step(step + 1) + (gr.update(), gr.update(), gr.update())
        
        def go_back(step):
            """Go to previous step"""
            return navigate_to_step(max(0, step - 1)) + (gr.update(), gr.update(), gr.update())
        
        # Event handlers
        next_btn.click(
            go_next,
            inputs=[current_step, name_input],
            outputs=[
                progress_display, back_btn, next_btn, main_content,
                name_input_group, llm_group, vision_group, whisper_group,
                tts_group, quiz_group, certificate_group, completion_display, current_step,
                quiz_question_display, quiz_options, quiz_question_idx
            ]
        )
        
        back_btn.click(
            go_back,
            inputs=[current_step],
            outputs=[
                progress_display, back_btn, next_btn, main_content,
                name_input_group, llm_group, vision_group, whisper_group,
                tts_group, quiz_group, certificate_group, completion_display, current_step,
                quiz_question_display, quiz_options, quiz_question_idx
            ]
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
        
        # Vision AI handlers
        vision_analyze_btn.click(
            analyze_image,
            inputs=[vision_image, vision_question],
            outputs=[vision_result]
        )
        
        # Whisper handlers
        whisper_transcribe_btn.click(
            transcribe_audio,
            inputs=[whisper_audio],
            outputs=[whisper_result]
        )
        
        # TTS handlers
        tts_speak_btn.click(
            text_to_speech,
            inputs=[tts_input],
            outputs=[tts_status, tts_audio]
        )
        
        # Quiz handlers
        quiz_submit_btn.click(
            submit_quiz_answer,
            inputs=[quiz_options, quiz_question_idx],
            outputs=[quiz_question_idx, quiz_question_display, quiz_options, quiz_feedback, certificate_image]
        )
        
        # Direct module navigation handlers
        def go_to_home_direct():
            """Jump directly to Home page"""
            return navigate_to_step(0) + (gr.update(), gr.update(), gr.update())
        
        def go_to_llm_direct():
            """Jump directly to LLM Chat module"""
            return navigate_to_step(3) + (gr.update(), gr.update(), gr.update())
        
        def go_to_vision_direct():
            """Jump directly to Vision AI module"""
            return navigate_to_step(4) + (gr.update(), gr.update(), gr.update())
        
        def go_to_whisper_direct():
            """Jump directly to Speech-to-Text module"""
            return navigate_to_step(5) + (gr.update(), gr.update(), gr.update())
        
        def go_to_tts_direct():
            """Jump directly to Text-to-Speech module"""
            return navigate_to_step(6) + (gr.update(), gr.update(), gr.update())
        
        home_direct_btn.click(
            go_to_home_direct,
            outputs=[
                progress_display, back_btn, next_btn, main_content,
                name_input_group, llm_group, vision_group, whisper_group,
                tts_group, quiz_group, certificate_group, completion_display, current_step,
                quiz_question_display, quiz_options, quiz_question_idx
            ]
        )
        
        llm_direct_btn.click(
            go_to_llm_direct,
            outputs=[
                progress_display, back_btn, next_btn, main_content,
                name_input_group, llm_group, vision_group, whisper_group,
                tts_group, quiz_group, certificate_group, completion_display, current_step,
                quiz_question_display, quiz_options, quiz_question_idx
            ]
        )
        
        vision_direct_btn.click(
            go_to_vision_direct,
            outputs=[
                progress_display, back_btn, next_btn, main_content,
                name_input_group, llm_group, vision_group, whisper_group,
                tts_group, quiz_group, certificate_group, completion_display, current_step,
                quiz_question_display, quiz_options, quiz_question_idx
            ]
        )
        
        whisper_direct_btn.click(
            go_to_whisper_direct,
            outputs=[
                progress_display, back_btn, next_btn, main_content,
                name_input_group, llm_group, vision_group, whisper_group,
                tts_group, quiz_group, certificate_group, completion_display, current_step,
                quiz_question_display, quiz_options, quiz_question_idx
            ]
        )
        
        tts_direct_btn.click(
            go_to_tts_direct,
            outputs=[
                progress_display, back_btn, next_btn, main_content,
                name_input_group, llm_group, vision_group, whisper_group,
                tts_group, quiz_group, certificate_group, completion_display, current_step,
                quiz_question_display, quiz_options, quiz_question_idx
            ]
        )
    
    return demo

if __name__ == "__main__":
    # Initialize models in background
    model_thread = threading.Thread(target=initialize_models)
    model_thread.daemon = True
    model_thread.start()
    
    # Create and launch the interface
    demo = create_interactive_tour()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True  # This will automatically open the browser
    )