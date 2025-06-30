"""
Interactive AI Demo Tour - Multi-GPU Optimized Version for 3x RTX 4090
Optimized for 20+ concurrent users with advanced load balancing and async processing
"""

import gradio as gr
import numpy as np
import tempfile
import os
import json
import time
import asyncio
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import threading
import qrcode
import base64
import io
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import logging
import psutil
import torch
import gc

from models import WhisperSTT, LocalTTS, LocalLLM, LocalVLM, OllamaLLM
from config import Config
from gpu_optimizer import gpu_optimizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelInstance:
    """Represents a model instance on a specific GPU"""
    model: Any
    gpu_id: int
    model_type: str
    load_factor: float = 0.0
    last_used: float = 0.0
    processing: bool = False
    instance_id: str = ""

@dataclass
class UserSession:
    """User session data with performance tracking"""
    session_id: str
    name: str = ""
    start_time: Optional[datetime] = None
    current_step: int = 0
    quiz_answers: List = None
    quiz_score: int = 0
    current_question: int = 0
    certificate_path: Optional[str] = None
    certificate_data: Optional[Dict] = None
    request_count: int = 0
    last_activity: float = 0.0
    
    def __post_init__(self):
        if self.quiz_answers is None:
            self.quiz_answers = []

class MultiGPUModelManager:
    """Advanced model manager for 3x RTX 4090 setup"""
    
    def __init__(self):
        self.gpu_count = torch.cuda.device_count()
        self.model_instances: Dict[str, List[ModelInstance]] = {
            'whisper': [],
            'tts': [],
            'llm': [],
            'vlm': []
        }
        self.processing_pools = {
            'whisper': ThreadPoolExecutor(max_workers=6),
            'tts': ThreadPoolExecutor(max_workers=4),
            'llm': ThreadPoolExecutor(max_workers=8),
            'vlm': ThreadPoolExecutor(max_workers=4)
        }
        self.performance_metrics = {
            'total_requests': 0,
            'completed_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0.0,
            'concurrent_users': 0,
            'gpu_utilization': {}
        }
        self.user_sessions: Dict[str, UserSession] = {}
        self.cleanup_thread = None
        self.monitoring_thread = None
        self.running = False
        
    def initialize_models(self):
        """Initialize multiple model instances across GPUs for optimal load distribution"""
        logger.info("🚀 Initializing Multi-GPU Model Instances for 20+ Users...")
        
        try:
            # GPU 0: Whisper (3 instances) + TTS (2 instances) - Audio processing
            self._load_whisper_instances(gpu_id=0, count=3)
            self._load_tts_instances(gpu_id=0, count=2)
            
            # GPU 1: VLM main instances (3 instances) - Vision processing
            self._load_vlm_instances(gpu_id=1, count=3)
            
            # GPU 2: LLM (4 instances) + VLM secondary (1 instance) - Language processing
            self._load_llm_instances(gpu_id=2, count=4)
            self._load_vlm_instances(gpu_id=2, count=1)
            
            logger.info("✅ All model instances initialized successfully!")
            self._print_model_distribution()
            
            # Start background services
            self.start_background_services()
            
        except Exception as e:
            logger.error(f"❌ Error initializing models: {e}")
            raise
    
    def _load_whisper_instances(self, gpu_id: int, count: int):
        """Load multiple Whisper instances on specified GPU"""
        for i in range(count):
            try:
                with torch.cuda.device(gpu_id):
                    torch.cuda.empty_cache()
                    model = WhisperSTT()
                    instance = ModelInstance(
                        model=model,
                        gpu_id=gpu_id,
                        model_type='whisper',
                        instance_id=f"whisper_{gpu_id}_{i}"
                    )
                    self.model_instances['whisper'].append(instance)
                    logger.info(f"✅ Whisper instance {i+1}/3 loaded on GPU {gpu_id}")
            except Exception as e:
                logger.error(f"❌ Failed to load Whisper instance {i+1}: {e}")
    
    def _load_tts_instances(self, gpu_id: int, count: int):
        """Load multiple TTS instances on specified GPU"""
        for i in range(count):
            try:
                with torch.cuda.device(gpu_id):
                    model = LocalTTS()
                    instance = ModelInstance(
                        model=model,
                        gpu_id=gpu_id,
                        model_type='tts',
                        instance_id=f"tts_{gpu_id}_{i}"
                    )
                    self.model_instances['tts'].append(instance)
                    logger.info(f"✅ TTS instance {i+1}/2 loaded on GPU {gpu_id}")
            except Exception as e:
                logger.error(f"❌ Failed to load TTS instance {i+1}: {e}")
    
    def _load_llm_instances(self, gpu_id: int, count: int):
        """Load multiple LLM instances on specified GPU"""
        for i in range(count):
            try:
                with torch.cuda.device(gpu_id):
                    torch.cuda.empty_cache()
                    try:
                        model = OllamaLLM()
                        is_available, _ = model.test_connection()
                        if not is_available:
                            raise Exception("Ollama not available")
                        model_type_name = "Ollama"
                    except:
                        model = LocalLLM()
                        model_type_name = "Local"
                    
                    instance = ModelInstance(
                        model=model,
                        gpu_id=gpu_id,
                        model_type='llm',
                        instance_id=f"llm_{gpu_id}_{i}"
                    )
                    self.model_instances['llm'].append(instance)
                    logger.info(f"✅ {model_type_name} LLM instance {i+1}/4 loaded on GPU {gpu_id}")
            except Exception as e:
                logger.error(f"❌ Failed to load LLM instance {i+1}: {e}")
    
    def _load_vlm_instances(self, gpu_id: int, count: int):
        """Load multiple VLM instances on specified GPU"""
        for i in range(count):
            try:
                with torch.cuda.device(gpu_id):
                    torch.cuda.empty_cache()
                    model = LocalVLM()
                    instance = ModelInstance(
                        model=model,
                        gpu_id=gpu_id,
                        model_type='vlm',
                        instance_id=f"vlm_{gpu_id}_{i}"
                    )
                    self.model_instances['vlm'].append(instance)
                    logger.info(f"✅ VLM instance {i+1} loaded on GPU {gpu_id}")
            except Exception as e:
                logger.error(f"❌ Failed to load VLM instance {i+1}: {e}")
    
    def _print_model_distribution(self):
        """Print model distribution across GPUs"""
        logger.info("=== Optimized Model Distribution for 20+ Users ===")
        total_instances = 0
        for model_type, instances in self.model_instances.items():
            gpu_dist = {}
            for instance in instances:
                gpu_id = instance.gpu_id
                gpu_dist[gpu_id] = gpu_dist.get(gpu_id, 0) + 1
                total_instances += 1
            logger.info(f"{model_type.upper()}: {gpu_dist} (Total: {len(instances)})")
        
        logger.info(f"Total Model Instances: {total_instances}")
        logger.info("GPU 0: Audio Processing (Whisper + TTS)")
        logger.info("GPU 1: Vision Processing (VLM Main)")
        logger.info("GPU 2: Language Processing (LLM + VLM Secondary)")
        logger.info("=" * 50)
    
    def get_best_instance(self, model_type: str) -> Optional[ModelInstance]:
        """Get the best available instance using intelligent load balancing"""
        instances = self.model_instances.get(model_type, [])
        if not instances:
            return None
        
        available = [inst for inst in instances if not inst.processing]
        if not available:
            available = sorted(instances, key=lambda x: x.load_factor)[:1]
        
        if not available:
            return None
        
        best = min(available, key=lambda x: (x.load_factor, x.last_used))
        return best
    
    async def process_request_async(self, model_type: str, data: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Process request asynchronously with load balancing"""
        start_time = time.time()
        self.performance_metrics['total_requests'] += 1
        
        if session_id in self.user_sessions:
            self.user_sessions[session_id].request_count += 1
            self.user_sessions[session_id].last_activity = time.time()
        
        try:
            instance = self.get_best_instance(model_type)
            if not instance:
                raise Exception(f"No available {model_type} instances")
            
            instance.processing = True
            instance.load_factor += 0.2
            instance.last_used = time.time()
            
            result = await self._execute_model_request(instance, model_type, data)
            
            processing_time = time.time() - start_time
            self.performance_metrics['completed_requests'] += 1
            
            total_completed = self.performance_metrics['completed_requests']
            current_avg = self.performance_metrics['avg_response_time']
            self.performance_metrics['avg_response_time'] = (
                (current_avg * (total_completed - 1) + processing_time) / total_completed
            )
            
            return {
                'success': True,
                'result': result,
                'processing_time': processing_time,
                'gpu_id': instance.gpu_id,
                'instance_id': instance.instance_id
            }
            
        except Exception as e:
            self.performance_metrics['failed_requests'] += 1
            logger.error(f"Request processing failed for {model_type}: {e}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
        
        finally:
            if instance:
                instance.processing = False
                instance.load_factor = max(0, instance.load_factor - 0.2)
    
    async def _execute_model_request(self, instance: ModelInstance, model_type: str, data: Dict[str, Any]) -> Any:
        """Execute the actual model request"""
        model = instance.model
        loop = asyncio.get_event_loop()
        
        if model_type == 'whisper':
            return await loop.run_in_executor(
                self.processing_pools['whisper'],
                model.transcribe,
                data['audio_path']
            )
        elif model_type == 'tts':
            return await loop.run_in_executor(
                self.processing_pools['tts'],
                model.speak,
                data['text']
            )
        elif model_type == 'llm':
            return await loop.run_in_executor(
                self.processing_pools['llm'],
                model.generate_response,
                data['prompt'],
                data.get('conversation_history', [])
            )
        elif model_type == 'vlm':
            return await loop.run_in_executor(
                self.processing_pools['vlm'],
                model.analyze_image,
                data['image'],
                data.get('question', 'Describe this image')
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def start_background_services(self):
        """Start background monitoring and cleanup services"""
        self.running = True
        
        self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_thread.start()
        
        self.monitoring_thread = threading.Thread(target=self._monitoring_worker, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("✅ Background services started")
    
    def _cleanup_worker(self):
        """Background worker for memory cleanup and session management"""
        while self.running:
            try:
                current_time = time.time()
                inactive_sessions = [
                    sid for sid, session in self.user_sessions.items()
                    if current_time - session.last_activity > 1800
                ]
                
                for sid in inactive_sessions:
                    del self.user_sessions[sid]
                    logger.info(f"Cleaned up inactive session: {sid}")
                
                if len(inactive_sessions) > 0 or current_time % 300 < 30:
                    self._cleanup_gpu_memory()
                
                time.sleep(30)
                
            except Exception as e:
                logger.error(f"Cleanup worker error: {e}")
    
    def _monitoring_worker(self):
        """Background worker for performance monitoring"""
        while self.running:
            try:
                self.performance_metrics['gpu_utilization'] = gpu_optimizer.get_gpu_utilization()
                self.performance_metrics['concurrent_users'] = len(self.user_sessions)
                
                if time.time() % 120 < 30:
                    self._log_performance_stats()
                
                time.sleep(30)
                
            except Exception as e:
                logger.error(f"Monitoring worker error: {e}")
    
    def _cleanup_gpu_memory(self):
        """Clean up GPU memory across all devices"""
        for gpu_id in range(self.gpu_count):
            with torch.cuda.device(gpu_id):
                torch.cuda.empty_cache()
        gc.collect()
    
    def _log_performance_stats(self):
        """Log current performance statistics"""
        metrics = self.performance_metrics
        logger.info(f"📊 Performance Stats - Users: {metrics['concurrent_users']}, "
                   f"Requests: {metrics['total_requests']}, "
                   f"Success Rate: {(metrics['completed_requests']/max(1, metrics['total_requests']))*100:.1f}%, "
                   f"Avg Response: {metrics['avg_response_time']:.2f}s")
    
    def get_or_create_session(self, session_id: str) -> UserSession:
        """Get existing session or create new one"""
        if session_id not in self.user_sessions:
            self.user_sessions[session_id] = UserSession(
                session_id=session_id,
                last_activity=time.time()
            )
        else:
            self.user_sessions[session_id].last_activity = time.time()
        
        return self.user_sessions[session_id]
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status for monitoring"""
        return {
            'performance_metrics': self.performance_metrics.copy(),
            'active_sessions': len(self.user_sessions),
            'model_instances': {
                model_type: len(instances)
                for model_type, instances in self.model_instances.items()
            },
            'system_resources': {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'memory_used_gb': psutil.virtual_memory().used / (1024**3)
            }
        }

# Global model manager instance
model_manager = MultiGPUModelManager()

# Quiz questions
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
    }
]

def generate_certificate_qr(name, score, total):
    """Generate a QR code certificate for the participant"""
    try:
        certificate_data = {
            "participant": name,
            "course": "AI Demo Tour - Multi-GPU Optimized",
            "score": f"{score}/{total}",
            "percentage": f"{(score/total)*100:.0f}%",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "issuer": "Local AI Demo Stack - 3x RTX 4090",
            "certificate_id": f"AIDT-{datetime.now().strftime('%Y%m%d')}-{hash(name) % 10000:04d}"
        }
        
        certificate_text = f"""
🎓 CERTIFICATE OF COMPLETION 🎓

Participant: {name}
Course: AI Demo Tour - Multi-GPU Optimized
Score: {score}/{total} ({(score/total)*100:.0f}%)
Date: {datetime.now().strftime("%B %d, %Y")}
Certificate ID: {certificate_data['certificate_id']}

This certifies that {name} has successfully completed the Interactive AI Demo Tour on a 3x RTX 4090 optimized system.

Issued by: Local AI Demo Stack - Multi-GPU Optimized
        """.strip()
        
        qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=10, border=4)
        qr.add_data(certificate_text)
        qr.make(fit=True)
        
        qr_img = qr.make_image(fill_color="black", back_color="white")
        
        cert_width, cert_height = 800, 600
        cert_img = Image.new('RGB', (cert_width, cert_height), 'white')
        draw = ImageDraw.Draw(cert_img)
        
        try:
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
            text_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        except:
            title_font = ImageFont.load_default()
            text_font = ImageFont.load_default()
        
        border_color = "#667eea"
        draw.rectangle([10, 10, cert_width-10, cert_height-10], outline=border_color, width=3)
        
        title_text = "🎓 CERTIFICATE OF COMPLETION 🎓"
        title_bbox = draw.textbbox((0, 0), title_text, font=title_font)
        title_width = title_bbox[2] - title_bbox[0]
        draw.text(((cert_width - title_width) // 2, 40), title_text, fill=border_color, font=title_font)
        
        name_text = f"This certifies that {name}"
        name_bbox = draw.textbbox((0, 0), name_text, font=text_font)
        name_width = name_bbox[2] - name_bbox[0]
        draw.text(((cert_width - name_width) // 2, 120), name_text, fill="black", font=text_font)
        
        course_name = "Interactive AI Demo Tour - Multi-GPU Optimized"
        course_name_bbox = draw.textbbox((0, 0), course_name, font=text_font)
        course_name_width = course_name_bbox[2] - course_name_bbox[0]
        draw.text(((cert_width - course_name_width) // 2, 160), course_name, fill=border_color, font=text_font)
        
        score_text = f"Score: {score}/{total} ({(score/total)*100:.0f}%)"
        score_bbox = draw.textbbox((0, 0), score_text, font=text_font)
        score_width = score_bbox[2] - score_bbox[0]
        draw.text(((cert_width - score_width) // 2, 200), score_text, fill="black", font=text_font)
        
        qr_size = 120
        qr_resized = qr_img.resize((qr_size, qr_size))
        qr_x = cert_width - qr_size - 40
        qr_y = cert_height - qr_size - 40
        cert_img.paste(qr_resized, (qr_x, qr_y))
        
        cert_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        cert_img.save(cert_temp.name, 'PNG')
        cert_temp.close()
        
        return cert_temp.name, certificate_data
        
    except Exception as e:
        logger.error(f"Error generating certificate: {e}")
        return None, None

def get_progress_html(step, total_steps=10):
    """Generate progress bar HTML with performance indicators"""
    if step >= 9:
        percentage = 100
    else:
        percentage = ((step + 1) / total_steps) * 100
    
    step_names = ["Intro", "About", "Name", "LLM Chat", "Vision AI", "Speech-to-Text", "Text-to-Speech", "Quiz Intro", "Quiz", "Complete"]
    current_step_name = step_names[min(step, len(step_names)-1)]
    
    status = model_manager.get_system_status()
    concurrent_users = status['active_sessions']
    avg_response = status['performance_metrics']['avg_response_time']
    
    return f"""
    <div style="background: #e2e8f0; border-radius: 10px; padding: 15px; margin: 20px 0;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-weight: 600; color: #333;">Progress: {current_step_name}</span>
            <span style="color: #666;">Step {step + 1} of {total_steps}</span>
        </div>
        <div style="background: #cbd5e1; border-radius: 3px; height: 8px; margin-bottom: 10px;">
            <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); height: 8px; border-radius: 3px; width: {percentage}%; transition: width 0.5s ease;"></div>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.9rem; color: #666;">
            <span>👥 Active Users: {concurrent_users}</span>
            <span>⚡ Avg Response: {avg_response:.2f}s</span>
        </div>
    </div>
    """

# Optimized AI processing functions
async def llm_chat_async(message, history, session_id):
    """Async LLM chat with multi-GPU load balancing"""
    if not message.strip():
        return history, ""
    
    try:
        result = await model_manager.process_request_async(
            model_type='llm',
            data={'prompt': message, 'conversation_history': []},
            session_id=session_id
        )
        
        if result['success']:
            ai_response = result['result']
            gpu_info = f" (GPU {result['gpu_id']})"
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": ai_response + gpu_info})
        else:
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": f"Error: {result['error']}"})
        
        return history, ""
        
    except Exception as e:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": f"Error: {str(e)}"})
        return history, ""

def llm_chat(message, history, session_id="default"):
    """Sync wrapper for LLM chat"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(llm_chat_async(message, history, session_id))
    finally:
        loop.close()

async def analyze_image_async(image, question, session_id):
    """Async image analysis with multi-GPU load balancing"""
    if image is None:
        return "Please upload an image or capture one with the camera first."
    
    try:
        if not question.strip():
            question = "Describe this image in detail, including objects, people, colors, and any text you can see."
        
        result = await model_manager.process_request_async(
            model_type='vlm',
            data={'image': image, 'question': question},
            session_id=session_id
        )
        
        if result['success']:
            return f"🤖 **AI Analysis (GPU {result['gpu_id']}):**\n\n{result['result']}"
        else:
            return f"Error analyzing image: {result['error']}"
        
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

def analyze_image(image, question, session_id="default"):
    """Sync wrapper for image analysis"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(analyze_image_async(image, question, session_id))
    finally:
        loop.close()

async def transcribe_audio_async(audio, session_id):
    """Async audio transcription with multi-GPU load balancing"""
    if audio is None:
        return "Please record some audio first."
    
    try:
        sample_rate, audio_data = audio
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        
        if audio_data.dtype != np.int16:
            audio_data = (audio_data * 32767).astype(np.int16)
        
        import soundfile as sf
        sf.write(temp_audio.name, audio_data, sample_rate)
        
        result = await model_manager.process_request_async(
            model_type='whisper',
            data={'audio_path': temp_audio.name},
            session_id=session_id
        )
        
        os.unlink(temp_audio.name)
        
        if result['success']:
            return f"🎤 **You said (GPU {result['gpu_id']}):** {result['result']}"
        else:
            return f"Error transcribing audio: {result['error']}"
        
    except Exception as e:
        return f"Error transcribing audio: {str(e)}"

def transcribe_audio(audio, session_id="default"):
    """Sync wrapper for audio transcription"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(transcribe_audio_async(audio, session_id))
    finally:
        loop.close()

async def text_to_speech_async(text, session_id):
    """Async text-to-speech with multi-GPU load balancing"""
    if not text.strip():
        return "Please enter some text first.", None
    
    try:
        result = await model_manager.process_request_async(
            model_type='tts',
            data={'text': text},
            session_id=session_id
        )
        
        if result['success']:
            return f"🔊 **Speaking (GPU {result['gpu_id']}):** {text}", result['result']
        else:
            return f"Error generating speech: {result['error']}", None
        
    except Exception as e:
        return f"Error generating speech: {str(e)}", None

def text_to_speech(text, session_id="default"):
    """Sync wrapper for text-to-speech"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(text_to_speech_async(text, session_id))
    finally:
        loop.close()

def submit_quiz_answer(selected_option, current_q, session_id="default"):
    """Submit a quiz answer and move to next question"""
    session = model_manager.get_or_create_session(session_id)
    
    if selected_option is None:
        return "Please select an answer first.", current_q, gr.update(), gr.update(), gr.update()
    
    question = QUIZ_QUESTIONS[current_q]
    selected_idx = question['options'].index(selected_option)
    
    is_correct = selected_idx == question['correct']
    if is_correct:
        session.quiz_score += 1
    
    session.quiz_answers.append({
        "question": current_q,
        "selected": selected_idx,
        "correct": is_correct
    })
    
    next_q = current_q + 1
    
    if next_q >= len(QUIZ_QUESTIONS):
        # Quiz completed - Generate certificate
        score = session.quiz_score
        total = len(QUIZ_QUESTIONS)
        name = session.name or "Participant"
        
        cert_path, cert_data = generate_certificate_qr(name, score, total)
        
        session.current_step = 9
        session.certificate_path = cert_path
        session.certificate_data = cert_data
        
        return (
            next_q,
            gr.update(value=""),
            gr.update(choices=[], value=None),
            gr.update(),
            gr.update(value=cert_path, visible=True) if cert_path else gr.update()
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
            gr.update(),
            gr.update()
        )

def create_optimized_interactive_tour():
    """Create the optimized interactive tour interface for 20+ concurrent users"""
    
    custom_css = """
    * {
        font-family: 'Helvetica Neue', 'Helvetica', 'DejaVu Sans', 'Arial', sans-serif !important;
    }
    
    .gradio-container {
        max-width: 1200px !important;
        margin: 0 auto !important;
        background: #f8fafc !important;
    }
    
    .performance-indicator {
        position: fixed;
        top: 10px;
        right: 10px;
        background: rgba(0, 0, 0, 0.8);
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-size: 12px;
        z-index: 1000;
    }
    
    .name-input-prominent {
        border: 2px solid #4facfe !important;
        border-radius: 10px !important;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.2) !important;
        transition: all 0.3s ease !important;
    }
    
    .name-input-prominent:focus {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3) !important;
        border-color: #00f2fe !important;
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
        title="🚀 Multi-GPU AI Demo Tour - Optimized for 20+ Users",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="purple",
            neutral_hue="slate",
            font=["Helvetica Neue", "Helvetica", "DejaVu Sans", "Arial", "sans-serif"]
        ),
        css=custom_css
    ) as demo:
        
        # Generate unique session ID for each user
        session_id = gr.State(lambda: str(uuid.uuid4()))
        current_step = gr.State(0)
        quiz_question_idx = gr.State(0)
        
        # Performance indicator
        status_display = gr.HTML("""
        <div class="performance-indicator">
            <div>🚀 Multi-GPU AI Server</div>
            <div>3x RTX 4090 Optimized</div>
            <div>Status: Ready for 20+ Users</div>
        </div>
        """)
        
        # Progress indicator with performance metrics
        progress_display = gr.HTML(get_progress_html(0))
        
        # Navigation buttons
        with gr.Row():
            back_btn = gr.Button("⬅️ Back", variant="secondary", visible=False, elem_classes=["btn-secondary"])
            next_btn = gr.Button("➡️ Next", variant="primary", elem_classes=["btn-primary"])
        
        # Quick access navigation
        with gr.Row():
            gr.HTML("<h3 style='text-align: center; margin: 20px 0; color: #667eea;'>🚀 Quick Access - Multi-GPU Optimized:</h3>")
        with gr.Row():
            home_direct_btn = gr.Button("🏠 Home", variant="secondary", elem_classes=["btn-secondary"])
            llm_direct_btn = gr.Button("💬 LLM Chat", variant="secondary", elem_classes=["btn-secondary"])
            vision_direct_btn = gr.Button("👁️ Vision AI", variant="secondary", elem_classes=["btn-secondary"])
            whisper_direct_btn = gr.Button("🎤 Speech-to-Text", variant="secondary", elem_classes=["btn-secondary"])
            tts_direct_btn = gr.Button("🔊 Text-to-Speech", variant="secondary", elem_classes=["btn-secondary"])
        
        # Name input field - prominently positioned above main content
        with gr.Group(visible=False) as name_input_group:
            name_input = gr.Textbox(
                placeholder="Enter your full name...",
                label="👤 Your Name",
                type="text",
                elem_classes=["name-input-prominent"]
            )
        
        # Main content area
        main_content = gr.HTML("""
        <div style="text-align: center; padding: 50px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 70vh; color: white; border-radius: 20px;">
            <div style="background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); border-radius: 20px; padding: 40px; max-width: 800px; margin: 0 auto;">
                <h1 style="font-size: 3rem; margin-bottom: 20px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                    🚀 Multi-GPU AI Demo Tour
                </h1>
                <p style="font-size: 1.3rem; margin-bottom: 30px; line-height: 1.6;">
                    Experience blazing-fast AI with 3x RTX 4090 optimization
                </p>
                <div style="background: rgba(255, 255, 255, 0.2); border-radius: 15px; padding: 25px; margin: 30px 0;">
                    <h3 style="margin-bottom: 15px;">⚡ Performance Features:</h3>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 20px;">
                        <div style="background: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 10px;">
                            <div style="font-size: 1.5rem;">🔥</div>
                            <div style="font-weight: bold;">20+ Users</div>
                            <div style="font-size: 0.9rem;">Concurrent support</div>
                        </div>
                        <div style="background: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 10px;">
                            <div style="font-size: 1.5rem;">⚡</div>
                            <div style="font-weight: bold;">Multi-GPU</div>
                            <div style="font-size: 0.9rem;">Load balancing</div>
                        </div>
                        <div style="background: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 10px;">
                            <div style="font-size: 1.5rem;">🚀</div>
                            <div style="font-weight: bold;">Async Processing</div>
                            <div style="font-size: 0.9rem;">Non-blocking operations</div>
                        </div>
                        <div style="background: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 10px;">
                            <div style="font-size: 1.5rem;">📊</div>
                            <div style="font-weight: bold;">Real-time Metrics</div>
                            <div style="font-size: 0.9rem;">Performance monitoring</div>
                        </div>
                    </div>
                </div>
                <p style="font-size: 1.1rem; margin-top: 30px;">
                    ⏱️ <strong>Duration:</strong> 15-20 minutes<br>
                    🏆 <strong>Completion:</strong> Quiz & Certificate<br>
                    🔥 <strong>Optimized for:</strong> 20+ concurrent users
                </p>
            </div>
        </div>
        """)
        
        # Interactive components with session support
        with gr.Group(visible=False) as llm_group:
            llm_chatbot = gr.Chatbot(
                label="💬 Chat with AI (Multi-GPU Load Balanced)",
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
                        sources=["webcam", "upload"],
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
                        label="🤖 AI Analysis (Multi-GPU)",
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
                    label="📝 Transcription (Multi-GPU)",
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
                        value="Hello! This is a demonstration of optimized text-to-speech technology running on multiple GPUs."
                    )
                    tts_speak_btn = gr.Button("🔊 Generate Speech", variant="primary")
                with gr.Column():
                    tts_status = gr.Textbox(label="Status (Multi-GPU)", lines=2)
                    tts_audio = gr.Audio(label="🎵 Generated Speech", autoplay=True)
        
        with gr.Group(visible=False) as quiz_group:
            quiz_question_display = gr.HTML()
            quiz_options = gr.Radio(
                choices=[],
                label="Select your answer:",
                interactive=True
            )
            quiz_submit_btn = gr.Button("✅ Submit Answer", variant="primary")
            quiz_feedback = gr.HTML(visible=False)
        
        # Certificate display
        with gr.Group(visible=False) as certificate_group:
            certificate_image = gr.Image(
                label="🎓 Your Certificate (Multi-GPU Optimized)",
                type="filepath",
                height=400,
                show_download_button=True
            )
        
        # Completion display
        completion_display = gr.HTML(visible=False)
        
        # Navigation functions with session support
        def navigate_to_step(step, session_id_val):
            """Navigate to a specific step with session support"""
            session = model_manager.get_or_create_session(session_id_val)
            session.current_step = step
            
            progress_html = get_progress_html(step)
            
            back_visible = step > 0 and step != 9
            
            if step == 7:
                next_text = "🧠 Start Quiz"
                next_visible = True
            elif step == 8:
                next_text = "➡️ Next"
                next_visible = True
            elif step == 9:
                next_text = "🔄 Start Again"
                next_visible = True
            else:
                next_text = "➡️ Next"
                next_visible = True
            
            if step == 9:  # Completion page
                score = session.quiz_score
                total = len(QUIZ_QUESTIONS)
                percentage = (score / total) * 100 if total > 0 else 0
                name = session.name or "Participant"
                cert_data = session.certificate_data or {}
                cert_id = cert_data.get("certificate_id", "N/A")
                
                content = f"""
                <div style="text-align: center; padding: 50px; background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); min-height: 70vh; color: white; border-radius: 20px;">
                    <div style="background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); border-radius: 20px; padding: 40px; max-width: 800px; margin: 0 auto;">
                        <h1 style="font-size: 3rem; margin-bottom: 20px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                            🎉 Congratulations, {name}!
                        </h1>
                        <h2 style="font-size: 1.8rem; margin-bottom: 30px;">
                            Multi-GPU AI Demo Tour Completed!
                        </h2>
                        
                        <div style="background: rgba(255, 255, 255, 0.2); border-radius: 15px; padding: 25px; margin: 30px 0;">
                            <h3 style="margin-bottom: 15px;">📊 Your Results:</h3>
                            <div style="font-size: 2rem; margin: 15px 0;">
                                <strong>{score}/{total}</strong> ({percentage:.0f}%)
                            </div>
                            <div style="background: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 10px; margin: 15px 0;">
                                {'🏆 Excellent work on the optimized system!' if percentage >= 80 else '👍 Good job with multi-GPU AI!' if percentage >= 60 else '📚 Keep exploring AI technologies!'}
                            </div>
                        </div>
                        
                        <div style="background: rgba(255, 255, 255, 0.2); border-radius: 15px; padding: 25px; margin: 30px 0;">
                            <h3 style="margin-bottom: 15px;">🎓 Your Multi-GPU Certificate</h3>
                            <p style="font-size: 1.1rem; line-height: 1.6;">
                                Certificate ID: {cert_id}<br>
                                <strong>Powered by:</strong> 3x RTX 4090 Multi-GPU System
                            </p>
                        </div>
                        
                        <p style="font-size: 1.2rem; margin-top: 30px;">
                            <strong>Thank you for testing our optimized AI system!</strong> 🚀
                        </p>
                    </div>
                </div>
                """
            elif step == 7:  # Quiz introduction
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
                                    <div style="font-weight: bold;">5 Questions</div>
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
            else:
                # Define complete page content array
                page_contents = [
                    # Page 0: Intro
                    """
                    <div style="text-align: center; padding: 50px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 70vh; color: white; border-radius: 20px;">
                        <div style="background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); border-radius: 20px; padding: 40px; max-width: 800px; margin: 0 auto;">
                            <h1 style="font-size: 3rem; margin-bottom: 20px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                                🚀 Multi-GPU AI Demo Tour!
                            </h1>
                            <p style="font-size: 1.3rem; margin-bottom: 30px; line-height: 1.6;">
                                Embark on an interactive journey through the world of Local AI - Optimized for 20+ Users
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
                                🏆 <strong>Completion:</strong> Quiz & Certificate<br>
                                🔥 <strong>Powered by:</strong> 3x RTX 4090 Multi-GPU
                            </p>
                        </div>
                    </div>
                    """,
                    # Page 1: About
                    """
                    <div style="padding: 40px; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); min-height: 70vh; color: white; border-radius: 20px;">
                        <div style="background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); border-radius: 20px; padding: 40px; max-width: 800px; margin: 0 auto;">
                            <h2 style="font-size: 2.5rem; text-align: center; margin-bottom: 30px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                                🚀 About This Multi-GPU AI Demo Stack
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
                                    <h3 style="margin-bottom: 15px;">🔥 Multi-GPU Power</h3>
                                    <p>Optimized for 3x RTX 4090 setup supporting 20+ concurrent users with intelligent load balancing.</p>
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
                    # Page 3: LLM Chat
                    """
                    <div style="padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 70vh; color: white; border-radius: 20px;">
                        <div style="background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); border-radius: 20px; padding: 40px; max-width: 800px; margin: 0 auto;">
                            <h2 style="font-size: 2.5rem; text-align: center; margin-bottom: 30px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                                💬 Large Language Model (LLM) Chat
                            </h2>
                            
                            <div style="background: rgba(255, 255, 255, 0.2); padding: 25px; border-radius: 15px; margin-bottom: 30px;">
                                <h3 style="margin-bottom: 15px;">🧠 What is an LLM?</h3>
                                <p style="line-height: 1.6;">
                                    Large Language Models are AI systems trained on vast amounts of text data to understand and generate human-like text.
                                    They can engage in conversations, answer questions, help with writing, coding, analysis, and much more.
                                </p>
                            </div>
                            
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 30px;">
                                <div style="background: rgba(255, 255, 255, 0.1); padding: 20px; border-radius: 10px;">
                                    <h4 style="margin-bottom: 10px;">🎯 Try asking about:</h4>
                                    <ul style="font-size: 0.9rem; line-height: 1.5;">
                                        <li>General knowledge questions</li>
                                        <li>Creative writing assistance</li>
                                        <li>Problem-solving help</li>
                                        <li>Code explanations</li>
                                    </ul>
                                </div>
                                
                                <div style="background: rgba(255, 255, 255, 0.1); padding: 20px; border-radius: 10px;">
                                    <h4 style="margin-bottom: 10px;">💡 Example prompts:</h4>
                                    <ul style="font-size: 0.9rem; line-height: 1.5;">
                                        <li>"Explain quantum computing"</li>
                                        <li>"Write a short poem about AI"</li>
                                        <li>"Help me plan a healthy meal"</li>
                                        <li>"What's the weather like?"</li>
                                    </ul>
                                </div>
                            </div>
                            
                            <div style="background: rgba(255, 255, 255, 0.2); padding: 20px; border-radius: 15px; text-align: center;">
                                <h3 style="margin-bottom: 10px;">🔥 Multi-GPU Optimization</h3>
                                <p style="font-size: 1rem; line-height: 1.5;">
                                    This LLM is running locally with intelligent load balancing across multiple GPUs,
                                    ensuring fast responses even with many concurrent users!
                                </p>
                            </div>
                        </div>
                    </div>
                    """,
                    # Page 4: Vision AI
                    """
                    <div style="padding: 40px; background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); min-height: 70vh; color: white; border-radius: 20px;">
                        <div style="background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); border-radius: 20px; padding: 40px; max-width: 800px; margin: 0 auto;">
                            <h2 style="font-size: 2.5rem; text-align: center; margin-bottom: 30px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                                👁️ Vision Language Model (VLM)
                            </h2>
                            
                            <div style="background: rgba(255, 255, 255, 0.2); padding: 25px; border-radius: 15px; margin-bottom: 30px;">
                                <h3 style="margin-bottom: 15px;">🔍 What is Vision AI?</h3>
                                <p style="line-height: 1.6;">
                                    Vision Language Models combine computer vision with language understanding. They can analyze images,
                                    identify objects, read text, understand scenes, and answer questions about visual content in natural language.
                                </p>
                            </div>
                            
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 30px;">
                                <div style="background: rgba(255, 255, 255, 0.1); padding: 20px; border-radius: 10px;">
                                    <h4 style="margin-bottom: 10px;">📸 What it can analyze:</h4>
                                    <ul style="font-size: 0.9rem; line-height: 1.5;">
                                        <li>Objects and people in photos</li>
                                        <li>Text within images (OCR)</li>
                                        <li>Scenes and environments</li>
                                        <li>Colors, styles, and composition</li>
                                    </ul>
                                </div>
                                
                                <div style="background: rgba(255, 255, 255, 0.1); padding: 20px; border-radius: 10px;">
                                    <h4 style="margin-bottom: 10px;">💡 Try asking:</h4>
                                    <ul style="font-size: 0.9rem; line-height: 1.5;">
                                        <li>"What's in this image?"</li>
                                        <li>"Read the text in this photo"</li>
                                        <li>"Describe the mood/atmosphere"</li>
                                        <li>"Count the objects you see"</li>
                                    </ul>
                                </div>
                            </div>
                            
                            <div style="background: rgba(255, 255, 255, 0.2); padding: 20px; border-radius: 15px; text-align: center;">
                                <h3 style="margin-bottom: 10px;">📱 How to use:</h3>
                                <p style="font-size: 1rem; line-height: 1.5;">
                                    Upload an image or use your camera to capture one, then ask any question about it.
                                    The AI will analyze the visual content and provide detailed descriptions!
                                </p>
                            </div>
                        </div>
                    </div>
                    """,
                    # Page 5: Speech-to-Text
                    """
                    <div style="padding: 40px; background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); min-height: 70vh; color: white; border-radius: 20px;">
                        <div style="background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); border-radius: 20px; padding: 40px; max-width: 800px; margin: 0 auto;">
                            <h2 style="font-size: 2.5rem; text-align: center; margin-bottom: 30px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                                🎤 Speech-to-Text (Whisper)
                            </h2>
                            
                            <div style="background: rgba(255, 255, 255, 0.2); padding: 25px; border-radius: 15px; margin-bottom: 30px;">
                                <h3 style="margin-bottom: 15px;">🗣️ What is Speech-to-Text?</h3>
                                <p style="line-height: 1.6;">
                                    Speech-to-Text technology converts spoken words into written text. We're using OpenAI's Whisper model,
                                    which is highly accurate and supports multiple languages, accents, and even handles background noise well.
                                </p>
                            </div>
                            
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 30px;">
                                <div style="background: rgba(255, 255, 255, 0.1); padding: 20px; border-radius: 10px;">
                                    <h4 style="margin-bottom: 10px;">🌟 Key Features:</h4>
                                    <ul style="font-size: 0.9rem; line-height: 1.5;">
                                        <li>High accuracy transcription</li>
                                        <li>Multiple language support</li>
                                        <li>Handles accents and dialects</li>
                                        <li>Works with background noise</li>
                                    </ul>
                                </div>
                                
                                <div style="background: rgba(255, 255, 255, 0.1); padding: 20px; border-radius: 10px;">
                                    <h4 style="margin-bottom: 10px;">💼 Real-world uses:</h4>
                                    <ul style="font-size: 0.9rem; line-height: 1.5;">
                                        <li>Meeting transcriptions</li>
                                        <li>Voice notes and memos</li>
                                        <li>Accessibility features</li>
                                        <li>Content creation</li>
                                    </ul>
                                </div>
                            </div>
                            
                            <div style="background: rgba(255, 255, 255, 0.2); padding: 20px; border-radius: 15px; text-align: center;">
                                <h3 style="margin-bottom: 10px;">🎙️ How to use:</h3>
                                <p style="font-size: 1rem; line-height: 1.5;">
                                    Click the microphone button to start recording, speak clearly, then click stop.
                                    The AI will transcribe your speech into text with high accuracy!
                                </p>
                            </div>
                        </div>
                    </div>
                    """,
                    # Page 6: Text-to-Speech
                    """
                    <div style="padding: 40px; background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); min-height: 70vh; color: #333; border-radius: 20px;">
                        <div style="background: rgba(255, 255, 255, 0.3); backdrop-filter: blur(10px); border-radius: 20px; padding: 40px; max-width: 800px; margin: 0 auto;">
                            <h2 style="font-size: 2.5rem; text-align: center; margin-bottom: 30px; text-shadow: 2px 2px 4px rgba(0,0,0,0.1);">
                                🔊 Text-to-Speech (TTS)
                            </h2>
                            
                            <div style="background: rgba(255, 255, 255, 0.4); padding: 25px; border-radius: 15px; margin-bottom: 30px;">
                                <h3 style="margin-bottom: 15px;">🎵 What is Text-to-Speech?</h3>
                                <p style="line-height: 1.6;">
                                    Text-to-Speech technology converts written text into natural-sounding speech. Modern TTS systems use
                                    neural networks to create human-like voices that can express emotions, adjust pace, and sound remarkably natural.
                                </p>
                            </div>
                            
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 30px;">
                                <div style="background: rgba(255, 255, 255, 0.3); padding: 20px; border-radius: 10px;">
                                    <h4 style="margin-bottom: 10px;">🎯 Applications:</h4>
                                    <ul style="font-size: 0.9rem; line-height: 1.5;">
                                        <li>Audiobook creation</li>
                                        <li>Voice assistants</li>
                                        <li>Accessibility tools</li>
                                        <li>Language learning</li>
                                    </ul>
                                </div>
                                
                                <div style="background: rgba(255, 255, 255, 0.3); padding: 20px; border-radius: 10px;">
                                    <h4 style="margin-bottom: 10px;">⚡ Benefits:</h4>
                                    <ul style="font-size: 0.9rem; line-height: 1.5;">
                                        <li>Natural-sounding voices</li>
                                        <li>Customizable speech rate</li>
                                        <li>Multiple voice options</li>
                                        <li>Real-time generation</li>
                                    </ul>
                                </div>
                            </div>
                            
                            <div style="background: rgba(255, 255, 255, 0.4); padding: 20px; border-radius: 15px; text-align: center;">
                                <h3 style="margin-bottom: 10px;">📝 How to use:</h3>
                                <p style="font-size: 1rem; line-height: 1.5;">
                                    Type or paste any text in the input field, then click "Generate Speech".
                                    The AI will convert your text into natural-sounding audio that you can play and download!
                                </p>
                            </div>
                        </div>
                    </div>
                    """
                ]
                
                # Use page content if available, otherwise show default
                if step < len(page_contents):
                    content = page_contents[step]
                else:
                    content = f"""
                    <div style="text-align: center; padding: 50px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 70vh; color: white; border-radius: 20px;">
                        <h1>Step {step + 1}: Multi-GPU Optimized AI Demo</h1>
                        <p>Experience the power of 3x RTX 4090 load balancing</p>
                    </div>
                    """
            
            # Component visibility
            name_visible = step == 2
            llm_visible = step == 3
            vision_visible = step == 4
            whisper_visible = step == 5
            tts_visible = step == 6
            quiz_visible = step == 8
            certificate_visible = step == 9 and session.certificate_path
            completion_visible = step == 9
            
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
                gr.update(visible=certificate_visible, value=session.certificate_path),
                gr.update(visible=completion_visible),
                step
            )
        
        def go_next(step, name_value, session_id_val):
            """Go to next step with session support"""
            session = model_manager.get_or_create_session(session_id_val)
            
            if step == 2 and name_value and name_value.strip():
                session.name = name_value.strip()
                session.start_time = datetime.now()
            
            if step == 9:  # Start again
                session.current_step = 0
                session.quiz_answers = []
                session.quiz_score = 0
                session.current_question = 0
                session.certificate_path = None
                session.certificate_data = None
                return navigate_to_step(0, session_id_val) + (gr.update(), gr.update(), gr.update())
            
            if step == 7:  # Start quiz
                session.quiz_answers = []
                session.quiz_score = 0
                session.current_question = 0
                
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
                
                result = navigate_to_step(step + 1, session_id_val)
                return result + (
                    gr.update(value=question_html, visible=True),
                    gr.update(choices=question['options'], value=None, visible=True),
                    0
                )
            
            return navigate_to_step(step + 1, session_id_val) + (gr.update(), gr.update(), gr.update())
        
        def go_back(step, session_id_val):
            """Go to previous step"""
            return navigate_to_step(max(0, step - 1), session_id_val) + (gr.update(), gr.update(), gr.update())
        
        # Event handlers with session support
        next_btn.click(
            go_next,
            inputs=[current_step, name_input, session_id],
            outputs=[
                progress_display, back_btn, next_btn, main_content,
                name_input_group, llm_group, vision_group, whisper_group,
                tts_group, quiz_group, certificate_group, completion_display, current_step,
                quiz_question_display, quiz_options, quiz_question_idx
            ]
        )
        
        back_btn.click(
            go_back,
            inputs=[current_step, session_id],
            outputs=[
                progress_display, back_btn, next_btn, main_content,
                name_input_group, llm_group, vision_group, whisper_group,
                tts_group, quiz_group, certificate_group, completion_display, current_step,
                quiz_question_display, quiz_options, quiz_question_idx
            ]
        )
        
        # AI processing handlers with session support
        llm_send_btn.click(
            lambda msg, hist, sid: llm_chat(msg, hist, sid),
            inputs=[llm_input, llm_chatbot, session_id],
            outputs=[llm_chatbot, llm_input]
        )
        
        llm_input.submit(
            lambda msg, hist, sid: llm_chat(msg, hist, sid),
            inputs=[llm_input, llm_chatbot, session_id],
            outputs=[llm_chatbot, llm_input]
        )
        
        vision_analyze_btn.click(
            lambda img, q, sid: analyze_image(img, q, sid),
            inputs=[vision_image, vision_question, session_id],
            outputs=[vision_result]
        )
        
        whisper_transcribe_btn.click(
            lambda audio, sid: transcribe_audio(audio, sid),
            inputs=[whisper_audio, session_id],
            outputs=[whisper_result]
        )
        
        tts_speak_btn.click(
            lambda text, sid: text_to_speech(text, sid),
            inputs=[tts_input, session_id],
            outputs=[tts_status, tts_audio]
        )
        
        quiz_submit_btn.click(
            lambda opt, q_idx, sid: submit_quiz_answer(opt, q_idx, sid),
            inputs=[quiz_options, quiz_question_idx, session_id],
            outputs=[quiz_question_idx, quiz_question_display, quiz_options, quiz_feedback, certificate_image]
        )
        
        # Direct navigation handlers
        def go_to_step_direct(target_step, session_id_val):
            return navigate_to_step(target_step, session_id_val) + (gr.update(), gr.update(), gr.update())
        
        home_direct_btn.click(
            lambda sid: go_to_step_direct(0, sid),
            inputs=[session_id],
            outputs=[
                progress_display, back_btn, next_btn, main_content,
                name_input_group, llm_group, vision_group, whisper_group,
                tts_group, quiz_group, certificate_group, completion_display, current_step,
                quiz_question_display, quiz_options, quiz_question_idx
            ]
        )
        
        llm_direct_btn.click(
            lambda sid: go_to_step_direct(3, sid),
            inputs=[session_id],
            outputs=[
                progress_display, back_btn, next_btn, main_content,
                name_input_group, llm_group, vision_group, whisper_group,
                tts_group, quiz_group, certificate_group, completion_display, current_step,
                quiz_question_display, quiz_options, quiz_question_idx
            ]
        )
        
        vision_direct_btn.click(
            lambda sid: go_to_step_direct(4, sid),
            inputs=[session_id],
            outputs=[
                progress_display, back_btn, next_btn, main_content,
                name_input_group, llm_group, vision_group, whisper_group,
                tts_group, quiz_group, certificate_group, completion_display, current_step,
                quiz_question_display, quiz_options, quiz_question_idx
            ]
        )
        
        whisper_direct_btn.click(
            lambda sid: go_to_step_direct(5, sid),
            inputs=[session_id],
            outputs=[
                progress_display, back_btn, next_btn, main_content,
                name_input_group, llm_group, vision_group, whisper_group,
                tts_group, quiz_group, certificate_group, completion_display, current_step,
                quiz_question_display, quiz_options, quiz_question_idx
            ]
        )
        
        tts_direct_btn.click(
            lambda sid: go_to_step_direct(6, sid),
            inputs=[session_id],
            outputs=[
                progress_display, back_btn, next_btn, main_content,
                name_input_group, llm_group, vision_group, whisper_group,
                tts_group, quiz_group, certificate_group, completion_display, current_step,
                quiz_question_display, quiz_options, quiz_question_idx
            ]
        )
    
    return demo

def main():
    """Main function to initialize and launch the optimized tour"""
    print("🚀 Starting Multi-GPU AI Demo Tour - Optimized for 20+ Users")
    print("=" * 60)
    
    # Initialize models in background
    model_thread = threading.Thread(target=model_manager.initialize_models)
    model_thread.daemon = True
    model_thread.start()
    
    # Create and launch the interface
    demo = create_optimized_interactive_tour()
    
    # Wait for models to load
    print("⏳ Waiting for model initialization...")
    model_thread.join()
    
    # Find free port
    free_port = Config.find_free_port(Config.GRADIO_PORT)
    if free_port != Config.GRADIO_PORT:
        print(f"⚠️ Port {Config.GRADIO_PORT} is busy, using port {free_port}")
    
    # Get local IP for network access
    local_ip = Config.get_local_ip()
    
    print(f"🌐 Launching Multi-GPU optimized interface...")
    print(f"   Local:   http://localhost:{free_port}")
    print(f"   Network: http://{local_ip}:{free_port}")
    print(f"")
    print(f"🔥 Performance Features:")
    print(f"   ✅ 20+ concurrent users supported")
    print(f"   ✅ 3x RTX 4090 load balancing")
    print(f"   ✅ Async processing pipeline")
    print(f"   ✅ Real-time performance monitoring")
    print(f"   ✅ Intelligent session management")
    print(f"")
    print(f"📊 Model Distribution:")
    print(f"   GPU 0: Whisper (3) + TTS (2) instances")
    print(f"   GPU 1: VLM (3) instances")
    print(f"   GPU 2: LLM (4) + VLM (1) instances")
    print(f"   Total: 13 model instances for maximum throughput")
    
    demo.launch(
        server_name="0.0.0.0",  # Bind to all interfaces
        server_port=free_port,
        share=False,
        show_error=True,
        inbrowser=True,
        max_threads=40  # Support high concurrency
    )

if __name__ == "__main__":
    main()