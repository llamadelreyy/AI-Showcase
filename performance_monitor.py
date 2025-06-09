"""
Performance monitoring for optimized AI models on 3x RTX 4090 setup
"""

import time
import torch
import psutil
import threading
from datetime import datetime
from gpu_optimizer import gpu_optimizer
from models import WhisperSTT, LocalTTS, LocalVLM
import json

class PerformanceMonitor:
    """Monitor performance metrics for AI models"""
    
    def __init__(self):
        self.metrics = {
            "whisper": [],
            "tts": [],
            "vlm": [],
            "system": []
        }
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self, interval=5):
        """Start continuous performance monitoring"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print(f"Performance monitoring started (interval: {interval}s)")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        print("Performance monitoring stopped")
    
    def _monitor_loop(self, interval):
        """Continuous monitoring loop"""
        while self.monitoring:
            timestamp = datetime.now().isoformat()
            
            # System metrics
            system_metrics = {
                "timestamp": timestamp,
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "memory_used_gb": psutil.virtual_memory().used / (1024**3)
            }
            
            # GPU metrics
            gpu_util = gpu_optimizer.get_gpu_utilization()
            system_metrics["gpu_utilization"] = gpu_util
            
            self.metrics["system"].append(system_metrics)
            
            time.sleep(interval)
    
    def benchmark_whisper(self, audio_file_path=None):
        """Benchmark Whisper performance"""
        print("Benchmarking Whisper STT...")
        
        try:
            # Initialize model
            start_time = time.time()
            whisper_model = WhisperSTT()
            init_time = time.time() - start_time
            
            # Create test audio if none provided
            if not audio_file_path:
                # Generate 10 seconds of test audio
                import numpy as np
                import soundfile as sf
                import tempfile
                
                sample_rate = 16000
                duration = 10
                test_audio = np.random.randn(sample_rate * duration).astype(np.float32) * 0.1
                
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    sf.write(f.name, test_audio, sample_rate)
                    audio_file_path = f.name
            
            # Benchmark transcription
            transcription_times = []
            for i in range(3):
                start_time = time.time()
                result = whisper_model.transcribe(audio_file_path)
                transcription_time = time.time() - start_time
                transcription_times.append(transcription_time)
                print(f"Transcription {i+1}: {transcription_time:.2f}s")
            
            metrics = {
                "model": "whisper",
                "initialization_time": init_time,
                "avg_transcription_time": sum(transcription_times) / len(transcription_times),
                "min_transcription_time": min(transcription_times),
                "max_transcription_time": max(transcription_times),
                "throughput_factor": 10 / (sum(transcription_times) / len(transcription_times))  # 10s audio
            }
            
            self.metrics["whisper"].append(metrics)
            return metrics
            
        except Exception as e:
            print(f"Whisper benchmark error: {e}")
            return None
    
    def benchmark_tts(self, test_text="This is a test of the text-to-speech system performance."):
        """Benchmark TTS performance"""
        print("Benchmarking TTS...")
        
        try:
            # Initialize model
            start_time = time.time()
            tts_model = LocalTTS()
            init_time = time.time() - start_time
            
            # Benchmark synthesis
            synthesis_times = []
            for i in range(3):
                start_time = time.time()
                audio_path = tts_model.speak(test_text)
                synthesis_time = time.time() - start_time
                synthesis_times.append(synthesis_time)
                print(f"Synthesis {i+1}: {synthesis_time:.2f}s")
                
                # Clean up temp file
                if audio_path:
                    import os
                    try:
                        os.unlink(audio_path)
                    except:
                        pass
            
            metrics = {
                "model": "tts",
                "initialization_time": init_time,
                "avg_synthesis_time": sum(synthesis_times) / len(synthesis_times),
                "min_synthesis_time": min(synthesis_times),
                "max_synthesis_time": max(synthesis_times),
                "text_length": len(test_text),
                "chars_per_second": len(test_text) / (sum(synthesis_times) / len(synthesis_times))
            }
            
            self.metrics["tts"].append(metrics)
            return metrics
            
        except Exception as e:
            print(f"TTS benchmark error: {e}")
            return None
    
    def benchmark_vlm(self, image_path=None):
        """Benchmark VLM performance"""
        print("Benchmarking VLM...")
        
        try:
            # Initialize model
            start_time = time.time()
            vlm_model = LocalVLM()
            init_time = time.time() - start_time
            
            # Create test image if none provided
            if not image_path:
                from PIL import Image
                import tempfile
                
                # Create a test image
                test_image = Image.new('RGB', (224, 224), color='red')
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
                    test_image.save(f.name)
                    image_path = f.name
            
            # Benchmark analysis
            analysis_times = []
            questions = [
                "What is in this image?",
                "Describe the colors in this image.",
                "What objects can you see?"
            ]
            
            for question in questions:
                start_time = time.time()
                result = vlm_model.analyze_image(image_path, question)
                analysis_time = time.time() - start_time
                analysis_times.append(analysis_time)
                print(f"Analysis '{question}': {analysis_time:.2f}s")
            
            metrics = {
                "model": "vlm",
                "initialization_time": init_time,
                "avg_analysis_time": sum(analysis_times) / len(analysis_times),
                "min_analysis_time": min(analysis_times),
                "max_analysis_time": max(analysis_times),
                "questions_tested": len(questions)
            }
            
            self.metrics["vlm"].append(metrics)
            return metrics
            
        except Exception as e:
            print(f"VLM benchmark error: {e}")
            return None
    
    def run_full_benchmark(self):
        """Run comprehensive benchmark of all models"""
        print("=== Starting Full Performance Benchmark ===")
        print(f"Hardware: 3x RTX 4090, {psutil.virtual_memory().total / (1024**3):.0f}GB RAM")
        print()
        
        # Clear GPU cache before starting
        gpu_optimizer.clear_gpu_cache()
        
        results = {}
        
        # Benchmark each model
        results["whisper"] = self.benchmark_whisper()
        print()
        
        results["tts"] = self.benchmark_tts()
        print()
        
        results["vlm"] = self.benchmark_vlm()
        print()
        
        # System summary
        gpu_util = gpu_optimizer.get_gpu_utilization()
        results["system_summary"] = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "gpu_utilization": gpu_util
        }
        
        print("=== Benchmark Complete ===")
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results):
        """Print benchmark summary"""
        print("\n=== Performance Summary ===")
        
        if results.get("whisper"):
            w = results["whisper"]
            print(f"Whisper STT:")
            print(f"  Initialization: {w['initialization_time']:.2f}s")
            print(f"  Avg Transcription: {w['avg_transcription_time']:.2f}s")
            print(f"  Real-time Factor: {w['throughput_factor']:.2f}x")
        
        if results.get("tts"):
            t = results["tts"]
            print(f"TTS:")
            print(f"  Initialization: {t['initialization_time']:.2f}s")
            print(f"  Avg Synthesis: {t['avg_synthesis_time']:.2f}s")
            print(f"  Speed: {t['chars_per_second']:.1f} chars/sec")
        
        if results.get("vlm"):
            v = results["vlm"]
            print(f"VLM:")
            print(f"  Initialization: {v['initialization_time']:.2f}s")
            print(f"  Avg Analysis: {v['avg_analysis_time']:.2f}s")
        
        print("\nGPU Utilization:")
        for gpu, stats in results["system_summary"]["gpu_utilization"].items():
            print(f"  {gpu}: {stats}")
    
    def save_metrics(self, filename="performance_metrics.json"):
        """Save metrics to file"""
        with open(filename, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Metrics saved to {filename}")
    
    def load_metrics(self, filename="performance_metrics.json"):
        """Load metrics from file"""
        try:
            with open(filename, 'r') as f:
                self.metrics = json.load(f)
            print(f"Metrics loaded from {filename}")
        except FileNotFoundError:
            print(f"Metrics file {filename} not found")

if __name__ == "__main__":
    monitor = PerformanceMonitor()
    
    # Run full benchmark
    results = monitor.run_full_benchmark()
    
    # Save results
    monitor.save_metrics()
    
    # Optional: Start continuous monitoring
    # monitor.start_monitoring(interval=10)
    # time.sleep(60)  # Monitor for 1 minute
    # monitor.stop_monitoring()