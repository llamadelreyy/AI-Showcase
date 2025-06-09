"""
Setup script for optimized AI models on 3x RTX 4090 setup
"""

import subprocess
import sys
import os
import torch
from gpu_optimizer import print_gpu_info

def install_requirements():
    """Install all required packages"""
    print("Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing requirements: {e}")
        return False
    return True

def verify_gpu_setup():
    """Verify GPU setup and CUDA availability"""
    print("\nVerifying GPU setup...")
    
    if not torch.cuda.is_available():
        print("✗ CUDA not available")
        return False
    
    gpu_count = torch.cuda.device_count()
    print(f"✓ CUDA available with {gpu_count} GPU(s)")
    
    if gpu_count < 3:
        print(f"⚠ Warning: Expected 3 GPUs, found {gpu_count}")
    
    # Print detailed GPU info
    print_gpu_info()
    
    return True

def download_models():
    """Download and cache the optimized models"""
    print("\nDownloading and caching models...")
    
    try:
        # Import models to trigger downloads
        from models import WhisperSTT, LocalTTS, LocalVLM
        
        print("Downloading Whisper large-v3...")
        whisper_model = WhisperSTT()
        print("✓ Whisper model ready")
        
        print("Downloading Coqui TTS...")
        tts_model = LocalTTS()
        print("✓ TTS model ready")
        
        print("Downloading LLaVA vision model...")
        vlm_model = LocalVLM()
        print("✓ VLM model ready")
        
        return True
        
    except Exception as e:
        print(f"✗ Error downloading models: {e}")
        return False

def run_performance_test():
    """Run a quick performance test"""
    print("\nRunning performance test...")
    
    try:
        from performance_monitor import PerformanceMonitor
        
        monitor = PerformanceMonitor()
        results = monitor.run_full_benchmark()
        
        if results:
            print("✓ Performance test completed successfully")
            return True
        else:
            print("✗ Performance test failed")
            return False
            
    except Exception as e:
        print(f"✗ Error running performance test: {e}")
        return False

def create_example_script():
    """Create an example usage script"""
    example_code = '''#!/usr/bin/env python3
"""
Example usage of optimized AI models
"""

from models import WhisperSTT, LocalTTS, LocalVLM
from gpu_optimizer import gpu_optimizer
import tempfile
from PIL import Image

def main():
    print("=== Optimized AI Models Demo ===")
    
    # Initialize models
    print("Initializing models...")
    whisper = WhisperSTT()
    tts = LocalTTS()
    vlm = LocalVLM()
    
    # Show GPU utilization
    print("\\nGPU Utilization:")
    util = gpu_optimizer.get_gpu_utilization()
    for gpu, stats in util.items():
        print(f"  {gpu}: {stats}")
    
    # Example TTS
    print("\\nTesting TTS...")
    audio_path = tts.speak("Hello! This is a test of the optimized text-to-speech system.")
    if audio_path:
        print(f"Audio generated: {audio_path}")
    
    # Example VLM with test image
    print("\\nTesting VLM...")
    test_image = Image.new('RGB', (224, 224), color='blue')
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        test_image.save(f.name)
        description = vlm.analyze_image(f.name, "What color is this image?")
        print(f"VLM response: {description}")
    
    print("\\n=== Demo Complete ===")

if __name__ == "__main__":
    main()
'''
    
    with open("example_usage.py", "w") as f:
        f.write(example_code)
    
    print("✓ Example script created: example_usage.py")

def main():
    """Main setup function"""
    print("=== Setting up Optimized AI Models for 3x RTX 4090 ===")
    print("This will install requirements and download optimized models")
    print()
    
    # Step 1: Verify GPU setup
    if not verify_gpu_setup():
        print("Please ensure CUDA is properly installed and GPUs are available")
        return False
    
    # Step 2: Install requirements
    if not install_requirements():
        print("Please fix installation issues before continuing")
        return False
    
    # Step 3: Download models
    if not download_models():
        print("Model download failed. Check your internet connection and try again")
        return False
    
    # Step 4: Run performance test
    if not run_performance_test():
        print("Performance test failed, but models should still work")
    
    # Step 5: Create example script
    create_example_script()
    
    print("\\n=== Setup Complete! ===")
    print("Your optimized AI models are ready to use.")
    print()
    print("Model Configuration:")
    print("- Whisper: large-v3 (best accuracy)")
    print("- TTS: Coqui TTS with high-quality vocoder")
    print("- VLM: LLaVA-1.6-34B (advanced vision understanding)")
    print()
    print("GPU Distribution:")
    print("- GPU 0: Whisper + TTS")
    print("- GPU 1: VLM Main")
    print("- GPU 2: VLM Secondary + LLM")
    print()
    print("Next steps:")
    print("1. Run 'python example_usage.py' to test the models")
    print("2. Run 'python performance_monitor.py' for detailed benchmarks")
    print("3. Use the models in your own applications")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)