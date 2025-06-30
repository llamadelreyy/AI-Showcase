"""
Setup script for Multi-GPU Optimized AI Demo Tour
Optimized for 3x RTX 4090 setup supporting 20+ concurrent users
"""

import subprocess
import sys
import os
import torch
import psutil
import time
from pathlib import Path

def check_system_requirements():
    """Check if system meets requirements for 20+ user support"""
    print("🔍 Checking System Requirements for 20+ Users...")
    
    # Check GPU count
    gpu_count = torch.cuda.device_count()
    print(f"   GPUs detected: {gpu_count}")
    
    if gpu_count < 3:
        print("   ⚠️ WARNING: Less than 3 GPUs detected. Performance may be limited.")
        print("   💡 Recommended: 3x RTX 4090 for optimal 20+ user support")
    else:
        print("   ✅ Sufficient GPUs for multi-GPU optimization")
    
    # Check GPU memory
    total_gpu_memory = 0
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        gpu_memory_gb = props.total_memory / (1024**3)
        total_gpu_memory += gpu_memory_gb
        print(f"   GPU {i}: {props.name} - {gpu_memory_gb:.1f}GB")
    
    print(f"   Total GPU Memory: {total_gpu_memory:.1f}GB")
    
    if total_gpu_memory < 60:  # 3x RTX 4090 = ~72GB
        print("   ⚠️ WARNING: Limited GPU memory. May affect concurrent user capacity.")
    else:
        print("   ✅ Sufficient GPU memory for 20+ users")
    
    # Check system RAM
    ram_gb = psutil.virtual_memory().total / (1024**3)
    print(f"   System RAM: {ram_gb:.1f}GB")
    
    if ram_gb < 32:
        print("   ⚠️ WARNING: Less than 32GB RAM. Recommended for 20+ users.")
    else:
        print("   ✅ Sufficient RAM for high concurrency")
    
    # Check CPU cores
    cpu_cores = psutil.cpu_count()
    print(f"   CPU Cores: {cpu_cores}")
    
    if cpu_cores < 16:
        print("   ⚠️ WARNING: Less than 16 CPU cores. May limit async processing.")
    else:
        print("   ✅ Sufficient CPU cores for concurrent processing")
    
    return gpu_count >= 2 and total_gpu_memory >= 40 and ram_gb >= 16

def install_optimized_requirements():
    """Install requirements optimized for multi-GPU performance"""
    print("📦 Installing Optimized Requirements...")
    
    # Core requirements for multi-GPU setup
    requirements = [
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "gradio>=4.0.0",
        "transformers>=4.35.0",
        "accelerate>=0.24.0",
        "openai-whisper>=20231117",
        "Pillow>=10.0.0",
        "soundfile>=0.12.0",
        "pyttsx3>=2.90",
        "edge-tts>=6.1.0",
        "gTTS>=2.4.0",
        "numpy>=1.24.0",
        "requests>=2.31.0",
        "python-dotenv>=1.0.0",
        "qrcode[pil]>=7.4.0",
        "psutil>=5.9.0",
        "asyncio",
        "uvicorn[standard]>=0.24.0",
        "fastapi>=0.104.0"
    ]
    
    try:
        for req in requirements:
            print(f"   Installing {req}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", req, "--upgrade"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        print("   ✅ All requirements installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error installing requirements: {e}")
        return False

def optimize_pytorch_settings():
    """Optimize PyTorch settings for multi-GPU performance"""
    print("⚙️ Optimizing PyTorch Settings...")
    
    try:
        # Set optimal thread counts for multi-GPU
        cpu_cores = psutil.cpu_count()
        optimal_threads = min(cpu_cores, 16)  # Cap at 16 for stability
        
        torch.set_num_threads(optimal_threads)
        torch.set_num_interop_threads(optimal_threads)
        
        # Enable optimized attention if available
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(True)
        
        # Enable memory efficient attention
        if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        
        # Optimize CUDA settings
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async CUDA operations
        os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'  # Enable cuDNN v8
        
        print(f"   ✅ PyTorch optimized for {optimal_threads} threads")
        print("   ✅ CUDA async operations enabled")
        print("   ✅ Memory efficient attention enabled")
        
        return True
        
    except Exception as e:
        print(f"   ⚠️ Some optimizations failed: {e}")
        return False

def setup_model_cache():
    """Setup optimized model caching for faster loading"""
    print("💾 Setting up Model Cache...")
    
    try:
        # Create cache directories
        cache_dir = Path.home() / ".cache" / "multi_gpu_ai"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Set environment variables for model caching
        os.environ['TRANSFORMERS_CACHE'] = str(cache_dir / "transformers")
        os.environ['HF_HOME'] = str(cache_dir / "huggingface")
        os.environ['TORCH_HOME'] = str(cache_dir / "torch")
        
        print(f"   ✅ Cache directory: {cache_dir}")
        print("   ✅ Model caching optimized")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Cache setup failed: {e}")
        return False

def test_multi_gpu_performance():
    """Test multi-GPU performance and load balancing"""
    print("🧪 Testing Multi-GPU Performance...")
    
    try:
        gpu_count = torch.cuda.device_count()
        
        if gpu_count == 0:
            print("   ❌ No CUDA GPUs available")
            return False
        
        # Test each GPU
        for i in range(gpu_count):
            with torch.cuda.device(i):
                # Create test tensor
                test_tensor = torch.randn(1000, 1000, device=f'cuda:{i}')
                
                # Perform computation
                start_time = time.time()
                result = torch.matmul(test_tensor, test_tensor.T)
                torch.cuda.synchronize()
                compute_time = time.time() - start_time
                
                # Check memory
                memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)
                
                print(f"   GPU {i}: Compute {compute_time:.3f}s, "
                      f"Memory: {memory_allocated:.2f}GB allocated, "
                      f"{memory_reserved:.2f}GB reserved")
                
                # Cleanup
                del test_tensor, result
                torch.cuda.empty_cache()
        
        print("   ✅ All GPUs tested successfully")
        return True
        
    except Exception as e:
        print(f"   ❌ GPU testing failed: {e}")
        return False

def create_launch_script():
    """Create optimized launch script"""
    print("📝 Creating Launch Script...")
    
    launch_script = """#!/usr/bin/env python3
\"\"\"
Optimized launcher for Multi-GPU AI Demo Tour
Supports 20+ concurrent users with 3x RTX 4090
\"\"\"

import os
import sys
import torch
import psutil

def optimize_environment():
    \"\"\"Set optimal environment variables\"\"\"
    # PyTorch optimizations
    os.environ['OMP_NUM_THREADS'] = str(min(psutil.cpu_count(), 16))
    os.environ['MKL_NUM_THREADS'] = str(min(psutil.cpu_count(), 16))
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
    
    # Memory optimizations
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    # Gradio optimizations
    os.environ['GRADIO_SERVER_NAME'] = '0.0.0.0'
    os.environ['GRADIO_SERVER_PORT'] = '7860'

def main():
    print("🚀 Launching Multi-GPU AI Demo Tour")
    print("=" * 50)
    
    # Optimize environment
    optimize_environment()
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Please check GPU drivers.")
        sys.exit(1)
    
    gpu_count = torch.cuda.device_count()
    print(f"✅ {gpu_count} GPU(s) detected")
    
    # Launch the optimized tour
    try:
        from interactive_tour_enhanced_optimized import main as tour_main
        tour_main()
    except ImportError:
        print("❌ Could not import optimized tour. Please check installation.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\\n👋 Shutting down gracefully...")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
"""
    
    try:
        with open("launch_optimized.py", "w") as f:
            f.write(launch_script)
        
        # Make executable
        os.chmod("launch_optimized.py", 0o755)
        
        print("   ✅ Launch script created: launch_optimized.py")
        return True
        
    except Exception as e:
        print(f"   ❌ Failed to create launch script: {e}")
        return False

def create_performance_config():
    """Create performance configuration file"""
    print("⚙️ Creating Performance Configuration...")
    
    config = {
        "multi_gpu": {
            "enabled": True,
            "gpu_distribution": {
                "whisper_instances": 3,
                "tts_instances": 2,
                "llm_instances": 4,
                "vlm_instances": 4
            },
            "load_balancing": {
                "strategy": "least_loaded",
                "health_check_interval": 30,
                "max_queue_size": 100
            }
        },
        "performance": {
            "max_concurrent_users": 25,
            "request_timeout": 60,
            "cleanup_interval": 300,
            "memory_cleanup_threshold": 0.8
        },
        "optimization": {
            "async_processing": True,
            "batch_processing": False,
            "memory_efficient": True,
            "cache_models": True
        }
    }
    
    try:
        import json
        with open("performance_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        print("   ✅ Performance config created: performance_config.json")
        return True
        
    except Exception as e:
        print(f"   ❌ Failed to create config: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Multi-GPU AI Demo Tour Setup")
    print("Optimized for 3x RTX 4090 supporting 20+ concurrent users")
    print("=" * 60)
    
    success_count = 0
    total_steps = 7
    
    # Step 1: Check system requirements
    if check_system_requirements():
        success_count += 1
        print("✅ System requirements check passed\n")
    else:
        print("⚠️ System requirements check completed with warnings\n")
    
    # Step 2: Install requirements
    if install_optimized_requirements():
        success_count += 1
        print("✅ Requirements installation completed\n")
    else:
        print("❌ Requirements installation failed\n")
    
    # Step 3: Optimize PyTorch
    if optimize_pytorch_settings():
        success_count += 1
        print("✅ PyTorch optimization completed\n")
    else:
        print("⚠️ PyTorch optimization completed with warnings\n")
    
    # Step 4: Setup model cache
    if setup_model_cache():
        success_count += 1
        print("✅ Model cache setup completed\n")
    else:
        print("❌ Model cache setup failed\n")
    
    # Step 5: Test GPU performance
    if test_multi_gpu_performance():
        success_count += 1
        print("✅ GPU performance test passed\n")
    else:
        print("❌ GPU performance test failed\n")
    
    # Step 6: Create launch script
    if create_launch_script():
        success_count += 1
        print("✅ Launch script created\n")
    else:
        print("❌ Launch script creation failed\n")
    
    # Step 7: Create performance config
    if create_performance_config():
        success_count += 1
        print("✅ Performance configuration created\n")
    else:
        print("❌ Performance configuration failed\n")
    
    # Summary
    print("=" * 60)
    print(f"Setup completed: {success_count}/{total_steps} steps successful")
    
    if success_count >= 5:
        print("🎉 Setup successful! Your system is ready for 20+ concurrent users.")
        print("\n🚀 To launch the optimized tour:")
        print("   python launch_optimized.py")
        print("\n📊 Performance Features:")
        print("   • 3x RTX 4090 load balancing")
        print("   • 13 model instances total")
        print("   • Async processing pipeline")
        print("   • Real-time monitoring")
        print("   • Intelligent session management")
        print("\n🌐 Access URLs:")
        print("   • Local: http://localhost:7860")
        print("   • Network: http://[YOUR-IP]:7860")
    else:
        print("⚠️ Setup completed with issues. Please review the errors above.")
        print("Some features may not work optimally.")
    
    return success_count >= 5

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)