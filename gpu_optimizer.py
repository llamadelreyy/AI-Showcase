"""
GPU Optimization utilities for 3x RTX 4090 setup
"""

import torch
import gc
from typing import Dict, List, Optional
import psutil

class GPUOptimizer:
    """Optimize model loading and inference across 3 RTX 4090 GPUs"""
    
    def __init__(self):
        self.gpu_count = torch.cuda.device_count()
        self.gpu_memory = self._get_gpu_memory()
        self.device_map = self._create_device_map()
        
    def _get_gpu_memory(self) -> List[int]:
        """Get available memory for each GPU in GB"""
        if not torch.cuda.is_available():
            return []
        
        memory_info = []
        for i in range(self.gpu_count):
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory / (1024**3)  # Convert to GB
            memory_info.append(int(total_memory))
        
        return memory_info
    
    def _create_device_map(self) -> Dict[str, int]:
        """Create optimal device mapping for 3-GPU setup"""
        if self.gpu_count <= 1:
            return "auto"
        
        # For 3x RTX 4090 setup, distribute models strategically
        device_map = {
            # GPU 0: Whisper + TTS (audio processing)
            "whisper": 0,
            "tts": 0,
            
            # GPU 1: VLM main processing (vision models)
            "vlm_main": 1,
            
            # GPU 2: VLM secondary + LLM (language processing)
            "vlm_secondary": 2,
            "llm": 2
        }
        
        return device_map
    
    def optimize_model_loading(self, model_type: str, model_size: str = "large") -> Dict:
        """Get optimized loading parameters for specific model types"""
        
        base_config = {
            "torch_dtype": torch.float16,
            "low_cpu_mem_usage": True,
            "device_map": "auto"
        }
        
        # Model-specific optimizations for 3-GPU setup
        if model_type == "whisper":
            if model_size == "large-v3":
                base_config.update({
                    "device": f"cuda:{self.device_map.get('whisper', 0)}",
                    "compute_type": "float16"
                })
        
        elif model_type == "vlm":
            if "34b" in model_size.lower() or "72b" in model_size.lower():
                # Large VLM models - use model parallelism across GPU 1 and 2
                base_config.update({
                    "device_map": {
                        "vision_tower": self.device_map.get('vlm_main', 1),
                        "multi_modal_projector": self.device_map.get('vlm_main', 1),
                        "language_model": self.device_map.get('vlm_secondary', 2)
                    },
                    "max_memory": {
                        1: "22GB",  # Leave some memory for other processes
                        2: "20GB"   # Leave more space for LLM on GPU 2
                    }
                })
            else:
                base_config.update({
                    "device_map": f"cuda:{self.device_map.get('vlm_main', 1)}"
                })
        
        elif model_type == "tts":
            base_config.update({
                "gpu": True,
                "device": f"cuda:{self.device_map.get('tts', 0)}"
            })
        
        elif model_type == "llm":
            base_config.update({
                "device_map": f"cuda:{self.device_map.get('llm', 2)}"
            })
        
        return base_config
    
    def clear_gpu_cache(self, device_id: Optional[int] = None):
        """Clear GPU cache for specific device or all devices"""
        if device_id is not None:
            with torch.cuda.device(device_id):
                torch.cuda.empty_cache()
        else:
            for i in range(self.gpu_count):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
    
    def get_gpu_utilization(self) -> Dict:
        """Get current GPU utilization stats"""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            utilization = {}
            
            for i, gpu in enumerate(gpus):
                utilization[f"GPU_{i}"] = {
                    "memory_used": f"{gpu.memoryUsed}MB",
                    "memory_total": f"{gpu.memoryTotal}MB",
                    "memory_percent": f"{gpu.memoryUtil * 100:.1f}%",
                    "gpu_load": f"{gpu.load * 100:.1f}%",
                    "temperature": f"{gpu.temperature}°C"
                }
            
            return utilization
            
        except ImportError:
            # Fallback if GPUtil not available
            utilization = {}
            for i in range(self.gpu_count):
                memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)
                
                utilization[f"GPU_{i}"] = {
                    "memory_allocated": f"{memory_allocated:.2f}GB",
                    "memory_reserved": f"{memory_reserved:.2f}GB"
                }
            
            return utilization
    
    def optimize_inference_settings(self, model_type: str) -> Dict:
        """Get optimized inference settings for different model types"""
        
        settings = {
            "whisper": {
                "fp16": True,
                "batch_size": 1,
                "beam_size": 1,
                "best_of": 1,
                "temperature": 0.0
            },
            
            "vlm": {
                "do_sample": False,
                "temperature": 0.1,
                "top_p": 0.9,
                "max_new_tokens": 200,
                "use_cache": True,
                "pad_token_id": None  # Will be set based on model
            },
            
            "tts": {
                "speed": 1.0,
                "emotion": "neutral",
                "style": "default"
            },
            
            "llm": {
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.1,
                "max_new_tokens": 512
            }
        }
        
        return settings.get(model_type, {})
    
    def get_optimal_batch_sizes(self) -> Dict[str, int]:
        """Get optimal batch sizes for each model type on 3-GPU setup"""
        return {
            "whisper": 1,  # Audio processing is typically sequential
            "tts": 1,      # TTS is typically one utterance at a time
            "vlm": 1,      # Large vision models work best with batch size 1
            "llm": 2       # Can handle small batches for text generation
        }

# Global optimizer instance
gpu_optimizer = GPUOptimizer()

def print_gpu_info():
    """Print detailed GPU information"""
    print("=== 3x RTX 4090 GPU Configuration ===")
    print(f"Available GPUs: {gpu_optimizer.gpu_count}")
    
    for i in range(gpu_optimizer.gpu_count):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name}")
        print(f"  Memory: {props.total_memory / (1024**3):.1f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Multiprocessors: {props.multi_processor_count}")
    
    print(f"\nOptimal Device Mapping:")
    print(f"  GPU 0: Whisper + TTS (Audio Processing)")
    print(f"  GPU 1: VLM Main (Vision Processing)")
    print(f"  GPU 2: VLM Secondary + LLM (Language Processing)")
    print("=" * 40)

if __name__ == "__main__":
    print_gpu_info()
    print("\nCurrent GPU Utilization:")
    utilization = gpu_optimizer.get_gpu_utilization()
    for gpu, stats in utilization.items():
        print(f"{gpu}: {stats}")