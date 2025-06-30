# 🚀 Multi-GPU AI Demo Tour - Optimized for 20+ Concurrent Users

This enhanced version of the AI Demo Tour is specifically optimized for high-concurrency scenarios using multiple RTX 4090 GPUs.

## 🔥 Quick Start

Simply run the enhanced quick start script:

```bash
chmod +x quick_start.sh
./quick_start.sh
```

The script will automatically:
- ✅ Detect your GPU configuration
- ✅ Install optimized packages for multi-GPU setup
- ✅ Configure environment variables for maximum performance
- ✅ Launch the appropriate version based on your hardware

## 🏗️ Architecture

### GPU Distribution (3x RTX 4090)
```
GPU 0: Audio Processing
├── Whisper STT: 3 instances
└── TTS: 2 instances

GPU 1: Vision Processing  
└── VLM: 3 instances

GPU 2: Language Processing
├── LLM: 4 instances
└── VLM: 1 instance (secondary)

Total: 13 model instances for maximum throughput
```

## ⚡ Performance Features

- **20+ Concurrent Users**: Optimized load balancing
- **Async Processing**: Non-blocking request handling
- **Real-time Monitoring**: Performance metrics display
- **Session Management**: Intelligent user session handling
- **Auto-scaling**: Dynamic resource allocation

## 📊 Expected Performance

| Users | Response Time | GPU Utilization |
|-------|---------------|-----------------|
| 5     | <1 second     | 20-30%         |
| 10    | 1-2 seconds   | 40-50%         |
| 15    | 2-3 seconds   | 60-70%         |
| 20    | 2-4 seconds   | 70-80%         |
| 25+   | 3-5 seconds   | 80-90%         |

## 🛠️ System Requirements

### Minimum (10-15 users)
- 2x RTX 4090 (or equivalent)
- 32GB RAM
- 16+ CPU cores

### Recommended (20+ users)
- 3x RTX 4090
- 64GB RAM  
- 24+ CPU cores
- NVMe SSD storage

## 🚀 Launch Options

When you run `./quick_start.sh`, you'll see different options based on your hardware:

### With 3+ GPUs:
1. **Multi-GPU Interactive Tour** (🔥 RECOMMENDED)
   - Optimized for 20+ concurrent users
   - Real-time performance monitoring
   - Advanced load balancing

2. **Standard Interactive Tour**
   - Single-user optimized experience

3. **Original AI Demo Stack**
   - Free-play mode

### With 1-2 GPUs:
1. **Interactive AI Demo Tour**
   - Standard guided experience

2. **Original AI Demo Stack**
   - Free-play mode

## 🔍 Monitoring

The multi-GPU version includes real-time performance monitoring:
- Active user count
- Average response times
- GPU utilization per device
- Request success rates

## 🎯 Key Files

- `interactive_tour_enhanced_optimized.py` - Multi-GPU optimized tour
- `setup_multi_gpu_optimized.py` - Advanced setup script
- `quick_start.sh` - Enhanced quick start with GPU detection
- `models.py` - Core AI model implementations
- `gpu_optimizer.py` - GPU optimization utilities

## 🔧 Manual Setup (Advanced)

If you prefer manual setup:

```bash
# 1. Install optimized requirements
pip install -r requirements_optimized.txt

# 2. Install multi-GPU packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install accelerate psutil asyncio

# 3. Set environment variables
export OMP_NUM_THREADS=16
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# 4. Launch optimized tour
python3 interactive_tour_enhanced_optimized.py
```

## 🆘 Troubleshooting

### GPU Not Detected
```bash
# Check GPU status
nvidia-smi

# Verify CUDA installation
python3 -c "import torch; print(torch.cuda.is_available())"
```

### Memory Issues
```bash
# Clear GPU cache
python3 -c "import torch; torch.cuda.empty_cache()"

# Monitor GPU memory
watch -n 1 nvidia-smi
```

### Performance Issues
- Ensure adequate cooling for GPUs
- Check network bandwidth for remote users
- Monitor CPU usage with `htop`
- Verify sufficient RAM availability

## 📈 Scaling Tips

1. **Network**: Use gigabit+ ethernet for 20+ users
2. **Storage**: NVMe SSD recommended for model loading
3. **Cooling**: Ensure adequate GPU cooling under load
4. **Memory**: 64GB+ RAM for optimal performance

## 🎉 Success Metrics

With proper setup, you should achieve:
- ✅ 20+ concurrent users supported
- ✅ <3 second average response times
- ✅ 99%+ request success rate
- ✅ Stable performance under load

---

**Ready to handle 20+ users with blazing-fast AI responses!** 🚀