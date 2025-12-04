#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice Chat Server với Bank Model
Server realtime để chat bằng giọng nói với model bank-model-2b.
Tính năng:
- 🎙️ Nhận giọng nói (Speech-to-Text)
- 🤖 Xử lý với model Qwen3-VL-2B
- 🔊 Trả lời bằng giọng nói (Text-to-Speech) realtime
- 📊 Hiển thị tài nguyên hệ thống (CPU, RAM, GPU) realtime
"""

import torch
import whisper
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    TextIteratorStreamer
)
import gradio as gr
import io
import base64
import os
import sys
import subprocess
import time
import re
import tempfile
import traceback
from typing import Optional, Tuple
from functools import lru_cache
from threading import Thread
import queue
import psutil

# Import TTS với fallback
TTS_AVAILABLE = False
pyttsx3_available = False
tts = None
tts_type = None

try:
    from TTS.api import TTS
    TTS_AVAILABLE = True
    print("✅ Coqui TTS đã sẵn sàng!")
except ImportError:
    print("⚠️  Coqui TTS không khả dụng, sẽ thử pyttsx3...")
    try:
        import pyttsx3
        pyttsx3_available = True
        print("✅ pyttsx3 đã sẵn sàng!")
    except ImportError:
        print("⚠️  Cả TTS và pyttsx3 đều không khả dụng. Cần cài một trong hai.")

# Global variables
model = None
processor = None
whisper_model = None


def install_tts():
    """Cài đặt TTS với nhiều fallback options"""
    global TTS_AVAILABLE, TTS_INSTALLED
    
    print("="*60)
    print("🔊 Đang cài TTS (Text-to-Speech) cho AI Voice Reply...")
    print("="*60)
    
    TTS_INSTALLED = False
    TTS_ERRORS = []
    
    # Kiểm tra xem TTS đã được cài chưa
    try:
        from TTS.api import TTS
        TTS_INSTALLED = True
        TTS_AVAILABLE = True
        print("✅ TTS đã được cài sẵn!")
        return True
    except ImportError:
        print("⚠️  TTS chưa được cài, đang thử cài...")
        
        # Option 1: Thử cài TTS từ PyPI (version cụ thể - ổn định hơn)
        print("\n📦 Option 1: Cài TTS từ PyPI (version ổn định)...")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "TTS==0.22.0"], 
                capture_output=True, 
                text=True,
                timeout=300
            )
            if result.returncode == 0:
                try:
                    from TTS.api import TTS
                    TTS_INSTALLED = True
                    TTS_AVAILABLE = True
                    print("✅ TTS đã được cài thành công từ PyPI!")
                    return True
                except ImportError:
                    TTS_ERRORS.append("TTS cài nhưng không import được")
        except subprocess.TimeoutExpired:
            TTS_ERRORS.append("PyPI install timeout")
        except Exception as e:
            TTS_ERRORS.append(f"PyPI install error: {str(e)[:200]}")
        
        # Option 2: Thử cài TTS từ PyPI (latest)
        if not TTS_INSTALLED:
            print("\n📦 Option 2: Cài TTS từ PyPI (latest version)...")
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-q", "TTS", "--no-deps"], 
                    capture_output=True, 
                    text=True,
                    timeout=300
                )
                if result.returncode == 0:
                    # Cài dependencies riêng
                    deps = ["numpy", "scipy", "librosa", "soundfile", "torch", "torchaudio"]
                    for dep in deps:
                        subprocess.run(
                            [sys.executable, "-m", "pip", "install", "-q", dep],
                            capture_output=True,
                            timeout=60
                        )
                    try:
                        from TTS.api import TTS
                        TTS_INSTALLED = True
                        TTS_AVAILABLE = True
                        print("✅ TTS đã được cài thành công từ PyPI (latest)!")
                        return True
                    except ImportError:
                        TTS_ERRORS.append("TTS cài nhưng không import được sau khi cài deps")
            except Exception as e:
                TTS_ERRORS.append(f"PyPI latest install error: {str(e)[:200]}")
        
        # Option 3: Thử cài từ GitHub (source)
        if not TTS_INSTALLED:
            print("\n📦 Option 3: Cài TTS từ GitHub source...")
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-q", 
                     "git+https://github.com/coqui-ai/TTS.git@v0.22.0"], 
                    capture_output=True,
                    text=True,
                    timeout=600
                )
                if result.returncode == 0:
                    try:
                        from TTS.api import TTS
                        TTS_INSTALLED = True
                        TTS_AVAILABLE = True
                        print("✅ TTS đã được cài thành công từ GitHub!")
                        return True
                    except ImportError:
                        TTS_ERRORS.append("TTS cài từ GitHub nhưng không import được")
            except subprocess.TimeoutExpired:
                TTS_ERRORS.append("GitHub install timeout")
            except Exception as e:
                TTS_ERRORS.append(f"GitHub install error: {str(e)[:200]}")
        
        # Option 4: Fallback - pyttsx3 (nhẹ hơn, không cần model lớn)
        if not TTS_INSTALLED:
            print("\n📦 Option 4: Cài pyttsx3 làm fallback (nhẹ hơn)...")
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-q", "pyttsx3"], 
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                if result.returncode == 0:
                    try:
                        import pyttsx3
                        print("✅ pyttsx3 đã được cài (fallback TTS)")
                        print("   ⚠️  Lưu ý: pyttsx3 có thể cần eSpeak trên Linux")
                        return True
                    except ImportError:
                        TTS_ERRORS.append("pyttsx3 cài nhưng không import được")
            except Exception as e:
                TTS_ERRORS.append(f"pyttsx3 install error: {str(e)[:200]}")
    
    # Tóm tắt kết quả
    print("\n" + "="*60)
    if TTS_INSTALLED:
        print("✅ TTS đã sẵn sàng để sử dụng!")
        print("   AI sẽ có thể trả lời bằng giọng nói")
        return True
    else:
        print("⚠️  TTS chưa được cài thành công")
        print("   Server vẫn hoạt động nhưng AI sẽ không trả lời bằng giọng nói")
        print("   (chỉ hiển thị text)")
        if TTS_ERRORS:
            print("\n   Các lỗi đã gặp:")
            for i, error in enumerate(TTS_ERRORS, 1):
                print(f"   {i}. {error}")
        print("\n   💡 Có thể thử cài thủ công:")
        print("      pip install TTS")
        print("      hoặc")
        print("      pip install git+https://github.com/coqui-ai/TTS.git")
        print("="*60)
        return False


def get_system_resources():
    """
    Lấy thông tin tài nguyên hệ thống (CPU, RAM, GPU)
    """
    # CPU
    cpu_percent = psutil.cpu_percent(interval=0.1)
    cpu_count = psutil.cpu_count()
    
    # RAM
    memory = psutil.virtual_memory()
    ram_total_gb = memory.total / (1024**3)
    ram_used_gb = memory.used / (1024**3)
    ram_available_gb = memory.available / (1024**3)
    ram_percent = memory.percent
    
    # GPU (nếu có)
    gpu_info = ""
    gpu_memory_used = 0
    gpu_memory_total = 0
    gpu_memory_percent = 0
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_info = f"GPU: {torch.cuda.get_device_name(0)}"
        gpu_memory_used = torch.cuda.memory_allocated(0) / (1024**3)
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_memory_percent = (gpu_memory_used / gpu_memory_total) * 100
    else:
        gpu_info = "GPU: Không có"
    
    return {
        "cpu_percent": cpu_percent,
        "cpu_count": cpu_count,
        "ram_total_gb": ram_total_gb,
        "ram_used_gb": ram_used_gb,
        "ram_available_gb": ram_available_gb,
        "ram_percent": ram_percent,
        "gpu_info": gpu_info,
        "gpu_memory_used": gpu_memory_used,
        "gpu_memory_total": gpu_memory_total,
        "gpu_memory_percent": gpu_memory_percent,
    }


def format_resources_info():
    """
    Format thông tin tài nguyên thành string để hiển thị
    """
    res = get_system_resources()
    
    info = f"""
### 📊 Tài nguyên hệ thống:

**CPU:**
- Sử dụng: {res['cpu_percent']:.1f}% / {res['cpu_count']} cores
- Còn lại: {100 - res['cpu_percent']:.1f}%

**RAM:**
- Tổng: {res['ram_total_gb']:.2f} GB
- Đã dùng: {res['ram_used_gb']:.2f} GB ({res['ram_percent']:.1f}%)
- Còn lại: {res['ram_available_gb']:.2f} GB ({100 - res['ram_percent']:.1f}%)

**{res['gpu_info']}**
"""
    
    if torch.cuda.is_available():
        info += f"""
- Tổng: {res['gpu_memory_total']:.2f} GB
- Đã dùng: {res['gpu_memory_used']:.2f} GB ({res['gpu_memory_percent']:.1f}%)
- Còn lại: {res['gpu_memory_total'] - res['gpu_memory_used']:.2f} GB ({100 - res['gpu_memory_percent']:.1f}%)
"""
    
    return info


def load_models(model_name: str = "hainguyen306201/bank-model-2b", install_tts_on_load: bool = False):
    """
    Load Whisper model và Bank Model
    """
    global model, processor, whisper_model, tts, tts_type, TTS_AVAILABLE
    
    # Load Whisper model cho Speech-to-Text
    print("Đang load Whisper model (tiny - nhanh nhất)...")
    if torch.cuda.is_available():
        print(f"🚀 GPU có sẵn: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        whisper_model = whisper.load_model("tiny", device="cuda")
        print("✅ Whisper model đã load trên GPU!")
    else:
        whisper_model = whisper.load_model("tiny", device="cpu")
        print("⚠️  Whisper model đã load trên CPU (không có GPU)")
    
    # Load Bank Model từ Hugging Face
    print("\n" + "="*50)
    print("Đang tải Bank Model từ Hugging Face...")
    print("="*50)
    
    print(f"Model: {model_name}")
    
    # Kiểm tra xem model có tồn tại không
    try:
        from huggingface_hub import model_info
        info = model_info(model_name)
        print(f"✅ Model tìm thấy trên Hugging Face!")
        print(f"   - Model ID: {info.modelId}")
        print(f"   - Files: {len(info.siblings)} files")
    except Exception as e:
        print(f"⚠️  Không thể kiểm tra model info: {e}")
        print("   Tiếp tục tải model...")
    
    # Cấu hình quantization 4-bit để tiết kiệm memory và tăng tốc
    print("\nĐang cấu hình quantization (4-bit)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load model với các tùy chọn để đảm bảo tải đúng và chạy trên GPU
    print("\nĐang tải model (có thể mất vài phút lần đầu)...")
    
    # Kiểm tra GPU và quyết định quantization
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {gpu_memory_gb:.2f} GB")
        
        # Nếu GPU >= 20GB, có thể load không quantization để nhanh hơn
        if gpu_memory_gb >= 20:
            print("✅ GPU đủ lớn, sẽ load model không quantization (full precision) để tối ưu tốc độ!")
            use_quantization = False
        else:
            print("⚠️  GPU nhỏ, sẽ dùng quantization 4-bit để tiết kiệm memory")
            use_quantization = True
    else:
        print("⚠️  Không có GPU, sẽ dùng quantization 4-bit")
        use_quantization = True
    
    try:
        if use_quantization:
            # Dùng quantization cho GPU nhỏ hoặc CPU
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                resume_download=True,
                force_download=False,
            )
        else:
            # Load full precision trên GPU lớn (nhanh hơn)
            # Kiểm tra xem có flash-attn không
            try:
                import flash_attn
                use_flash_attention = True
                print("✅ Flash Attention 2 được phát hiện, sẽ sử dụng để tối ưu tốc độ")
            except ImportError:
                use_flash_attention = False
                print("⚠️  Flash Attention 2 chưa được cài, sử dụng attention mặc định")
                print("   Có thể cài: pip install flash-attn (tùy chọn, để tăng tốc)")
            
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                resume_download=True,
                force_download=False,
                attn_implementation="flash_attention_2" if use_flash_attention else "sdpa",
            )
        
        # Đảm bảo model trên GPU và tối ưu
        if torch.cuda.is_available():
            model_device = next(model.parameters()).device
            print(f"✅ Model đã được tải và load thành công!")
            print(f"   Model device: {model_device}")
            if model_device.type == "cuda":
                print(f"   ✅ Model đang chạy trên GPU: {torch.cuda.get_device_name(model_device.index)}")
                # Tối ưu: Compile model nếu PyTorch >= 2.0
                try:
                    if hasattr(torch, 'compile') and torch.__version__ >= "2.0.0":
                        print("   🔧 Đang compile model để tăng tốc (PyTorch 2.0+)...")
                        model = torch.compile(model, mode="reduce-overhead")
                        print("   ✅ Model đã được compile!")
                except Exception as e:
                    print(f"   ⚠️  Không thể compile model: {e} (không ảnh hưởng chức năng)")
            else:
                print(f"   ⚠️  Model đang chạy trên {model_device.type}, đang chuyển lên GPU...")
                model = model.to("cuda")
                try:
                    if hasattr(torch, 'compile') and torch.__version__ >= "2.0.0":
                        model = torch.compile(model, mode="reduce-overhead")
                        print("   ✅ Model đã được compile!")
                except:
                    pass
        else:
            print("✅ Model đã được tải và load thành công trên CPU!")
            
    except Exception as e:
        print(f"❌ Lỗi khi tải model: {e}")
        print("\nThử tải lại với force_download=True...")
        if use_quantization:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                force_download=True,
            )
        else:
            try:
                import flash_attn
                use_flash_attention = True
            except ImportError:
                use_flash_attention = False
            
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                force_download=True,
                attn_implementation="flash_attention_2" if use_flash_attention else "sdpa",
            )
        print("✅ Model đã được tải lại thành công!")
        
        # Đảm bảo model trên GPU
        if torch.cuda.is_available():
            model = model.to("cuda")
            print(f"✅ Model đã được chuyển lên GPU!")
    
    # Load processor
    print("\nĐang tải processor...")
    processor = AutoProcessor.from_pretrained(
        model_name, 
        trust_remote_code=True,
        resume_download=True,
    )
    print("✅ Processor đã load!")
    
    # Kiểm tra processor có đúng không
    if not hasattr(processor, 'apply_chat_template'):
        print("⚠️  Processor không có apply_chat_template, có thể có vấn đề")
    else:
        print("✅ Processor có apply_chat_template - OK")
    
    # Kiểm tra tokenizer
    if hasattr(processor, 'tokenizer'):
        print(f"✅ Tokenizer: {type(processor.tokenizer).__name__}")
        if hasattr(processor.tokenizer, 'eos_token_id') and processor.tokenizer.eos_token_id:
            print(f"   EOS token ID: {processor.tokenizer.eos_token_id}")
        else:
            print("   ⚠️  EOS token ID không được set, sẽ dùng pad_token_id")
    
    # Kiểm tra và hiển thị thông tin GPU
    print("\n" + "="*50)
    print("📊 Thông tin GPU và Model:")
    print("="*50)
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        
        # Kiểm tra model đã được load chưa
        try:
            if model is not None:
                model_device = next(model.parameters()).device
                if model_device.type == "cuda":
                    print(f"   ✅ Model đang chạy trên GPU: {torch.cuda.get_device_name(model_device.index)}")
                    print(f"   GPU Memory đã dùng: {torch.cuda.memory_allocated(0) / (1024**3):.2f} GB")
                else:
                    print(f"   ⚠️  Model đang chạy trên {model_device.type}")
        except Exception as e:
            print(f"   ⚠️  Không thể kiểm tra model device: {e}")
        
        # Kiểm tra Whisper model
        try:
            if whisper_model is not None:
                if hasattr(whisper_model, 'encoder') and hasattr(whisper_model.encoder, 'parameters'):
                    whisper_device = next(whisper_model.encoder.parameters()).device
                    if whisper_device.type == "cuda":
                        print(f"   ✅ Whisper đang chạy trên GPU")
                    else:
                        print(f"   ⚠️  Whisper đang chạy trên CPU")
                else:
                    print(f"   ✅ Whisper đã được load với device phù hợp")
        except Exception as e:
            print(f"   ℹ️  Không thể kiểm tra Whisper device: {e}")
    else:
        print("⚠️  Không có GPU, tất cả đang chạy trên CPU")
    
    print("\n" + "="*50)
    print("✅ Bank Model đã sẵn sàng để sử dụng!")
    print("="*50)
    
    # Tối ưu: Warmup model để tăng tốc lần đầu generate
    print("\n🔥 Đang warmup model (lần đầu có thể chậm)...")
    try:
        warmup_messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
        warmup_inputs = processor.apply_chat_template(
            warmup_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        if torch.cuda.is_available():
            warmup_inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in warmup_inputs.items()}
        
        model.eval()
        with torch.inference_mode():
            _ = model.generate(
                **warmup_inputs,
                max_new_tokens=10,
                do_sample=False,
                use_cache=True,
            )
        print("✅ Model đã được warmup - sẵn sàng generate nhanh!")
    except Exception as e:
        print(f"⚠️  Warmup không thành công (không ảnh hưởng chức năng): {e}")
    
    # Load TTS offline
    print("\n" + "="*50)
    print("Đang load TTS model (offline)...")
    print("="*50)
    
    # Đảm bảo TTS chạy trên GPU nếu có
    use_gpu_tts = torch.cuda.is_available()
    if use_gpu_tts:
        print(f"🚀 TTS sẽ chạy trên GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  TTS sẽ chạy trên CPU (không có GPU)")
    
    # Kiểm tra lại TTS
    if install_tts_on_load:
        install_tts()
    
    if not TTS_AVAILABLE:
        print("⚠️  Coqui TTS chưa được cài đặt, đang thử cài lại...")
        try:
            print("Thử cài TTS==0.22.0...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "TTS==0.22.0"],
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode == 0:
                print("✅ Đã cài TTS từ PyPI, đang import lại...")
                try:
                    from TTS.api import TTS
                    TTS_AVAILABLE = True
                    print("✅ Coqui TTS đã sẵn sàng!")
                except ImportError:
                    print("⚠️  Vẫn không thể import TTS sau khi cài")
        except Exception as e:
            print(f"⚠️  Lỗi khi cài TTS: {e}")
    
    # Load TTS model với nhiều fallback options
    if TTS_AVAILABLE:
        print("\n🔊 Đang thử load TTS models (theo thứ tự ưu tiên)...")
        
        # Option 1: XTTS v2
        if tts is None:
            try:
                print("📦 Option 1: Thử load XTTS v2 (chất lượng cao, multilingual)...")
                tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=use_gpu_tts)
                tts_type = "coqui_xtts"
                if use_gpu_tts:
                    print("✅ TTS model (XTTS v2) đã load trên GPU!")
                else:
                    print("✅ TTS model (XTTS v2) đã load trên CPU!")
            except Exception as e:
                print(f"⚠️  Không thể load XTTS v2: {str(e)[:200]}")
                tts = None
        
        # Option 2: Model tiếng Việt
        if tts is None:
            try:
                print("📦 Option 2: Thử load TTS model tiếng Việt...")
                tts = TTS(model_name="tts_models/vi/vietnamese", gpu=use_gpu_tts)
                tts_type = "coqui_vi"
                if use_gpu_tts:
                    print("✅ TTS model tiếng Việt đã load trên GPU!")
                else:
                    print("✅ TTS model tiếng Việt đã load trên CPU!")
            except Exception as e:
                print(f"⚠️  Không thể load TTS tiếng Việt: {str(e)[:200]}")
                tts = None
        
        # Option 3: Model đơn giản
        if tts is None:
            try:
                print("📦 Option 3: Thử load TTS model đơn giản (English)...")
                tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", gpu=use_gpu_tts)
                tts_type = "coqui_en"
                if use_gpu_tts:
                    print("✅ TTS model đơn giản đã load trên GPU!")
                else:
                    print("✅ TTS model đơn giản đã load trên CPU!")
            except Exception as e:
                print(f"⚠️  Không thể load TTS đơn giản: {str(e)[:200]}")
                tts = None
        
        # Option 4: Model mặc định
        if tts is None:
            try:
                print("📦 Option 4: Thử load TTS model mặc định...")
                tts = TTS(gpu=use_gpu_tts)
                tts_type = "coqui_default"
                if use_gpu_tts:
                    print("✅ TTS model mặc định đã load trên GPU!")
                else:
                    print("✅ TTS model mặc định đã load trên CPU!")
            except Exception as e:
                print(f"⚠️  Không thể load TTS mặc định: {str(e)[:200]}")
                tts = None
    
    # Fallback: dùng pyttsx3
    if tts is None:
        print("\n📦 Fallback: Thử dùng pyttsx3...")
        try:
            import pyttsx3
            tts = pyttsx3.init()
            tts_type = "pyttsx3"
            try:
                voices = tts.getProperty('voices')
                for voice in voices:
                    if 'vietnamese' in voice.name.lower() or 'vi' in voice.id.lower():
                        tts.setProperty('voice', voice.id)
                        print(f"✅ Đã chọn giọng: {voice.name}")
                        break
            except:
                pass
            print("✅ TTS model (pyttsx3) đã load!")
            print("   ⚠️  Lưu ý: pyttsx3 có thể cần eSpeak trên Linux")
        except Exception as e:
            print(f"⚠️  Không thể load pyttsx3: {str(e)[:200]}")
            tts = None
    
    # Tóm tắt kết quả
    print("\n" + "="*50)
    if tts is not None:
        print(f"✅ TTS đã sẵn sàng! Type: {tts_type}")
        print("   AI sẽ có thể trả lời bằng giọng nói")
    else:
        print("⚠️  CẢNH BÁO: Không có TTS nào khả dụng!")
        print("="*50)
        print("Server vẫn hoạt động nhưng AI sẽ KHÔNG trả lời bằng giọng nói")
        print("(chỉ hiển thị text response)")
        print("\n💡 Có thể thử các cách sau để cài TTS:")
        print("   1. Chạy lại với install_tts_on_load=True")
        print("   2. Cài thủ công: pip install TTS==0.22.0")
        print("   3. Hoặc cài từ source: pip install git+https://github.com/coqui-ai/TTS.git@v0.22.0")
        print("   4. Hoặc cài pyttsx3 (nhẹ hơn): pip install pyttsx3")
        print("="*50)


def speech_to_text(audio_path: Optional[str]) -> str:
    """
    Chuyển đổi file audio thành text sử dụng Whisper (tối ưu tốc độ)
    """
    global whisper_model
    
    if whisper_model is None:
        return "[Lỗi: Whisper model chưa được load]"
    
    if audio_path is None:
        return ""
    
    try:
        use_fp16 = torch.cuda.is_available()
        result = whisper_model.transcribe(
            audio_path,
            language="vi",
            fp16=use_fp16,
            verbose=False,
            condition_on_previous_text=False,
            initial_prompt="Đây là một cuộc trò chuyện bằng tiếng Việt.",
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            no_speech_threshold=0.6,
            beam_size=1,
            best_of=1,
            temperature=0.0,
        )
        text = result["text"].strip()
        return text
    except Exception as e:
        print(f"Lỗi trong speech-to-text: {e}")
        try:
            result = whisper_model.transcribe(
                audio_path,
                language="vi",
                fp16=torch.cuda.is_available(),
                verbose=False,
                beam_size=1,
            )
            return result["text"].strip()
        except:
            return ""


def process_with_model_stream(text: str):
    """
    Xử lý text với Bank Model với streaming (yield từng phần text)
    """
    global model, processor
    
    if model is None:
        yield "Xin lỗi, model chưa được load. Vui lòng chạy load_models() trước."
        return
    
    if processor is None:
        yield "Xin lỗi, processor chưa được load. Vui lòng chạy load_models() trước."
        return
    
    if not text.strip():
        yield "Xin lỗi, tôi không hiểu. Bạn có thể viết lại được không?"
        return
    
    try:
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": text}]
            }
        ]
        
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        if not isinstance(inputs, dict):
            raise ValueError("Inputs phải là dict sau apply_chat_template")
        if "input_ids" not in inputs:
            raise ValueError("Inputs phải có 'input_ids'")
        
        if len(inputs["input_ids"].shape) != 2:
            inputs["input_ids"] = inputs["input_ids"].unsqueeze(0)
        
        if torch.cuda.is_available():
            try:
                model_device = next(model.parameters()).device
                if model_device.type != "cuda":
                    model = model.to("cuda")
                    model_device = next(model.parameters()).device
            except Exception as e:
                print(f"⚠️  Lỗi khi kiểm tra model device: {e}")
                model_device = torch.device("cuda")
                try:
                    model = model.to("cuda")
                except:
                    model_device = torch.device("cpu")
            
            device = model_device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        else:
            device = torch.device("cpu")
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        tokenizer = processor.tokenizer
        eos_token_id = getattr(tokenizer, 'eos_token_id', None) or getattr(tokenizer, 'pad_token_id', None)
        
        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=60.0,
            clean_up_tokenization_spaces=True,
        )
        
        model.eval()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        generation_kwargs = {
            **inputs,
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "do_sample": True,
            "pad_token_id": eos_token_id,
            "eos_token_id": eos_token_id,
            "use_cache": True,
            "num_beams": 1,
            "repetition_penalty": 1.1,
            "streamer": streamer,
        }
        
        generation_error = [None]
        
        def generate_with_error_handling():
            try:
                with torch.inference_mode():
                    model.generate(**generation_kwargs)
            except Exception as e:
                generation_error[0] = e
                print(f"❌ Lỗi trong generation thread: {e}")
        
        thread = Thread(target=generate_with_error_handling)
        thread.daemon = True
        thread.start()
        
        generated_text = ""
        try:
            for new_text in streamer:
                if generation_error[0]:
                    raise generation_error[0]
                generated_text += new_text
                yield generated_text
        except Exception as e:
            print(f"⚠️  Lỗi trong streaming: {e}")
            if generated_text:
                yield generated_text
            else:
                yield f"Xin lỗi, có lỗi xảy ra khi generate: {str(e)}"
        finally:
            thread.join(timeout=5.0)
            if thread.is_alive():
                print("⚠️  Generation thread vẫn chạy sau timeout")
    
    except Exception as e:
        print(f"Lỗi trong model processing: {e}")
        traceback.print_exc()
        yield f"Xin lỗi, có lỗi xảy ra: {str(e)}"


def process_with_model(text: str) -> str:
    """
    Xử lý text với Bank Model (tối ưu tốc độ) - chỉ text, không có ảnh
    """
    global model, processor
    
    if model is None:
        return "Xin lỗi, model chưa được load. Vui lòng chạy load_models() trước."
    
    if processor is None:
        return "Xin lỗi, processor chưa được load. Vui lòng chạy load_models() trước."
    
    if not text.strip():
        return "Xin lỗi, tôi không nghe rõ. Bạn có thể nói lại được không?"
    
    try:
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": text}]
            }
        ]
        
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        if not isinstance(inputs, dict) or "input_ids" not in inputs:
            raise ValueError("Inputs không đúng format")
        
        if len(inputs["input_ids"].shape) != 2:
            inputs["input_ids"] = inputs["input_ids"].unsqueeze(0)
        
        tokenizer = processor.tokenizer
        eos_token_id = getattr(tokenizer, 'eos_token_id', None) or getattr(tokenizer, 'pad_token_id', None)
        if eos_token_id is None:
            raise ValueError("Tokenizer phải có eos_token_id hoặc pad_token_id")
        
        if torch.cuda.is_available():
            try:
                model_device = next(model.parameters()).device
                if model_device.type != "cuda":
                    print("⚠️  Model không trên GPU, đang chuyển lên GPU...")
                    model = model.to("cuda")
                    model_device = next(model.parameters()).device
                    print(f"✅ Model đã chuyển lên GPU: {torch.cuda.get_device_name(model_device.index)}")
            except Exception as e:
                print(f"⚠️  Lỗi khi kiểm tra model device: {e}")
                try:
                    model = model.to("cuda")
                    model_device = torch.device("cuda")
                    print("✅ Model đã được chuyển lên GPU")
                except:
                    model_device = torch.device("cpu")
                    print("⚠️  Không thể chuyển model lên GPU, sử dụng CPU")
            
            device = model_device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        else:
            device = torch.device("cpu")
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            print("⚠️  Không có GPU, model đang chạy trên CPU")
        
        model.eval()
        with torch.inference_mode():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                do_sample=True,
                pad_token_id=eos_token_id,
                eos_token_id=eos_token_id,
                use_cache=True,
                num_beams=2,
                repetition_penalty=1.1,
                length_penalty=1.0,
                early_stopping=True,
                output_scores=False,
                return_dict_in_generate=False,
            )
        
        input_length = inputs["input_ids"].shape[1]
        
        if len(generated_ids.shape) == 1:
            generated_ids = generated_ids.unsqueeze(0)
        
        generated_ids_trimmed = [
            out_ids[input_length:].cpu()
            for out_ids in generated_ids
        ]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]
        
        output_text = output_text.strip()
        output_text = re.sub(r'\s+', ' ', output_text)
        output_text = output_text.strip()
        
        if not output_text:
            return "Xin lỗi, tôi không thể tạo response. Vui lòng thử lại."
        
        return output_text
    
    except Exception as e:
        print(f"Lỗi trong model processing: {e}")
        return f"Xin lỗi, có lỗi xảy ra: {str(e)}"


def text_to_speech(text: str, lang: str = "vi") -> Optional[str]:
    """
    Chuyển đổi text thành file audio sử dụng TTS (offline, tối ưu tốc độ)
    """
    global tts, tts_type
    
    if tts is None:
        print("⚠️  TTS chưa được load")
        return None
    
    if not text.strip():
        return None
    
    try:
        max_chars = 500
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        
        fd, audio_path = tempfile.mkstemp(suffix=".wav", prefix="tts_")
        os.close(fd)
        
        try:
            if tts_type == "pyttsx3":
                tts.save_to_file(text, audio_path)
                tts.runAndWait()
            elif tts_type and tts_type.startswith("coqui"):
                if tts_type == "coqui_xtts":
                    tts.tts_to_file(
                        text=text,
                        file_path=audio_path,
                        language=lang,
                        speaker_wav=None,
                        speed=1.3,
                    )
                else:
                    tts.tts_to_file(text=text, file_path=audio_path)
            else:
                if hasattr(tts, 'tts_to_file'):
                    tts.tts_to_file(text=text, file_path=audio_path)
                else:
                    print("TTS không hỗ trợ tts_to_file")
                    return None
            
            return audio_path
        except Exception as e:
            if os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except:
                    pass
            raise e
        
    except Exception as e:
        print(f"Lỗi trong text-to-speech: {e}")
        return None


def create_gradio_interface():
    """Tạo Gradio interface"""
    
    # Hàm xử lý text input với streaming (không có TTS)
    def chat_text_stream(user_text, history):
        if not user_text or not user_text.strip():
            return history, "", None
        
        if history is None:
            history = []
        history.append((user_text.strip(), None))
        
        response_text = ""
        try:
            for partial_response in process_with_model_stream(user_text.strip()):
                response_text = partial_response
                history[-1] = (user_text.strip(), response_text)
                yield history, "", None
        except Exception as e:
            print(f"Lỗi trong chat_stream: {e}")
            response_text = f"Xin lỗi, có lỗi xảy ra: {str(e)}"
            history[-1] = (user_text.strip(), response_text)
        
        yield history, "", None
    
    # Hàm xử lý text input với streaming + TTS
    def chat_text_stream_with_tts(user_text, history):
        if not user_text or not user_text.strip():
            return history, "", None
        
        if history is None:
            history = []
        history.append((user_text.strip(), None))
        
        response_text = ""
        try:
            for partial_response in process_with_model_stream(user_text.strip()):
                response_text = partial_response
                history[-1] = (user_text.strip(), response_text)
                yield history, "", None
        except Exception as e:
            print(f"Lỗi trong chat_stream: {e}")
            response_text = f"Xin lỗi, có lỗi xảy ra: {str(e)}"
            history[-1] = (user_text.strip(), response_text)
        
        audio_output_path = None
        if response_text and response_text.strip():
            try:
                print("🔊 Đang tạo AI voice reply...")
                audio_output_path = text_to_speech(response_text, lang="vi")
                if audio_output_path:
                    print(f"✅ AI voice reply đã được tạo: {audio_output_path}")
            except Exception as e:
                print(f"⚠️  Lỗi khi tạo TTS: {e}")
        
        yield history, "", audio_output_path
    
    # Hàm xử lý voice input với streaming + TTS
    def chat_voice_stream_with_tts(audio_input, history):
        if audio_input is None:
            return history, None
        
        if whisper_model is None:
            error_msg = "Whisper model chưa được load"
            if history is None:
                history = []
            history.append(("[Lỗi]", error_msg))
            return history, None
        
        user_text = speech_to_text(audio_input)
        
        if not user_text or not user_text.strip():
            error_msg = "Xin lỗi, tôi không nghe rõ. Bạn có thể nói lại được không?"
            if history is None:
                history = []
            history.append(("[Không nghe rõ]", error_msg))
            audio_output_path = None
            try:
                audio_output_path = text_to_speech(error_msg, lang="vi")
            except:
                pass
            return history, audio_output_path
        
        if history is None:
            history = []
        history.append((user_text.strip(), None))
        
        response_text = ""
        try:
            for partial_response in process_with_model_stream(user_text.strip()):
                response_text = partial_response
                history[-1] = (user_text.strip(), response_text)
                yield history, None
        except Exception as e:
            print(f"Lỗi trong chat_voice_stream: {e}")
            response_text = f"Xin lỗi, có lỗi xảy ra: {str(e)}"
            history[-1] = (user_text.strip(), response_text)
        
        audio_output_path = None
        if response_text and response_text.strip():
            try:
                print("🔊 Đang tạo AI voice reply...")
                audio_output_path = text_to_speech(response_text, lang="vi")
                if audio_output_path:
                    print(f"✅ AI voice reply đã được tạo: {audio_output_path}")
            except Exception as e:
                print(f"⚠️  Lỗi khi tạo TTS: {e}")
        
        yield history, audio_output_path
    
    # Tạo Gradio interface
    with gr.Blocks(title="Chat với Bank Model - Text & Voice Input + AI Voice Reply") as demo:
        gr.Markdown("""
        # 💬 Chat với Bank Model - Text & Voice Input + AI Voice Reply
        
        Chat bằng text HOẶC giọng nói với AI model, AI sẽ trả lời bằng giọng nói!
        - ✍️ Nhập text và nhấn Enter hoặc nút Gửi
        - 🎙️ Hoặc nói vào microphone
        - ⚡ Response được stream từng phần (không cần chờ)
        - 🔊 AI trả lời bằng giọng nói (TTS)
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="💬 Lịch sử chat",
                    height=400,
                    show_label=True,
                    type="tuples",
                    allow_tags=False
                )
                
                audio_output = gr.Audio(
                    label="🔊 AI Voice Reply",
                    type="filepath",
                    show_label=True,
                    visible=True
                )
            
            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.Tab("✍️ Text Input"):
                        text_input = gr.Textbox(
                            label="Nhập câu hỏi",
                            placeholder="Nhập câu hỏi của bạn ở đây...",
                            lines=3,
                            show_label=True
                        )
                        text_submit_btn = gr.Button("Gửi Text", variant="primary", size="lg")
                        text_with_voice_btn = gr.Button("Gửi Text + AI Voice", variant="secondary", size="lg")
                    
                    with gr.Tab("🎙️ Voice Input"):
                        audio_input = gr.Audio(
                            sources=["microphone"],
                            type="filepath",
                            label="Nói vào đây",
                            show_label=True
                        )
                        audio_submit_btn = gr.Button("Gửi Voice + AI Voice", variant="primary", size="lg")
                
                clear_btn = gr.Button("Xóa lịch sử", variant="secondary")
                
                resources_display = gr.Markdown(
                    value=format_resources_info(),
                    label="📊 Tài nguyên hệ thống"
                )
        
        # Event handlers
        text_submit_btn.click(
            fn=chat_text_stream,
            inputs=[text_input, chatbot],
            outputs=[chatbot, text_input, audio_output],
            show_progress=False
        )
        
        text_input.submit(
            fn=chat_text_stream,
            inputs=[text_input, chatbot],
            outputs=[chatbot, text_input, audio_output],
            show_progress=False
        )
        
        text_with_voice_btn.click(
            fn=chat_text_stream_with_tts,
            inputs=[text_input, chatbot],
            outputs=[chatbot, text_input, audio_output],
            show_progress=False
        )
        
        audio_submit_btn.click(
            fn=chat_voice_stream_with_tts,
            inputs=[audio_input, chatbot],
            outputs=[chatbot, audio_output],
            show_progress=False
        )
        
        audio_input.stop_recording(
            fn=chat_voice_stream_with_tts,
            inputs=[audio_input, chatbot],
            outputs=[chatbot, audio_output],
            show_progress=False
        )
        
        clear_btn.click(
            fn=lambda: [],
            outputs=[chatbot]
        )
        
        demo.load(
            fn=lambda: format_resources_info(),
            inputs=None,
            outputs=resources_display
        )
        
        gr.Markdown("""
        ### Hướng dẫn sử dụng:
        1. **Text Input**: 
           - ✍️ Nhập câu hỏi vào ô text
           - ⏎ Nhấn Enter hoặc nút "Gửi Text" (chỉ text, không có voice)
           - 🔊 Hoặc nhấn "Gửi Text + AI Voice" (AI sẽ nói trả lời)
        
        2. **Voice Input**:
           - 🎙️ Nhấn nút microphone và bắt đầu nói
           - ⏹️ Dừng recording hoặc nhấn "Gửi Voice + AI Voice"
           - 🔊 AI sẽ tự động trả lời bằng giọng nói
        
        3. ⚡ Xem response được stream từng phần (không cần chờ)
        4. 🔊 Nghe AI voice reply ở phần "AI Voice Reply" bên dưới
        5. 📊 Xem tài nguyên hệ thống ở bên phải
        
        ### ⚡ Tính năng:
        - Response được stream realtime, không cần chờ toàn bộ
        - Hỗ trợ cả text và voice input
        - Model chạy trên GPU để tối ưu tốc độ
        - 🔊 **AI trả lời bằng giọng nói (TTS)** - Tính năng mới!
        - TTS offline (Coqui TTS hoặc pyttsx3)
        
        ### Lưu ý:
        - Có thể dùng text HOẶC voice để input
        - Response hiển thị text VÀ có audio (AI nói)
        - Response được stream từng token
        - TTS sẽ được tạo sau khi có full response
        - Nếu TTS không khả dụng, chỉ hiển thị text
        """)
    
    return demo


def main():
    """Hàm main để chạy server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Voice Chat Server với Bank Model")
    parser.add_argument("--model-name", type=str, default="hainguyen306201/bank-model-2b",
                       help="Tên model trên Hugging Face")
    parser.add_argument("--install-tts", action="store_true",
                       help="Cài đặt TTS khi load model")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                       help="Host để chạy server")
    parser.add_argument("--port", type=int, default=7860,
                       help="Port để chạy server")
    parser.add_argument("--share", action="store_true",
                       help="Tạo public link (Gradio share)")
    parser.add_argument("--debug", action="store_true",
                       help="Chạy ở chế độ debug")
    
    args = parser.parse_args()
    
    print("="*60)
    print("🚀 Khởi động Voice Chat Server với Bank Model")
    print("="*60)
    
    # Load models
    print("\n📦 Đang load models...")
    load_models(model_name=args.model_name, install_tts_on_load=args.install_tts)
    
    # Tạo Gradio interface
    print("\n🎨 Đang tạo Gradio interface...")
    demo = create_gradio_interface()
    print("✅ Gradio interface đã được tạo!")
    
    # Khởi động server
    print("\n🌐 Đang khởi động server...")
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"   Share: {args.share}")
    print(f"   Debug: {args.debug}")
    
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=args.debug
    )


if __name__ == "__main__":
    main()

