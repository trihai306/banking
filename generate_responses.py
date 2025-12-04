#!/usr/bin/env python3
"""
Script để generate assistant responses cho dataset
Sử dụng model base để tạo responses cho các user messages
"""

import json
import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


def generate_response(model, processor, messages, max_new_tokens=512):
    """Generate response từ model"""
    # Apply chat template
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=processor.tokenizer.eos_token_id
        )
    
    # Decode response
    input_length = inputs["input_ids"].shape[1]
    generated_ids_trimmed = generated_ids[:, input_length:]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    
    return output_text[0].strip()


def process_dataset(
    input_file: str,
    output_file: str,
    model_name: str,
    max_new_tokens: int = 512,
    batch_size: int = 1
):
    """Process dataset và generate responses"""
    print(f"📥 Đang load model: {model_name}...")
    
    # Load model
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    
    print("✅ Model đã được load!\n")
    
    # Đọc dataset
    print(f"📂 Đang đọc dataset: {input_file}...")
    samples = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    
    print(f"✅ Đã đọc {len(samples)} samples\n")
    
    # Process từng sample
    print("🔄 Đang generate responses...")
    processed_samples = []
    
    for i, sample in enumerate(tqdm(samples, desc="Generating")):
        if "messages" not in sample:
            continue
        
        messages = sample["messages"]
        
        # Kiểm tra xem đã có assistant response chưa
        has_assistant = any(msg.get("role") == "assistant" for msg in messages)
        if has_assistant:
            # Đã có response, giữ nguyên
            processed_samples.append(sample)
            continue
        
        # Tìm user message cuối cùng
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        if not user_messages:
            continue
        
        # Generate response
        try:
            response = generate_response(model, processor, messages, max_new_tokens)
            
            # Thêm assistant response vào messages
            new_messages = messages.copy()
            new_messages.append({
                "role": "assistant",
                "content": response
            })
            
            # Tạo sample mới
            new_sample = {
                "messages": new_messages
            }
            
            # Giữ metadata nếu có
            if "metadata" in sample:
                new_sample["metadata"] = sample["metadata"]
            
            processed_samples.append(new_sample)
            
        except Exception as e:
            print(f"\n⚠️  Lỗi khi generate response cho sample {i+1}: {e}")
            # Giữ nguyên sample gốc nếu có lỗi
            processed_samples.append(sample)
    
    # Lưu dataset mới
    print(f"\n💾 Đang lưu dataset mới: {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in processed_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"✅ Đã lưu {len(processed_samples)} samples vào {output_file}")
    
    # Thống kê
    with_assistant = sum(
        1 for s in processed_samples
        if any(msg.get("role") == "assistant" for msg in s.get("messages", []))
    )
    print(f"\n📊 Thống kê:")
    print(f"   - Tổng samples: {len(processed_samples)}")
    print(f"   - Có assistant responses: {with_assistant} ({with_assistant/len(processed_samples)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate assistant responses cho dataset"
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="File dataset input (JSONL)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="File dataset output (JSONL)"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="hainguyen306201/bank-model-2b",
        help="Tên model để generate responses"
    )
    
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Max tokens cho response (default: 512)"
    )
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"❌ File không tồn tại: {args.input}")
        return
    
    process_dataset(
        args.input,
        args.output,
        args.model,
        args.max_new_tokens
    )


if __name__ == "__main__":
    main()

