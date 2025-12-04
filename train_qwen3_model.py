#!/usr/bin/env python3
"""
Train Qwen3 Model - Fine-tune Qwen3/Qwen3VL Model

Script Python để fine-tune Qwen3 model với LoRA.

Usage:
    python train_qwen3_model.py --model hainguyen306201/bank-model-2b --dataset dataset.json
    python train_qwen3_model.py --model hainguyen306201/bank-model-2b --dataset dataset.jsonl --output ./checkpoints
    python train_qwen3_model.py --model hainguyen306201/bank-model-2b --dataset dataset.json --epochs 3 --batch-size 4

Theo tài liệu: https://qwen.readthedocs.io/en/latest/
"""

import os
import json
import argparse
import torch
from pathlib import Path
from typing import Optional
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, Dataset
from tqdm import tqdm
from huggingface_hub import login, HfApi, create_repo


def load_model_and_processor(
    model_name: str,
    use_quantization: bool = True,
    use_flash_attention: bool = False
):
    """
    Load model và processor
    
    Args:
        model_name: Tên model trên Hugging Face
        use_quantization: Có dùng 4-bit quantization không
        use_flash_attention: Có dùng Flash Attention 2 không
    """
    print(f"📥 Đang load model: {model_name}...")
    
    # Cấu hình quantization
    bnb_config = None
    if use_quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        print("✅ Sử dụng 4-bit quantization")
    
    # Load model
    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
    }
    
    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config
    
    # Flash Attention 2
    if use_flash_attention:
        try:
            import flash_attn
            model_kwargs["attn_implementation"] = "flash_attention_2"
            print("✅ Sử dụng Flash Attention 2")
        except ImportError:
            print("⚠️  Flash Attention 2 chưa được cài, sử dụng attention mặc định")
            model_kwargs["attn_implementation"] = "sdpa"
    else:
        model_kwargs["attn_implementation"] = "sdpa"
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        **model_kwargs
    )
    
    # Load processor
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    
    print("✅ Model và processor đã được load!")
    
    return model, processor


def setup_lora(model, r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05):
    """
    Setup LoRA cho model
    
    Args:
        model: Model đã load
        r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
    """
    print("🔧 Đang setup LoRA...")
    
    # Chuẩn bị model cho training
    model = prepare_model_for_kbit_training(model)
    
    # Cấu hình LoRA
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Áp dụng LoRA
    model = get_peft_model(model, lora_config)
    
    # In thông tin
    model.print_trainable_parameters()
    
    return model


def load_dataset_from_file(file_path: str) -> Dataset:
    """
    Load dataset từ file (JSON hoặc JSONL)
    
    Args:
        file_path: Đường dẫn file dataset
    """
    print(f"📂 Đang load dataset từ {file_path}...")
    
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext == ".jsonl":
        # Load từ JSONL
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        dataset = Dataset.from_list(data)
    elif file_ext == ".json":
        # Load từ JSON
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            dataset = Dataset.from_list(data)
        else:
            dataset = Dataset.from_list([data])
    else:
        # Thử load bằng datasets library
        dataset = load_dataset(file_path, split="train")
    
    print(f"✅ Đã load {len(dataset)} samples")
    return dataset


def preprocess_function(examples, processor):
    """
    Preprocess function cho dataset
    
    Args:
        examples: Batch từ dataset
        processor: AutoProcessor
    """
    messages_list = examples["messages"]
    
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    
    for messages in messages_list:
        # Apply chat template - theo chuẩn Qwen3
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,  # False cho training
            return_dict=True,
            return_tensors="pt"
        )
        
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.squeeze(0)
        else:
            attention_mask = torch.ones_like(input_ids)
        
        # Labels: copy input_ids (sẽ mask user tokens sau nếu cần)
        labels = input_ids.clone()
        
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)
    
    # Pad sequences
    max_length = max(len(ids) for ids in input_ids_list)
    
    # Pad function
    def pad_sequence(seq, max_len, pad_value=processor.tokenizer.pad_token_id):
        if len(seq) >= max_len:
            return seq[:max_len]
        padded = torch.cat([seq, torch.full((max_len - len(seq),), pad_value, dtype=seq.dtype)])
        return padded
    
    pad_token_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    
    input_ids_padded = torch.stack([pad_sequence(ids, max_length, pad_token_id) for ids in input_ids_list])
    attention_mask_padded = torch.stack([pad_sequence(mask, max_length, 0) for mask in attention_mask_list])
    labels_padded = torch.stack([pad_sequence(labels, max_length, -100) for labels in labels_list])
    
    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask_padded,
        "labels": labels_padded,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train Qwen3 Model - Fine-tune với LoRA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  # Training cơ bản
  python train_qwen3_model.py --model hainguyen306201/bank-model-2b --dataset dataset.json
  
  # Training với custom output và epochs
  python train_qwen3_model.py --model hainguyen306201/bank-model-2b --dataset dataset.jsonl --output ./checkpoints --epochs 5
  
  # Training với batch size lớn hơn
  python train_qwen3_model.py --model hainguyen306201/bank-model-2b --dataset dataset.json --batch-size 8 --gradient-accumulation 2
        """
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Tên model trên Hugging Face (ví dụ: hainguyen306201/bank-model-2b)"
    )
    
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        required=True,
        help="Đường dẫn file dataset (JSON hoặc JSONL)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./checkpoints",
        help="Thư mục lưu checkpoints (default: ./checkpoints)"
    )
    
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=3,
        help="Số epochs (default: 3)"
    )
    
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=4,
        help="Batch size (default: 4)"
    )
    
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=1,
        help="Gradient accumulation steps (default: 1)"
    )
    
    parser.add_argument(
        "--learning-rate", "-lr",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)"
    )
    
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank (default: 16)"
    )
    
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha (default: 32)"
    )
    
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout (default: 0.05)"
    )
    
    parser.add_argument(
        "--no-quantization",
        action="store_true",
        help="Không dùng quantization (cần GPU lớn)"
    )
    
    parser.add_argument(
        "--flash-attention",
        action="store_true",
        help="Sử dụng Flash Attention 2 (cần cài flash-attn)"
    )
    
    parser.add_argument(
        "--save-steps",
        type=int,
        default=500,
        help="Lưu checkpoint mỗi N steps (default: 500)"
    )
    
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=50,
        help="Log mỗi N steps (default: 50)"
    )
    
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Max sequence length (default: 2048)"
    )
    
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="Warmup steps (default: 100)"
    )
    
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Upload model lên Hugging Face sau khi train xong"
    )
    
    parser.add_argument(
        "--hub-model-id",
        type=str,
        default=None,
        help="Tên model trên Hugging Face (ví dụ: username/model-name). Cần thiết nếu dùng --push-to-hub"
    )
    
    parser.add_argument(
        "--hub-token",
        type=str,
        default=None,
        help="Hugging Face token. Nếu không chỉ định, sẽ dùng token từ môi trường hoặc cache"
    )
    
    parser.add_argument(
        "--hub-private",
        action="store_true",
        help="Tạo private repository trên Hugging Face"
    )
    
    args = parser.parse_args()
    
    # Kiểm tra dataset
    if not os.path.exists(args.dataset):
        print(f"❌ File dataset không tồn tại: {args.dataset}")
        return
    
    # Kiểm tra GPU
    if not torch.cuda.is_available():
        print("⚠️  Không có GPU, training sẽ rất chậm!")
    else:
        print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    
    # Load model và processor
    model, processor = load_model_and_processor(
        args.model,
        use_quantization=not args.no_quantization,
        use_flash_attention=args.flash_attention
    )
    
    # Setup LoRA
    model = setup_lora(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
    
    # Load dataset
    dataset = load_dataset_from_file(args.dataset)
    
    # Preprocess dataset
    print("🔄 Đang preprocess dataset...")
    processed_dataset = dataset.map(
        lambda x: preprocess_function({"messages": [x["messages"]]}, processor),
        batched=False,
        remove_columns=dataset.column_names,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        fp16=torch.cuda.is_available(),
        bf16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
        optim="adamw_torch",
        report_to="tensorboard",
        remove_unused_columns=False,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=processor.tokenizer,
        mlm=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        data_collator=data_collator,
    )
    
    # Training
    print("\n" + "="*50)
    print("🚀 BẮT ĐẦU TRAINING")
    print("="*50)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset} ({len(dataset)} samples)")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Output: {args.output}")
    print("="*50 + "\n")
    
    trainer.train()
    
    # Save final model
    print("\n💾 Đang lưu model cuối cùng...")
    trainer.save_model()
    processor.save_pretrained(args.output)
    
    print(f"\n✅ Training hoàn thành! Model đã được lưu tại: {args.output}")
    
    # Upload lên Hugging Face nếu được yêu cầu
    if args.push_to_hub:
        if not args.hub_model_id:
            print("❌ Cần chỉ định --hub-model-id để upload lên Hugging Face")
            return
        
        print(f"\n📤 Đang upload model lên Hugging Face: {args.hub_model_id}...")
        
        # Login vào Hugging Face
        if args.hub_token:
            login(token=args.hub_token)
        else:
            try:
                login()  # Thử login với token đã lưu
            except Exception as e:
                print(f"❌ Lỗi khi login Hugging Face: {e}")
                print("💡 Hãy chạy: huggingface-cli login hoặc chỉ định --hub-token")
                return
        
        # Tạo repository nếu chưa tồn tại
        api = HfApi()
        try:
            create_repo(
                repo_id=args.hub_model_id,
                repo_type="model",
                private=args.hub_private,
                exist_ok=True
            )
            print(f"✅ Repository {args.hub_model_id} đã sẵn sàng")
        except Exception as e:
            print(f"⚠️  Repository có thể đã tồn tại hoặc có lỗi: {e}")
        
        # Upload model
        try:
            # Upload từ output directory
            api.upload_folder(
                folder_path=args.output,
                repo_id=args.hub_model_id,
                repo_type="model",
                commit_message=f"Upload fine-tuned model after {args.epochs} epochs training"
            )
            print(f"\n✅ Đã upload model thành công lên: https://huggingface.co/{args.hub_model_id}")
        except Exception as e:
            print(f"❌ Lỗi khi upload model: {e}")
            print(f"💡 Bạn có thể upload thủ công bằng lệnh:")
            print(f"   huggingface-cli upload {args.hub_model_id} {args.output} --repo-type model")


if __name__ == "__main__":
    main()

