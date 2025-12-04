#!/usr/bin/env python3
"""
Qwen3 Dataset Builder - Tạo Training Data theo Chuẩn Qwen3

Script Python để xây dựng và chuẩn bị dataset cho training Qwen3/Qwen3VL model.

Usage:
    python qwen3_dataset_builder.py --input data.csv --output dataset.json
    python qwen3_dataset_builder.py --input data.json --format qa --output dataset.jsonl
    python qwen3_dataset_builder.py --input data.jsonl --format jsonl --output dataset.json --split 0.2

Theo tài liệu: https://qwen.readthedocs.io/en/latest/
"""

import json
import os
import argparse
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd
from datasets import Dataset, DatasetDict
from tqdm import tqdm


class Qwen3DatasetBuilder:
    """
    Dataset Builder cho Qwen3/Qwen3VL training
    Theo chuẩn Qwen3: https://qwen.readthedocs.io/en/latest/
    
    Format chuẩn Qwen3:
    {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "..."}]},
            {"role": "assistant", "content": [{"type": "text", "text": "..."}]}
        ]
    }
    """
    
    def __init__(self, include_image: bool = False):
        """
        Args:
            include_image: Nếu True, hỗ trợ image trong content (Qwen3VL)
        """
        self.include_image = include_image
        self.dataset = []
        self.stats = {
            "total_samples": 0,
            "total_conversations": 0,
            "total_turns": 0,
            "avg_turns_per_conv": 0,
            "total_tokens_estimate": 0,
        }
    
    def add_conversation(self, messages: List[Dict[str, Any]]) -> None:
        """Thêm một conversation vào dataset"""
        if not self._validate_messages(messages):
            raise ValueError("Messages không đúng format Qwen3")
        
        self.dataset.append({"messages": messages})
        self._update_stats(messages)
    
    def add_simple_qa(self, question: str, answer: str, image_path: Optional[str] = None) -> None:
        """Thêm một Q&A đơn giản (1 turn conversation)"""
        user_content = []
        
        if image_path and self.include_image:
            user_content.append({"type": "image", "image": image_path})
        
        user_content.append({"type": "text", "text": question})
        
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]}
        ]
        
        self.add_conversation(messages)
    
    def add_multi_turn_conversation(self, turns: List[Dict[str, str]], image_path: Optional[str] = None) -> None:
        """Thêm multi-turn conversation"""
        messages = []
        
        for i, turn in enumerate(turns):
            if "user" not in turn or "assistant" not in turn:
                raise ValueError(f"Turn {i} phải có 'user' và 'assistant'")
            
            user_content = []
            if image_path and self.include_image and i == 0:
                user_content.append({"type": "image", "image": image_path})
            user_content.append({"type": "text", "text": turn["user"]})
            messages.append({"role": "user", "content": user_content})
            
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": turn["assistant"]}]
            })
        
        self.add_conversation(messages)
    
    def load_from_json(self, file_path: str, format_type: str = "auto") -> None:
        """Load dataset từ JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if format_type == "auto":
            format_type = self._detect_format(data)
        
        if format_type == "qwen3":
            if isinstance(data, list):
                for item in data:
                    if "messages" in item:
                        self.add_conversation(item["messages"])
            elif isinstance(data, dict) and "messages" in data:
                self.add_conversation(data["messages"])
        elif format_type == "qa":
            for item in data:
                self.add_simple_qa(item.get("question", ""), item.get("answer", ""))
        elif format_type == "conversation":
            for item in data:
                if "turns" in item:
                    self.add_multi_turn_conversation(item["turns"])
        
        print(f"✅ Đã load {len(data) if isinstance(data, list) else 1} samples từ {file_path}")
    
    def load_from_csv(self, file_path: str, question_col: str = "question", answer_col: str = "answer") -> None:
        """Load dataset từ CSV file"""
        df = pd.read_csv(file_path)
        
        if question_col not in df.columns or answer_col not in df.columns:
            raise ValueError(f"CSV phải có columns: {question_col}, {answer_col}")
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading CSV"):
            question = str(row[question_col]).strip()
            answer = str(row[answer_col]).strip()
            
            if question and answer:
                self.add_simple_qa(question, answer)
        
        print(f"✅ Đã load {len(df)} samples từ {file_path}")
    
    def load_from_jsonl(self, file_path: str) -> None:
        """Load từ JSONL file (mỗi dòng một JSON conversation)"""
        count = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    if "messages" in data:
                        self.add_conversation(data["messages"])
                        count += 1
                except json.JSONDecodeError as e:
                    print(f"⚠️  Lỗi ở dòng {line_num}: {e}")
        
        print(f"✅ Đã load {count} samples từ {file_path}")
    
    def _validate_messages(self, messages: List[Dict[str, Any]]) -> bool:
        """Validate messages format theo chuẩn Qwen3"""
        if not isinstance(messages, list) or len(messages) == 0:
            return False
        
        for msg in messages:
            if not isinstance(msg, dict):
                return False
            
            if "role" not in msg or "content" not in msg:
                return False
            
            if msg["role"] not in ["user", "assistant", "system"]:
                return False
            
            if not isinstance(msg["content"], list):
                return False
            
            for content_item in msg["content"]:
                if not isinstance(content_item, dict):
                    return False
                if "type" not in content_item:
                    return False
                if content_item["type"] == "text":
                    if "text" not in content_item:
                        return False
                elif content_item["type"] == "image":
                    if "image" not in content_item:
                        return False
                    if not self.include_image:
                        return False
                else:
                    return False
        
        return True
    
    def _detect_format(self, data: Any) -> str:
        """Tự động detect format của data"""
        if isinstance(data, list) and len(data) > 0:
            first_item = data[0]
            if isinstance(first_item, dict):
                if "messages" in first_item:
                    return "qwen3"
                elif "question" in first_item and "answer" in first_item:
                    return "qa"
                elif "turns" in first_item:
                    return "conversation"
        elif isinstance(data, dict):
            if "messages" in data:
                return "qwen3"
        
        return "qa"
    
    def _update_stats(self, messages: List[Dict[str, Any]]) -> None:
        """Cập nhật thống kê"""
        self.stats["total_conversations"] += 1
        self.stats["total_turns"] += len([m for m in messages if m["role"] == "user"])
        
        total_chars = 0
        for msg in messages:
            for content in msg.get("content", []):
                if content.get("type") == "text":
                    total_chars += len(content.get("text", ""))
        self.stats["total_tokens_estimate"] += total_chars // 4
        
        self.stats["total_samples"] = len(self.dataset)
        if self.stats["total_conversations"] > 0:
            self.stats["avg_turns_per_conv"] = self.stats["total_turns"] / self.stats["total_conversations"]
    
    def get_stats(self) -> Dict[str, Any]:
        """Lấy thống kê dataset"""
        return self.stats.copy()
    
    def preview(self, n: int = 3) -> None:
        """Preview n samples đầu tiên"""
        print(f"\n📊 Preview {min(n, len(self.dataset))} samples đầu tiên:\n")
        for i, sample in enumerate(self.dataset[:n], 1):
            print(f"--- Sample {i} ---")
            print(json.dumps(sample, ensure_ascii=False, indent=2))
            print()
    
    def to_huggingface_dataset(self) -> Dataset:
        """Convert sang Hugging Face Dataset"""
        return Dataset.from_list(self.dataset)
    
    def save_to_json(self, file_path: str, indent: int = 2) -> None:
        """Lưu dataset ra JSON file"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.dataset, f, ensure_ascii=False, indent=indent)
        print(f"✅ Đã lưu {len(self.dataset)} samples vào {file_path}")
    
    def save_to_jsonl(self, file_path: str) -> None:
        """Lưu dataset ra JSONL file"""
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in self.dataset:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"✅ Đã lưu {len(self.dataset)} samples vào {file_path}")
    
    def split_train_test(self, test_ratio: float = 0.1, shuffle: bool = True, seed: int = 42) -> DatasetDict:
        """Chia dataset thành train và test"""
        dataset = self.to_huggingface_dataset()
        
        if shuffle:
            dataset = dataset.shuffle(seed=seed)
        
        split_dataset = dataset.train_test_split(test_size=test_ratio, seed=seed)
        return split_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3 Dataset Builder - Tạo Training Data theo Chuẩn Qwen3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  # Load từ CSV và export ra JSON
  python qwen3_dataset_builder.py --input data.csv --output dataset.json
  
  # Load từ JSON với format Q&A
  python qwen3_dataset_builder.py --input data.json --format qa --output dataset.jsonl
  
  # Load từ JSONL và chia train/test
  python qwen3_dataset_builder.py --input data.jsonl --format jsonl --output dataset.json --split 0.2
  
  # Preview dataset
  python qwen3_dataset_builder.py --input data.json --preview 5
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Đường dẫn file input (CSV, JSON, hoặc JSONL)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Đường dẫn file output (JSON hoặc JSONL). Nếu không chỉ định, chỉ preview"
    )
    
    parser.add_argument(
        "--format", "-f",
        type=str,
        default="auto",
        choices=["auto", "qwen3", "qa", "conversation", "csv", "jsonl"],
        help="Format của file input (auto: tự động detect)"
    )
    
    parser.add_argument(
        "--question-col",
        type=str,
        default="question",
        help="Tên cột chứa câu hỏi (cho CSV)"
    )
    
    parser.add_argument(
        "--answer-col",
        type=str,
        default="answer",
        help="Tên cột chứa câu trả lời (cho CSV)"
    )
    
    parser.add_argument(
        "--split",
        type=float,
        default=None,
        help="Tỷ lệ test set (0.0-1.0). Nếu chỉ định, sẽ tạo train/test split"
    )
    
    parser.add_argument(
        "--preview", "-p",
        type=int,
        default=0,
        help="Số samples để preview (0 = không preview)"
    )
    
    parser.add_argument(
        "--include-image",
        action="store_true",
        help="Hỗ trợ image trong content (cho Qwen3VL)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed cho train/test split"
    )
    
    args = parser.parse_args()
    
    # Kiểm tra file input
    if not os.path.exists(args.input):
        print(f"❌ File không tồn tại: {args.input}")
        return
    
    # Tạo builder
    builder = Qwen3DatasetBuilder(include_image=args.include_image)
    
    # Load data
    print(f"📂 Đang load data từ {args.input}...")
    input_ext = Path(args.input).suffix.lower()
    
    if args.format == "jsonl" or input_ext == ".jsonl":
        builder.load_from_jsonl(args.input)
    elif args.format == "csv" or input_ext == ".csv":
        builder.load_from_csv(args.input, question_col=args.question_col, answer_col=args.answer_col)
    else:
        builder.load_from_json(args.input, format_type=args.format)
    
    # Preview
    if args.preview > 0:
        builder.preview(args.preview)
    
    # Thống kê
    stats = builder.get_stats()
    print("\n" + "="*50)
    print("📊 THỐNG KÊ DATASET")
    print("="*50)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Export
    if args.output:
        output_ext = Path(args.output).suffix.lower()
        
        if args.split is not None:
            # Chia train/test
            print(f"\n🔄 Đang chia dataset (test ratio: {args.split})...")
            split_dataset = builder.split_train_test(test_ratio=args.split, shuffle=True, seed=args.seed)
            
            print(f"✅ Train set: {len(split_dataset['train'])} samples")
            print(f"✅ Test set: {len(split_dataset['test'])} samples")
            
            # Lưu train/test
            train_path = str(Path(args.output).with_suffix('.train.jsonl'))
            test_path = str(Path(args.output).with_suffix('.test.jsonl'))
            
            # Convert train/test về list và lưu
            train_data = [{"messages": item["messages"]} for item in split_dataset['train']]
            test_data = [{"messages": item["messages"]} for item in split_dataset['test']]
            
            with open(train_path, 'w', encoding='utf-8') as f:
                for item in train_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            with open(test_path, 'w', encoding='utf-8') as f:
                for item in test_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            print(f"✅ Đã lưu train set vào {train_path}")
            print(f"✅ Đã lưu test set vào {test_path}")
        else:
            # Lưu toàn bộ dataset
            if output_ext == ".jsonl":
                builder.save_to_jsonl(args.output)
            else:
                builder.save_to_json(args.output)
    
    print("\n✅ Hoàn thành!")


if __name__ == "__main__":
    main()

