#!/usr/bin/env python3
"""
Script kiểm tra dataset trước khi training
Kiểm tra format, số lượng samples, và cảnh báo các vấn đề tiềm ẩn
"""

import json
import sys
from pathlib import Path
from collections import Counter


def check_dataset(file_path: str):
    """Kiểm tra dataset và hiển thị thống kê"""
    print(f"📂 Đang kiểm tra dataset: {file_path}\n")
    
    if not Path(file_path).exists():
        print(f"❌ File không tồn tại: {file_path}")
        return False
    
    # Đọc dataset
    samples = []
    issues = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                samples.append((line_num, data))
            except json.JSONDecodeError as e:
                issues.append(f"Dòng {line_num}: JSON không hợp lệ - {e}")
    
    if not samples:
        print("❌ Dataset trống!")
        return False
    
    print(f"✅ Đã đọc {len(samples)} samples\n")
    
    # Kiểm tra format
    role_counts = Counter()
    has_system = 0
    has_user = 0
    has_assistant = 0
    missing_assistant = []
    
    for line_num, data in samples:
        if "messages" not in data:
            issues.append(f"Dòng {line_num}: Thiếu field 'messages'")
            continue
        
        messages = data.get("messages", [])
        if not isinstance(messages, list) or len(messages) == 0:
            issues.append(f"Dòng {line_num}: Messages rỗng hoặc không phải list")
            continue
        
        # Đếm roles
        roles = [msg.get("role") for msg in messages if isinstance(msg, dict)]
        role_counts.update(roles)
        
        # Kiểm tra có system/user/assistant không
        if any(msg.get("role") == "system" for msg in messages):
            has_system += 1
        if any(msg.get("role") == "user" for msg in messages):
            has_user += 1
        if any(msg.get("role") == "assistant" for msg in messages):
            has_assistant += 1
        else:
            missing_assistant.append(line_num)
    
    # Hiển thị thống kê
    print("=" * 60)
    print("📊 THỐNG KÊ DATASET")
    print("=" * 60)
    print(f"Tổng số samples: {len(samples)}")
    print(f"\nPhân bố roles:")
    for role, count in role_counts.most_common():
        print(f"  - {role}: {count} messages")
    
    print(f"\nSamples có:")
    print(f"  - System message: {has_system}/{len(samples)} ({has_system/len(samples)*100:.1f}%)")
    print(f"  - User message: {has_user}/{len(samples)} ({has_user/len(samples)*100:.1f}%)")
    print(f"  - Assistant message: {has_assistant}/{len(samples)} ({has_assistant/len(samples)*100:.1f}%)")
    
    # Cảnh báo
    print("\n" + "=" * 60)
    print("⚠️  CẢNH BÁO")
    print("=" * 60)
    
    if missing_assistant:
        print(f"\n❌ QUAN TRỌNG: {len(missing_assistant)} samples THIẾU assistant responses!")
        print("   Dataset cần có cả user và assistant messages để training hiệu quả.")
        print("   Các dòng thiếu assistant:")
        if len(missing_assistant) <= 10:
            print(f"   {missing_assistant}")
        else:
            print(f"   {missing_assistant[:10]} ... và {len(missing_assistant) - 10} dòng khác")
        print("\n   💡 Giải pháp:")
        print("   1. Generate assistant responses bằng model base trước")
        print("   2. Sử dụng dataset khác đã có assistant responses")
        print("   3. Nếu dataset chỉ có user messages, bạn cần tạo responses trước")
    
    if has_assistant == 0:
        print("\n❌ Dataset KHÔNG CÓ assistant responses nào!")
        print("   Không thể training với dataset này.")
        return False
    
    if has_assistant < len(samples) * 0.9:
        print(f"\n⚠️  Chỉ {has_assistant}/{len(samples)} samples có assistant responses")
        print("   Nên có ít nhất 90% samples có assistant responses")
    
    if has_system == 0:
        print("\n⚠️  Dataset không có system messages")
        print("   Nên thêm system prompt để model hiểu context tốt hơn")
    
    if issues:
        print(f"\n⚠️  Có {len(issues)} vấn đề trong dataset:")
        for issue in issues[:10]:
            print(f"   - {issue}")
        if len(issues) > 10:
            print(f"   ... và {len(issues) - 10} vấn đề khác")
    
    # Kết luận
    print("\n" + "=" * 60)
    print("✅ KẾT LUẬN")
    print("=" * 60)
    
    if has_assistant == len(samples) and has_user == len(samples):
        print("✅ Dataset hợp lệ và sẵn sàng cho training!")
        return True
    elif has_assistant > 0:
        print("⚠️  Dataset có thể training nhưng không tối ưu")
        print("   Nên bổ sung assistant responses cho tất cả samples")
        return True
    else:
        print("❌ Dataset không thể training (thiếu assistant responses)")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_dataset.py <dataset_file>")
        print("Ví dụ: python check_dataset.py data/dataset_10k_qwen_user_only.jsonl")
        sys.exit(1)
    
    file_path = sys.argv[1]
    is_valid = check_dataset(file_path)
    
    sys.exit(0 if is_valid else 1)

