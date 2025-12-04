# Qwen3 Dataset Builder & Training Scripts

Hướng dẫn sử dụng các script Python để tạo dataset và training Qwen3 model.

## 📋 Yêu cầu

```bash
pip install -r requirements.txt
```

## 📝 1. Tạo Dataset (qwen3_dataset_builder.py)

Script để tạo và chuẩn bị dataset theo chuẩn Qwen3.

### Cài đặt

```bash
# Cài dependencies
pip install transformers datasets pandas tqdm
```

### Sử dụng

#### Load từ CSV và export ra JSON

```bash
python qwen3_dataset_builder.py \
    --input data.csv \
    --output dataset.json
```

#### Load từ JSON với format Q&A

```bash
python qwen3_dataset_builder.py \
    --input data.json \
    --format qa \
    --output dataset.jsonl
```

#### Load từ JSONL và chia train/test

```bash
python qwen3_dataset_builder.py \
    --input data.jsonl \
    --format jsonl \
    --output dataset.json \
    --split 0.2
```

#### Preview dataset

```bash
python qwen3_dataset_builder.py \
    --input data.json \
    --preview 5
```

### Các format hỗ trợ

1. **CSV**: File CSV với columns `question` và `answer`
2. **JSON (Q&A)**: `[{"question": "...", "answer": "..."}]`
3. **JSON (Qwen3)**: `[{"messages": [...]}]`
4. **JSONL**: Mỗi dòng là một JSON conversation

### Format chuẩn Qwen3

```json
{
  "messages": [
    {
      "role": "user",
      "content": [{"type": "text", "text": "Câu hỏi"}]
    },
    {
      "role": "assistant",
      "content": [{"type": "text", "text": "Câu trả lời"}]
    }
  ]
}
```

### Options

```
--input, -i          Đường dẫn file input (required)
--output, -o         Đường dẫn file output (optional)
--format, -f         Format input (auto, qwen3, qa, conversation, csv, jsonl)
--question-col       Tên cột question (cho CSV, default: question)
--answer-col         Tên cột answer (cho CSV, default: answer)
--split              Tỷ lệ test set (0.0-1.0)
--preview, -p        Số samples để preview
--include-image      Hỗ trợ image (cho Qwen3VL)
--seed               Random seed (default: 42)
```

## 🚀 2. Training Model (train_qwen3_model.py)

Script để fine-tune Qwen3 model với LoRA.

### Cài đặt

```bash
# Cài dependencies
pip install transformers accelerate peft bitsandbytes datasets torch
```

### Sử dụng

#### Training cơ bản

```bash
python train_qwen3_model.py \
    --model hainguyen306201/bank-model-2b \
    --dataset dataset.json \
    --output ./checkpoints
```

#### Training với custom parameters

```bash
python train_qwen3_model.py \
    --model hainguyen306201/bank-model-2b \
    --dataset dataset.jsonl \
    --output ./checkpoints \
    --epochs 5 \
    --batch-size 8 \
    --learning-rate 2e-4
```

#### Training với gradient accumulation

```bash
python train_qwen3_model.py \
    --model hainguyen306201/bank-model-2b \
    --dataset dataset.json \
    --batch-size 4 \
    --gradient-accumulation 2 \
    --epochs 3
```

#### Training không quantization (cần GPU lớn)

```bash
python train_qwen3_model.py \
    --model hainguyen306201/bank-model-2b \
    --dataset dataset.json \
    --no-quantization \
    --batch-size 2
```

#### Training với Flash Attention 2 (tăng tốc)

```bash
# Cài flash-attn trước
pip install flash-attn --no-build-isolation

# Training
python train_qwen3_model.py \
    --model hainguyen306201/bank-model-2b \
    --dataset dataset.json \
    --flash-attention
```

### Options

```
--model, -m              Tên model trên Hugging Face (required)
--dataset, -d            Đường dẫn file dataset (required)
--output, -o              Thư mục lưu checkpoints (default: ./checkpoints)
--epochs, -e              Số epochs (default: 3)
--batch-size, -b          Batch size (default: 4)
--gradient-accumulation   Gradient accumulation steps (default: 1)
--learning-rate, -lr      Learning rate (default: 2e-4)
--lora-r                  LoRA rank (default: 16)
--lora-alpha              LoRA alpha (default: 32)
--lora-dropout            LoRA dropout (default: 0.05)
--no-quantization         Không dùng quantization
--flash-attention         Sử dụng Flash Attention 2
--save-steps              Lưu checkpoint mỗi N steps (default: 500)
--logging-steps           Log mỗi N steps (default: 50)
--max-length              Max sequence length (default: 2048)
--warmup-steps            Warmup steps (default: 100)
```

## 📊 Workflow hoàn chỉnh

### Bước 1: Chuẩn bị data

```bash
# Tạo dataset từ CSV
python qwen3_dataset_builder.py \
    --input raw_data.csv \
    --output training_data.jsonl \
    --split 0.2
```

Kết quả:
- `training_data.jsonl.train.jsonl` - Train set
- `training_data.jsonl.test.jsonl` - Test set

### Bước 2: Training

```bash
# Training với train set
python train_qwen3_model.py \
    --model hainguyen306201/bank-model-2b \
    --dataset training_data.jsonl.train.jsonl \
    --output ./checkpoints \
    --epochs 5 \
    --batch-size 4
```

### Bước 3: Evaluate (optional)

Sử dụng test set để đánh giá model sau khi training.

## 💡 Tips

1. **Dataset size**: Nên có ít nhất 100-500 samples để training hiệu quả
2. **Batch size**: Điều chỉnh theo GPU memory (4-bit quantization: batch_size=4-8)
3. **Learning rate**: Bắt đầu với 2e-4, điều chỉnh nếu loss không giảm
4. **LoRA rank**: Tăng `--lora-r` (16→32→64) nếu cần chất lượng cao hơn
5. **Flash Attention**: Cài `flash-attn` để tăng tốc ~20-30%

## 📚 Tài liệu tham khảo

- [Qwen Documentation](https://qwen.readthedocs.io/en/latest/)
- [PEFT LoRA](https://huggingface.co/docs/peft/task_guides/clm-lora)
- [Transformers Training](https://huggingface.co/docs/transformers/training)

## ⚠️ Lưu ý

- Cần GPU với ít nhất 16GB VRAM (với 4-bit quantization)
- Training có thể mất vài giờ tùy dataset size
- Backup checkpoints thường xuyên
- Monitor GPU memory và temperature

