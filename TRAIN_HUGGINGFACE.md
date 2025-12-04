# 🚀 Hướng Dẫn Train Model và Upload lên Hugging Face

## ⚡ Quick Start

```bash
# 1. Kiểm tra dataset
python check_dataset.py data/dataset_10k_qwen_user_only.jsonl

# 2. Generate responses (nếu dataset thiếu assistant responses)
python generate_responses.py \
  --input data/dataset_10k_qwen_user_only.jsonl \
  --output data/dataset_with_responses.jsonl \
  --model hainguyen306201/bank-model-2b

# 3. Train và upload
python train_qwen3_model.py \
  --model hainguyen306201/bank-model-2b \
  --dataset data/dataset_with_responses.jsonl \
  --push-to-hub \
  --hub-model-id "username/bank-model-finetuned"
```

---

## 📋 Các Bước Chi Tiết

### Bước 1: Kiểm tra Dataset

Dataset hiện tại (`data/dataset_10k_qwen_user_only.jsonl`) **chỉ có user messages**, không có assistant responses. 

**Kiểm tra:**
```bash
python check_dataset.py data/dataset_10k_qwen_user_only.jsonl
```

**Kết quả:** Dataset sẽ báo thiếu assistant responses.

### Bước 2: Generate Assistant Responses

Vì dataset thiếu responses, bạn cần generate trước:

```bash
python generate_responses.py \
  --input data/dataset_10k_qwen_user_only.jsonl \
  --output data/dataset_10k_with_responses.jsonl \
  --model hainguyen306201/bank-model-2b \
  --max-new-tokens 512
```

**Kiểm tra lại:**
```bash
python check_dataset.py data/dataset_10k_with_responses.jsonl
```

### Bước 3: Cài đặt Dependencies

```bash
pip install -r requirements.txt
```

### Bước 4: Login Hugging Face

```bash
huggingface-cli login
# Nhập token của bạn
```

Hoặc set environment variable:
```bash
export HF_TOKEN="your_token_here"
```

### Bước 5: Train Model

**Cách 1: Train cơ bản (không upload)**
```bash
python train_qwen3_model.py \
  --model hainguyen306201/bank-model-2b \
  --dataset data/dataset_10k_with_responses.jsonl \
  --output ./checkpoints \
  --epochs 3 \
  --batch-size 4
```

**Cách 2: Train và upload tự động**
```bash
python train_qwen3_model.py \
  --model hainguyen306201/bank-model-2b \
  --dataset data/dataset_10k_with_responses.jsonl \
  --output ./checkpoints \
  --epochs 3 \
  --batch-size 4 \
  --push-to-hub \
  --hub-model-id "username/bank-model-finetuned"
```

**Cách 3: Sử dụng shell script**
```bash
./train_and_upload.sh \
  --hub-model-id "username/bank-model-finetuned" \
  --epochs 3 \
  --batch-size 4
```

### Bước 6: Upload thủ công (nếu chưa upload tự động)

```bash
huggingface-cli upload username/bank-model-finetuned ./checkpoints --repo-type model
```

---

## ⚙️ Các Tham Số Training

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `--model` | Model base | `hainguyen306201/bank-model-2b` |
| `--dataset` | Đường dẫn dataset | Bắt buộc |
| `--output` | Thư mục lưu checkpoints | `./checkpoints` |
| `--epochs` | Số epochs | `3` |
| `--batch-size` | Batch size | `4` |
| `--learning-rate` | Learning rate | `2e-4` |
| `--push-to-hub` | Upload lên HF sau training | `False` |
| `--hub-model-id` | Tên model trên HF | Bắt buộc nếu dùng `--push-to-hub` |
| `--hub-token` | HF token (optional) | Dùng token đã login |
| `--hub-private` | Tạo private repo | `False` |

---

## 🐛 Troubleshooting

### Dataset thiếu assistant responses
**Giải pháp:** Chạy `generate_responses.py` trước khi training

### Out of Memory
**Giải pháp:**
- Giảm `--batch-size` (từ 4 xuống 2)
- Tăng `--gradient-accumulation` (từ 1 lên 4)
- Giảm `--max-length` (từ 2048 xuống 1024)

### Lỗi Hugging Face authentication
**Giải pháp:**
```bash
huggingface-cli login
```

---

## 📊 Monitor Training

Xem logs bằng TensorBoard:
```bash
tensorboard --logdir ./checkpoints/runs
```

---

## ✅ Checklist

- [ ] Đã cài đặt dependencies (`pip install -r requirements.txt`)
- [ ] Đã kiểm tra dataset (`python check_dataset.py`)
- [ ] Đã generate responses nếu cần (`python generate_responses.py`)
- [ ] Đã login Hugging Face (`huggingface-cli login`)
- [ ] Đã train model (`python train_qwen3_model.py`)
- [ ] Đã upload lên Hugging Face (tự động hoặc thủ công)
- [ ] Đã test model sau training

---

## 🔗 Tài Liệu

- [Qwen3 Docs](https://qwen.readthedocs.io/en/latest/)
- [Hugging Face Hub](https://huggingface.co/docs/hub/index)

---

Chúc bạn training thành công! 🎉

