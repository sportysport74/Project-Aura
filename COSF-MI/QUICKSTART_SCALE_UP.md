# Quick Start: Scale Up to Useful COSF Transformer

**Goal:** Train a 50M or 100M parameter COSF transformer on real data in the next few days.

## Model Sizes Available

| Model  | Params  | d_model | Layers | Training Time (10 epochs) |
|--------|---------|---------|--------|---------------------------|
| tiny   | 16M     | 384     | 7      | ~2 hours                  |
| small  | 78M     | 768     | 12     | ~30-60 hours              |
| medium | 143M    | 1024    | 13     | ~60-150 hours             |
| large  | 340M    | 1280    | 21     | ~200-400 hours            |

**Recommended:** Start with `small` (78M) for best iteration speed.

## Step-by-Step Guide

### Step 1: Install Dependencies (5 minutes)

```bash
cd c:\Users\fatba\COSF-MI
venv\Scripts\activate
pip install datasets sentencepiece
```

### Step 2: Download TinyStories (30-60 minutes)

```bash
cd COSF-MI

# Download TinyStories dataset (500MB recommended)
..\venv\Scripts\python -m cosf_mi.data.prepare_tinystories \
    --output data/tinystories \
    --size-mb 500 \
    --val-split 0.05
```

**Expected output:**
- `data/tinystories/train.txt` (~475MB)
- `data/tinystories/val.txt` (~25MB)
- `data/tinystories/metadata.json`

### Step 3: Train BPE Tokenizer (10-20 minutes)

```bash
# Create models directory
mkdir models

# Train 8K vocabulary BPE tokenizer
..\venv\Scripts\python -m cosf_mi.data.tokenizer \
    --train data/tinystories/train.txt \
    --model-prefix models/tokenizer_8k \
    --vocab-size 8000
```

**Expected output:**
- `models/tokenizer_8k.model`
- `models/tokenizer_8k.vocab`
- `models/tokenizer_8k.json`

### Step 4: Create Training Script (TODO)

We need to create a new training script that uses BPE tokens instead of characters.

**What needs to be done:**
1. Create `TokenDataset` class (like `CharDataset` but for BPE)
2. Create `train_cosf_large.py` with:
   - Model size selection (`--model-size small`)
   - BPE tokenizer support
   - TinyStories data loading
   - All resume/checkpoint features

### Step 5: Train the Model (2-7 days)

```bash
# Train small model (78M params) on TinyStories
..\venv\Scripts\python -m cosf_mi.training.train_cosf_large \
    --model-size small \
    --data data/tinystories \
    --tokenizer models/tokenizer_8k.model \
    --epochs 20 \
    --batch-size 64 \
    --device cuda \
    --checkpoint-dir checkpoints/cosf_small

# Expected time: ~30-60 hours (2-3 days)
```

## Immediate Next Steps

### TODAY:

1. **Install dependencies:**
   ```bash
   pip install datasets sentencepiece
   ```

2. **Download TinyStories:**
   ```bash
   cd COSF-MI
   ..\venv\Scripts\python -m cosf_mi.data.prepare_tinystories \
       --output data/tinystories \
       --size-mb 500
   ```

3. **Train tokenizer:**
   ```bash
   ..\venv\Scripts\python -m cosf_mi.data.tokenizer \
       --train data/tinystories/train.txt \
       --model-prefix models/tokenizer_8k \
       --vocab-size 8000
   ```

### THIS WEEK:

4. **Create TokenDataset class** - Modify `CharDataset` to use BPE tokens

5. **Create train_cosf_large.py** - New training script with:
   - Model size selection
   - BPE tokenizer integration
   - Better data loading

6. **Start training!** - Launch small model training

## What You'll Get

After training the **small** model (78M params) for 20 epochs on TinyStories:

**Example Generation:**
```
Prompt: "Once upon a time, there was a little"

Current (9M char model on Shakespeare):
"Once upon a time, there was a little the the the mare..."

Expected (78M BPE model on TinyStories):
"Once upon a time, there was a little girl named Emma who loved
to play with her toys. One day, she went to the park with her
mom and saw a big, colorful playground. Emma ran to the swings
and started swinging higher and higher..."
```

**Metrics:**
- Training PPL: ~5-10
- Validation PPL: ~8-15
- Coherent multi-sentence stories
- Follows simple prompts
- Actually useful for story generation

## Hardware Requirements

✅ **You have:** RTX 5090 (32GB VRAM)

**Memory usage estimates:**
- Small (78M): ~8-12GB VRAM with batch_size=64
- Medium (143M): ~15-20GB VRAM with batch_size=32
- Large (340M): ~25-30GB VRAM with batch_size=16

**You're good to go for all models!**

## Cost Estimate

**Electricity only** (RTX 5090 @ 450W):
- Small model training (60 hours): ~27 kWh = $3-5
- Medium model training (150 hours): ~68 kWh = $8-12
- **vs. Cloud (AWS p4d):** $32/hour × 60h = $1,920

**Savings:** ~$1,900+ per model! 💰

## Files Already Created

✅ `src/python/cosf_mi/data/prepare_tinystories.py` - Data downloader
✅ `src/python/cosf_mi/data/tokenizer.py` - BPE tokenizer
✅ `src/python/cosf_mi/models/model_configs.py` - Model size presets
✅ `DEPLOYMENT_ROADMAP.md` - Full roadmap

⏳ `src/python/cosf_mi/data/token_dataset.py` - BPE dataset (TODO)
⏳ `src/python/cosf_mi/training/train_cosf_large.py` - Large model trainer (TODO)

## Ready to Start?

**Run these commands NOW:**

```bash
cd c:\Users\fatba\COSF-MI\COSF-MI

# 1. Install deps
..\venv\Scripts\pip install datasets sentencepiece

# 2. Download data
..\venv\Scripts\python -m cosf_mi.data.prepare_tinystories \
    --output data/tinystories \
    --size-mb 500

# 3. Train tokenizer
..\venv\Scripts\python -m cosf_mi.data.tokenizer \
    --train data/tinystories/train.txt \
    --model-prefix models/tokenizer_8k \
    --vocab-size 8000
```

**After that, I'll help you create the training script and launch training!** 🚀
