# Setup and Train Large COSF Transformer

**Complete step-by-step guide to train a 78M parameter COSF transformer on TinyStories.**

## Quick Start Commands

Copy and paste these commands in order:

### Step 1: Install Dependencies (2 minutes)

```bash
cd c:\Users\fatba\COSF-MI
venv\Scripts\pip install datasets sentencepiece
```

### Step 2: Download TinyStories (30-60 minutes)

```bash
cd COSF-MI

# Download 500MB of TinyStories data
..\venv\Scripts\python -m cosf_mi.data.prepare_tinystories --output data/tinystories --size-mb 500
```

**Expected output:**
```
Loading TinyStories dataset...
Loaded 2,119,719 stories
Total: X stories, ~500MB
Saved to data/tinystories/train.txt and val.txt
```

### Step 3: Train BPE Tokenizer (10-20 minutes)

```bash
# Create models directory
mkdir models

# Train tokenizer
..\venv\Scripts\python -m cosf_mi.data.tokenizer --train data/tinystories/train.txt --model-prefix models/tokenizer_8k --vocab-size 8000
```

**Expected output:**
```
Training SentencePiece bpe tokenizer...
Tokenizer trained successfully!
  Model: models/tokenizer_8k.model
  Vocab: models/tokenizer_8k.vocab
```

### Step 4: Test Setup (Optional but Recommended)

```bash
# Verify tokenizer works
..\venv\Scripts\python -c "from cosf_mi.data.tokenizer import BPETokenizer; tok = BPETokenizer('models/tokenizer_8k.model'); text = 'Once upon a time'; print(f'Test: {text}'); encoded = tok.encode(text); print(f'Tokens: {encoded}'); print(f'Decoded: {tok.decode(encoded)}')"
```

**Expected output:**
```
Test: Once upon a time
Tokens: [...]
Decoded: Once upon a time
```

### Step 5: Train Small Model (3-6 days)

```bash
# Train 78M parameter model on TinyStories
..\venv\Scripts\python -m cosf_mi.training.train_cosf_large --model-size small --data-dir data/tinystories --tokenizer models/tokenizer_8k.model --epochs 20 --device cuda --checkpoint-dir checkpoints/cosf_small
```

**This will run for ~3-6 days. You can let it run in the background.**

## What's Happening During Training

### Initial Output:
```
======================================================================
COSF LARGE TRANSFORMER TRAINING
======================================================================

Device: cuda
Loading tokenizer from: models/tokenizer_8k.model
Loaded BPE tokenizer with vocab size: 8000

Loading data from: data/tinystories
Tokenizing 500,000,000 characters...
Tokenized to 150,000,000 tokens
Train: 300,000 sequences
Val:   15,000 sequences

Creating small model...
Model parameters: 78,500,000
  Embedding: 6,144,000
  Transformer: 71,000,000
  Output: 1,356,000

Auto-set batch size: 64

======================================================================
CONFIGURATION
======================================================================
Model size: small
Parameters: 78,500,000
Vocab size: 8000
Sequence length: 512
Batch size: 64
Learning rate: 0.0003
Warmup steps: 2000
Epochs: 20
Device: cuda

======================================================================
STARTING TRAINING
======================================================================
```

### During Training:
```
Step    100 | Loss: 8.5234 | PPL:  5023.45 | LR: 0.000015
Step    200 | Loss: 7.2341 | PPL:  1379.24 | LR: 0.000030
Step    500 | Loss: 5.8923 | PPL:   361.32 | LR: 0.000075
  Validation: Loss 5.6234, PPL 275.43
Step   1000 | Loss: 4.9234 | PPL:   137.45 | LR: 0.000150

----------------------------------------------------------------------
Generated sample:
Once upon a time, there was a little girl who loved to play with her toys...
----------------------------------------------------------------------

... continues for ~3-6 days ...

Epoch 20/20 (5234.5s):
  Train Loss: 2.1234
  Val Loss:   2.3456
  Val PPL:    10.44
Saved checkpoint: checkpoints/cosf_small/checkpoint_epoch_20.pt
New best model! Val loss: 2.3456

======================================================================
TRAINING COMPLETE
======================================================================
Best validation loss: 2.3456
Best validation PPL: 10.44
```

## Monitor Training Progress

While training is running, check progress anytime:

```bash
cd COSF-MI

# Check latest checkpoints
ls -lt checkpoints/cosf_small/checkpoint_epoch_*.pt | head -5

# Check training progress (if you saved the script earlier)
..\venv\Scripts\python check_training_progress.py
```

## After Training Completes

### Generate Text from Trained Model

```bash
# Generate stories from best model
..\venv\Scripts\python -m cosf_mi.training.generate_text --checkpoint checkpoints/cosf_small/best_model.pt --length 500 --prompts "Once upon a time" "There was a little" "A long time ago"
```

### Resume Training (if needed)

```bash
# Continue training for 10 more epochs
..\venv\Scripts\python -m cosf_mi.training.train_cosf_large --model-size small --resume checkpoints/cosf_small/checkpoint_epoch_20.pt --epochs 10
```

## Training Options

### Different Model Sizes

```bash
# Tiny (16M params) - fastest, testing only
--model-size tiny

# Small (78M params) - recommended, 3-6 days
--model-size small

# Medium (143M params) - better quality, 5-12 days
--model-size medium

# Large (340M params) - best quality, 10-20 days
--model-size large
```

### Adjust Batch Size

```bash
# Smaller batch (if OOM)
--batch-size 32

# Larger batch (if you have VRAM)
--batch-size 128
```

### Adjust Learning Rate

```bash
# Conservative (safer)
--lr 1e-4

# Aggressive (faster but riskier)
--lr 5e-4
```

## Expected Results

### After 5 Epochs:
- Train PPL: ~15-25
- Val PPL: ~18-30
- Output: Simple coherent sentences

### After 10 Epochs:
- Train PPL: ~8-15
- Val PPL: ~10-20
- Output: Multi-sentence coherent paragraphs

### After 20 Epochs:
- Train PPL: ~5-10
- Val PPL: ~8-15
- Output: Full coherent stories with plot

## Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce batch size
--batch-size 32
# Or reduce sequence length
--seq-len 256
```

### Training Too Slow
```bash
# Use smaller model
--model-size tiny
# Or reduce data
--data-dir data/tinystories_small
```

### Tokenizer Not Found
```bash
# Make sure you ran Step 3
..\venv\Scripts\python -m cosf_mi.data.tokenizer --train data/tinystories/train.txt --model-prefix models/tokenizer_8k --vocab-size 8000
```

### Data Not Found
```bash
# Make sure you ran Step 2
..\venv\Scripts\python -m cosf_mi.data.prepare_tinystories --output data/tinystories --size-mb 500
```

## Full Command Reference

### All Available Arguments

```
--model-size {tiny,small,medium,large}   Model size
--max-seq-len INT                         Max sequence length (default: 512)
--data-dir PATH                           Data directory
--tokenizer PATH                          Tokenizer model path
--batch-size INT                          Batch size (auto if not set)
--seq-len INT                             Training sequence length
--epochs INT                              Number of epochs
--lr FLOAT                                Learning rate
--warmup-steps INT                        Warmup steps
--checkpoint-dir PATH                     Checkpoint directory
--resume PATH                             Resume from checkpoint
--device {cuda,cpu}                       Device
```

## Complete Training Command (All Options)

```bash
..\venv\Scripts\python -m cosf_mi.training.train_cosf_large \
    --model-size small \
    --max-seq-len 512 \
    --data-dir data/tinystories \
    --tokenizer models/tokenizer_8k.model \
    --batch-size 64 \
    --seq-len 512 \
    --epochs 20 \
    --lr 3e-4 \
    --warmup-steps 2000 \
    --checkpoint-dir checkpoints/cosf_small \
    --device cuda
```

## Next Steps After Training

1. **Evaluate quality:** Generate many samples, check coherence
2. **Fine-tune:** Train on instruction data for helpfulness
3. **Deploy:** Create API endpoint for inference
4. **Share:** Upload to HuggingFace Hub

---

**Ready to train? Start with Step 1! 🚀**
