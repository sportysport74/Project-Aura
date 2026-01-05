# COSF Transformer: Path to Deployment

**Goal:** Build a useful, deployable COSF-native language model trained on your RTX 5090.

## Current State ✅

- **Model:** 9.65M parameters, character-level
- **Data:** Shakespeare (1.1MB)
- **Performance:** PPL ~1.9 on Shakespeare
- **Hardware:** RTX 5090 (32GB VRAM), CUDA 12.8
- **Status:** Currently training epoch 11-30 with PPL ~1.9

## Roadmap to Useful AI

### Phase 1: Better Data 📚 (START HERE)

**Goal:** Move beyond Shakespeare to diverse, useful text.

#### Option A: TinyStories (Recommended for First Scale-Up)
- **What:** Simple stories dataset, 2.1M stories
- **Size:** ~500MB-2GB of text
- **Why:** Perfect for 50-100M param models, clean data, good for storytelling/generation
- **Time:** 2-3 hours to download and prepare

```bash
# Install dependencies
cd COSF-MI
../venv/Scripts/pip install datasets sentencepiece

# Download and prepare TinyStories
../venv/Scripts/python -m cosf_mi.data.prepare_tinystories \
    --output data/tinystories \
    --size-mb 500

# Train BPE tokenizer (8K vocab)
../venv/Scripts/python -m cosf_mi.data.tokenizer \
    --train data/tinystories/train.txt \
    --model-prefix models/tokenizer_8k \
    --vocab-size 8000
```

#### Option B: OpenWebText
- **What:** Web pages, diverse topics
- **Size:** 40GB (can use subset)
- **Why:** General knowledge, more challenging
- **Time:** Longer download, more preprocessing needed

#### Option C: Mix (Best for Production)
- TinyStories (50%) - storytelling ability
- Wikipedia Simple (30%) - factual knowledge
- Instruction data (20%) - helpfulness

### Phase 2: Scale Up the Model 🚀

**Current:** 9.65M params → **Target:** 50-100M params

#### Small Model (50M params) - Quick Iteration
```python
# Model: small
- d_model: 768
- layers: 12 (Fibonacci)
- heads: 12
- vocab: 8000 (BPE)
- seq_len: 512
- params: ~50M

# Training estimate (TinyStories 500MB):
- Batch size: 64
- Time: ~3-5 hours per epoch on RTX 5090
- Epochs: 20-30 for good performance
- Total: 60-150 hours (~3-6 days)
```

#### Medium Model (100M params) - Best Performance
```python
# Model: medium
- d_model: 1024
- layers: 13 (Fibonacci)
- heads: 13
- vocab: 8000 (BPE)
- seq_len: 512
- params: ~100M

# Training estimate (TinyStories 1GB):
- Batch size: 32
- Time: ~6-10 hours per epoch on RTX 5090
- Epochs: 20-30
- Total: 120-300 hours (~5-12 days)
```

### Phase 3: Efficient Training 🔧

#### Required Changes:
1. **BPE Tokenization** ✅ (already implemented)
   - 8K-32K vocab instead of 256
   - ~3-5x more efficient
   - Better generalization

2. **Data Loading**
   - Update `CharDataset` → `TokenDataset`
   - Handle BPE tokens instead of characters
   - Add data caching for speed

3. **Training Script Updates**
   - Support model size selection (`--model-size small`)
   - Automatic batch size based on model
   - Better checkpoint management
   - Optional: W&B or TensorBoard logging

4. **Resume Training** ✅ (already implemented)
   - Can stop/start anytime
   - Preserves optimizer state
   - Continue from any checkpoint

### Phase 4: Training Plan ⏱️

#### Recommended Sequence:

**Week 1: Data & Tokenizer**
- [ ] Download TinyStories (500MB-1GB)
- [ ] Train 8K BPE tokenizer
- [ ] Verify data pipeline works
- **Time:** 1 day

**Week 2-3: Small Model Training**
- [ ] Train 50M param model on TinyStories
- [ ] 20-30 epochs
- [ ] Monitor perplexity, generate samples
- **Time:** 3-6 days (can run overnight)

**Week 4-5: Medium Model Training** (if small model works well)
- [ ] Train 100M param model
- [ ] Longer sequences (1024 tokens)
- [ ] Better performance
- **Time:** 5-12 days

**Week 6: Evaluation & Fine-Tuning**
- [ ] Generate sample outputs
- [ ] Test on various prompts
- [ ] Optional: Fine-tune on instruction data
- **Time:** 2-3 days

### Phase 5: Fine-Tuning for Helpfulness 🎯

After base training, fine-tune on instruction data to make it helpful.

#### Option A: Instruction Following
```
Download Alpaca or Dolly instruction dataset
Fine-tune for 1-3 epochs
Test: "Explain the golden ratio"
```

#### Option B: Conversational
```
Download conversation dataset (OpenAssistant subset)
Fine-tune for helpfulness
Test: Multi-turn conversations
```

### Phase 6: Deployment 🌐

#### Local Deployment:
```bash
# Simple Flask API
from flask import Flask, request, jsonify
from cosf_mi.models.cosf_transformer import COSFTransformer
from cosf_mi.data.tokenizer import Tokenizer

app = Flask(__name__)
model = COSFTransformer.from_pretrained("models/cosf_100m_best.pt")
tokenizer = Tokenizer.from_pretrained("models/tokenizer_8k.json")

@app.route("/generate", methods=["POST"])
def generate():
    prompt = request.json["prompt"]
    # ... generation logic
    return jsonify({"text": generated_text})
```

#### Cloud Deployment:
- Docker container with model
- Deploy to AWS/GCP/Azure
- Or: HuggingFace Spaces (free tier!)

### Phase 7: Optimization (Optional) ⚡

- **Quantization:** INT8 or INT4 for faster inference
- **ONNX Export:** For deployment flexibility
- **Distillation:** Train smaller model from large model
- **Pruning:** Remove unnecessary parameters

## Implementation Steps (This Week)

### Step 1: Install Dependencies
```bash
cd COSF-MI
../venv/Scripts/pip install datasets sentencepiece wandb
```

### Step 2: Download Data
```bash
# TinyStories (recommended)
../venv/Scripts/python -m cosf_mi.data.prepare_tinystories \
    --output data/tinystories \
    --size-mb 500 \
    --val-split 0.05
```

### Step 3: Train Tokenizer
```bash
../venv/Scripts/python -m cosf_mi.data.tokenizer \
    --train data/tinystories/train.txt \
    --model-prefix models/tokenizer_8k \
    --vocab-size 8000
```

### Step 4: Test Model Configs
```bash
# See all available model sizes
cd COSF-MI
../venv/Scripts/python -m cosf_mi.models.model_configs
```

### Step 5: Create Training Script for BPE
Need to create `train_cosf_large.py` that:
- Uses BPE tokenizer instead of char
- Supports model size selection
- Works with TinyStories data
- Has all the resume/logging features

### Step 6: Start Training
```bash
# Train small model (50M params)
../venv/Scripts/python -m cosf_mi.training.train_cosf_large \
    --model-size small \
    --data data/tinystories \
    --tokenizer models/tokenizer_8k.model \
    --epochs 20 \
    --device cuda

# Takes ~3-6 days on RTX 5090
```

## Expected Outcomes

### After Small Model (50M params, 20 epochs):
- **PPL:** 10-15 on TinyStories
- **Capability:** Generate coherent short stories
- **Quality:** Better than current Shakespeare model
- **Example:**
  ```
  Prompt: "Once upon a time, there was a"
  Output: "little girl named Emma who loved to play with her toys.
          One day, she found a shiny red ball in the garden..."
  ```

### After Medium Model (100M params, 30 epochs):
- **PPL:** 5-8 on TinyStories
- **Capability:** High-quality story generation, basic reasoning
- **Quality:** Competitive with GPT-2 small on similar data
- **Example:**
  ```
  Prompt: "Write a story about friendship"
  Output: [Coherent 200-300 word story with plot, characters, resolution]
  ```

### After Fine-Tuning:
- **PPL:** N/A (task-specific)
- **Capability:** Answer questions, follow instructions, helpful responses
- **Quality:** Actually useful for simple tasks
- **Example:**
  ```
  Prompt: "Explain why the sky is blue in simple terms."
  Output: "The sky appears blue because of how sunlight interacts
          with the Earth's atmosphere. When sunlight enters the
          atmosphere, blue light scatters more than other colors..."
  ```

## Cost & Time Estimates

### Hardware: RTX 5090
- **Cost:** Already have it! 🎉
- **Power:** ~450W × 24hr × electricity rate
- **Estimate:** ~$2-5/day in electricity (US average)

### Time Investment:
- **Data prep:** 1 day
- **Small model:** 3-6 days training
- **Medium model:** 5-12 days training
- **Fine-tuning:** 1-3 days
- **Total:** 2-4 weeks to useful model

### Comparison to Commercial Training:
- **AWS p4d.24xlarge (8×A100):** $32/hour
- **Small model training:** 100 hours = $3,200
- **You:** Electricity + time = ~$50-100
- **Savings:** ~$3,000+ per model!

## Success Metrics

### Phase 1 (Data + Tokenizer):
- ✅ TinyStories downloaded
- ✅ 8K BPE tokenizer trained
- ✅ Can encode/decode text correctly

### Phase 2 (Small Model):
- ✅ Training loss decreasing
- ✅ PPL < 20 on validation
- ✅ Generated text is coherent

### Phase 3 (Medium Model):
- ✅ PPL < 10 on validation
- ✅ Can generate multi-paragraph stories
- ✅ Follows prompts reasonably well

### Phase 4 (Deployment Ready):
- ✅ Model responds helpfully to instructions
- ✅ Inference < 1 second for 100 tokens
- ✅ Can deploy as API

## Next Steps (Action Items)

1. **Today:**
   - [x] Install datasets and sentencepiece
   - [ ] Download TinyStories (500MB)
   - [ ] Train 8K BPE tokenizer

2. **This Week:**
   - [ ] Create `TokenDataset` class for BPE tokens
   - [ ] Create `train_cosf_large.py` with model size support
   - [ ] Test training on small data sample
   - [ ] Start small model training (3-6 days)

3. **Next Week:**
   - [ ] Monitor small model training
   - [ ] Generate samples, evaluate quality
   - [ ] Decide: continue with medium model or fine-tune small

4. **Week 3-4:**
   - [ ] Scale to medium model if desired
   - [ ] Prepare instruction/conversation data for fine-tuning
   - [ ] Fine-tune best model

5. **Week 4+:**
   - [ ] Create deployment API
   - [ ] Test with real prompts
   - [ ] Deploy locally or to cloud
   - [ ] **Ship it!** 🚀

## Files Created

- ✅ `src/python/cosf_mi/data/prepare_tinystories.py` - Data downloader
- ✅ `src/python/cosf_mi/data/tokenizer.py` - BPE tokenizer
- ✅ `src/python/cosf_mi/models/model_configs.py` - Model size presets
- ⏳ `src/python/cosf_mi/training/train_cosf_large.py` - Large model trainer (TODO)
- ⏳ `src/python/cosf_mi/training/token_dataset.py` - BPE dataset (TODO)

## Questions?

- **Why TinyStories?** Clean, simple, perfect for 50-100M models. Proven to work well.
- **Why 8K vocab?** Sweet spot for small models. Not too big, covers most words.
- **Can I use bigger models?** Yes! But >200M params might be slower on single RTX 5090.
- **What about GPT-4 level?** Need ~100B+ params, not feasible on single GPU.
- **Is COSF framework worth it?** Yes! φ-scaling and Julia bounds provide unique benefits.

---

**Ready to build a useful COSF AI? Let's start with Phase 1! 🚀**
