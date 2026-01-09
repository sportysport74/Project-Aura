# Style Token Conditioning for COSF-MI

## Overview

Added style token conditioning to COSF-MI training data while preserving ALL oscillator features.

**Key principle**: This is **DATA PREPROCESSING ONLY** - no model architecture changes.

## What Changed

### Data Files Created

1. **data/tinystories/styled_train.txt**
   - Original train.txt with `[STORY] ` prepended to each story
   - Same tokenizer, same model

2. **data/tinystories/styled_val.txt**
   - Original val.txt with `[STORY] ` prepended to each story

### Scripts Created

1. **prepare_styled_tinystories.py**
   - Reads train.txt and val.txt
   - Prepends `[STORY] ` to each story
   - Saves to styled_train.txt and styled_val.txt
   - Usage: `python prepare_styled_tinystories.py`

2. **train_with_styled_data.py**
   - Modified from train_with_gentle_sleep.py
   - Uses styled_train.txt and styled_val.txt
   - Same COSF architecture (no changes)
   - Usage: `python train_with_styled_data.py --model-size tiny --epochs 3 --batch-size 8`

3. **train_styled.bat**
   - Windows batch file to run styled training
   - Usage: `train_styled.bat`

## What Did NOT Change

**All COSF features preserved:**

1. **Oscillators** - Same phase oscillator architecture
2. **Chirality** - Same learnable handedness
3. **Homeostatic sleep** - Same gentle sleep implementation
4. **Julia bounds** - Same stability constraints
5. **Phase coherence** - Same synchronization metrics

**Model architecture**: Completely unchanged

**Training logic**: Completely unchanged

**Only difference**: Input data has `[STORY] ` token at the beginning

## How It Works

### Before (Original Data)

```
Once upon a time, there was a little girl named Lily...<|endoftext|>
```

### After (Styled Data)

```
[STORY] Once upon a time, there was a little girl named Lily...<|endoftext|>
```

### Model Behavior

The COSF oscillator architecture will:
1. Tokenize `[STORY]` as a regular token
2. Process it through oscillators like any other token
3. Learn to condition its output on the style token
4. Use phase relationships to encode style information

**No special handling needed** - the model naturally learns style conditioning through its oscillatory dynamics.

## Usage

### Step 1: Prepare Styled Data

```bash
cd COSF-MI
python prepare_styled_tinystories.py
```

**Output**:
- data/tinystories/styled_train.txt
- data/tinystories/styled_val.txt

### Step 2: Train with Styled Data

**Option A: Using batch file**
```bash
train_styled.bat
```

**Option B: Using Python directly**
```bash
python train_with_styled_data.py --model-size tiny --epochs 3 --batch-size 8
```

### Step 3: Observe Style Conditioning

During training, the model will learn to:
- Associate `[STORY]` token with story-like text patterns
- Encode style in phase relationships
- Use chirality for semantic polarity within stories

## Expected Results

### Tokenizer Behavior

The tokenizer will encode `[STORY]` as a token sequence:
- If `[STORY]` is in vocabulary: single token
- If not: multiple subword tokens (e.g., `[`, `ST`, `ORY`, `]`)

**No tokenizer changes needed** - BPE handles it automatically.

### Model Learning

The COSF oscillators will:
1. **Phase alignment**: `[STORY]` tokens will synchronize with story-like patterns
2. **Frequency adaptation**: Story tokens will converge to harmonic frequencies
3. **Chirality differentiation**: Positive vs negative story elements will learn opposite handedness

### Performance

**Expected behavior**:
- Similar training loss to non-styled version
- Model learns implicit style conditioning
- Can later add more style tokens (e.g., `[POEM]`, `[DIALOG]`)

## Future Extensions

### Adding More Styles

Create multiple styled datasets:

```python
# prepare_multi_style.py
styles = {
    'train.txt': '[STORY]',
    'poetry.txt': '[POEM]',
    'dialog.txt': '[DIALOG]',
}

for file, token in styles.items():
    prepare_styled_data(file, token)
```

### Style Interpolation

With multiple style tokens, you could:
1. Train on mixed dataset
2. At inference, use different style tokens
3. Observe phase coherence differences between styles

### Style Transfer

Potential future experiment:
1. Train on `[STORY]` data
2. At inference, replace `[STORY]` with `[POEM]`
3. Observe if oscillator dynamics shift toward poetic patterns

## Files Modified

### New Files

1. `prepare_styled_tinystories.py` - Data preprocessing script
2. `train_with_styled_data.py` - Training script for styled data
3. `train_styled.bat` - Batch file for easy execution
4. `data/tinystories/styled_train.txt` - Styled training data
5. `data/tinystories/styled_val.txt` - Styled validation data
6. `STYLE_TOKEN_CONDITIONING.md` - This document

### Unchanged Files

- All COSF architecture files (`cosf_oscillator.py`, etc.)
- All training utilities (`sleep_cycle.py`, etc.)
- Tokenizer files
- Original data files (train.txt, val.txt)

## Technical Details

### Data Format

**Input**: TinyStories text file with `<|endoftext|>` separators

```
Story 1<|endoftext|>Story 2<|endoftext|>Story 3<|endoftext|>
```

**Output**: Same format with style token prepended

```
[STORY] Story 1<|endoftext|>[STORY] Story 2<|endoftext|>[STORY] Story 3<|endoftext|>
```

### Tokenization

The tokenizer processes styled text normally:

```python
# Example tokenization
tokenizer.encode("[STORY] Once upon a time...")
# Returns: [token_ids for entire sequence]
```

**No special handling** - style token is just another token.

### Training

Training loop is identical to non-styled version:

```python
for batch in train_loader:
    inputs, targets = batch
    logits, loss = model(inputs, targets)
    # inputs already contains [STORY] token at position 0
```

**No code changes** - model processes styled data naturally.

## Comparison to Original Training

### Original (train_with_gentle_sleep.py)

- Uses train.txt and val.txt
- No style tokens
- Saves to checkpoints/gentle_sleep/

### Styled (train_with_styled_data.py)

- Uses styled_train.txt and styled_val.txt
- Has [STORY] token at beginning
- Saves to checkpoints/styled_cosf/

**Everything else identical**: same model, same training loop, same sleep logic.

## Verification

### Check Data Preparation

```bash
python -c "
with open('data/tinystories/styled_train.txt', 'r') as f:
    stories = f.read().split('<|endoftext|>')
    print(f'Stories: {len(stories)}')
    print(f'First story starts with [STORY]: {stories[0].strip().startswith(\"[STORY] \")}')"
```

### Check Training

Run training for a few steps and verify:
1. Loss decreases normally
2. No tokenizer errors
3. Chirality metrics are healthy
4. Sleep cycles work as expected

## Conclusion

**Style token conditioning added successfully!**

- ✅ Data preprocessing complete
- ✅ Training script updated
- ✅ All COSF features preserved
- ✅ No architecture changes
- ✅ Ready to train

**Next steps**:
1. Run `train_styled.bat` to train with styled data
2. Compare performance to non-styled version
3. Analyze how oscillators encode style information
4. Consider adding more style tokens for multi-style training

---

**Date**: 2026-01-05
**Author**: COSF-MI Development Team
**Status**: Ready for training
