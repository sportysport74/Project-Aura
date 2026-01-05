# Text Generation with COSF Transformer

This guide shows how to generate text from your trained COSF Transformer model.

## Quick Start

Generate text from the best trained model:

```bash
cd COSF-MI
../venv/Scripts/python -m cosf_mi.training.generate_text
```

This will generate 500 characters from 10 different prompts using the best checkpoint.

## Usage Options

### Generate 500 Characters (as requested)

```bash
# Use default prompts (JULIET, KING, etc.)
../venv/Scripts/python -m cosf_mi.training.generate_text --length 500

# Use custom prompts
../venv/Scripts/python -m cosf_mi.training.generate_text \
    --length 500 \
    --prompts "JULIET:" "KING:" "To be or not to be"
```

### Adjust Temperature

Temperature controls randomness (0.1 = conservative, 2.0 = very random):

```bash
# More conservative/coherent (0.5-0.7)
../venv/Scripts/python -m cosf_mi.training.generate_text --temperature 0.6

# More creative/random (0.8-1.2)
../venv/Scripts/python -m cosf_mi.training.generate_text --temperature 1.0

# Very experimental (1.5+)
../venv/Scripts/python -m cosf_mi.training.generate_text --temperature 1.5
```

### Use Different Checkpoint

```bash
# Use a specific epoch checkpoint
../venv/Scripts/python -m cosf_mi.training.generate_text \
    --checkpoint checkpoints/cosf_transformer/checkpoint_epoch_10.pt

# Use best model (default)
../venv/Scripts/python -m cosf_mi.training.generate_text \
    --checkpoint checkpoints/cosf_transformer/best_model.pt
```

### Custom Prompts

```bash
# Single custom prompt
../venv/Scripts/python -m cosf_mi.training.generate_text \
    --prompts "Once upon a time"

# Multiple custom prompts
../venv/Scripts/python -m cosf_mi.training.generate_text \
    --prompts "JULIET:" "ROMEO:" "HAMLET:" "MACBETH:"

# Shakespeare-style prompts
../venv/Scripts/python -m cosf_mi.training.generate_text \
    --prompts "To be or not to be" "What light through yonder window" "Friends, Romans, countrymen"
```

## Default Prompts

The script includes 10 predefined Shakespeare-inspired prompts:

1. `JULIET:\n`
2. `KING:\n`
3. `To be or not to be`
4. `ROMEO:\n`
5. `What light through yonder window breaks?`
6. `Friends, Romans, countrymen,`
7. `HAMLET:\n`
8. `MACBETH:\n`
9. `The quality of mercy`
10. `All the world's a stage`

## Full Options

```bash
../venv/Scripts/python -m cosf_mi.training.generate_text --help
```

Available arguments:
- `--checkpoint`: Path to checkpoint file (default: `checkpoints/cosf_transformer/best_model.pt`)
- `--length`: Characters to generate per prompt (default: 500)
- `--temperature`: Sampling temperature 0.1-2.0 (default: 0.8)
- `--device`: cuda or cpu (default: cuda if available)
- `--prompts`: Custom prompts to use (default: predefined Shakespeare prompts)

## Example Output

With validation loss of 2.97 after 1 epoch, the model generates Shakespeare-like text:

```
JULIET:
O, for, sir, for, crawit, tield,
Upes, tise therere, tinke! Miroug�o$es Hes, f'cur�e Mat,
Colad, Chis f here, weayore, Rind He, Mareat:
Weshare, hare ct'st'tis, be waprelatinge!
```

**Note**: After just 1 epoch, the model has learned:
- Character-level patterns
- Word-like structures
- Punctuation placement
- Character name formats (e.g., "JULIET:", "KING:")
- Basic Shakespeare-style vocabulary patterns

With more training (10+ epochs), the text becomes significantly more coherent!

## Tips

1. **Lower temperature (0.5-0.7)** for more coherent but repetitive text
2. **Medium temperature (0.8-1.0)** for balanced creativity and coherence
3. **Higher temperature (1.2-2.0)** for experimental/creative outputs
4. The model is trained on **character-level** Shakespeare, so expect archaic language patterns
5. Longer training (more epochs) produces much better results

## Training Status

Current best model:
- Validation loss: **2.9669**
- Validation perplexity: **~19.5**
- Training: **1 epoch** complete
- Model size: **9.6M parameters**
- Device: **RTX 5090 (CUDA)**

Continue training for better results:
```bash
cd COSF-MI
../venv/Scripts/python -m cosf_mi.training.train_cosf_transformer
```
