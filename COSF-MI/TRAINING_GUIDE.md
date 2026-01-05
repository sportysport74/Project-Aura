# COSF Transformer Training Guide

Complete guide to training the COSF-native transformer on character-level text.

## Quick Start

```bash
# Train with default settings (tiny-shakespeare, 10 epochs)
cd COSF-MI
python -m cosf_mi.training.train_cosf_transformer
```

The script will:
1. Auto-download tiny-shakespeare if not found
2. Train for 10 epochs (~30-60 minutes on RTX 5090)
3. Save checkpoints to `checkpoints/cosf_transformer/`
4. Generate sample text every 500 steps
5. Track COSF-specific metrics (Julia bounds, φ-shells)

## Configuration

All training parameters are in `TrainConfig` dataclass:

```python
@dataclass
class TrainConfig:
    # Data
    data_path: str = "data/shakespeare.txt"
    train_split: float = 0.9

    # Model
    vocab_size: int = 256          # Character-level
    d_model: int = 384             # ~10M parameters
    n_layers: int = 7              # φ^4 ≈ 6.85
    n_heads: int = 8

    # Training
    batch_size: int = 64
    seq_len: int = 256             # Half of max for memory
    num_epochs: int = 10
    learning_rate: float = 3e-4
    warmup_steps: int = 500
    grad_clip: float = 1.0

    # Monitoring
    log_every_steps: int = 100
    generate_every_steps: int = 500
    track_julia_bounds: bool = True
    track_phi_shells: bool = True
```

## Custom Training

Edit the script's `__main__` section or create your own:

```python
from cosf_mi.training.train_cosf_transformer import TrainConfig, train

# Custom configuration
config = TrainConfig(
    data_path="data/my_text.txt",
    d_model=512,                   # Larger model
    n_layers=10,
    batch_size=32,                 # Smaller batches
    num_epochs=20,
    learning_rate=1e-4,
)

train(config)
```

## Training Outputs

### Checkpoints

Saved to `checkpoints/cosf_transformer/`:

```
checkpoints/cosf_transformer/
├── best_model.pt              # Best validation loss
├── checkpoint_epoch_1.pt      # After epoch 1
├── checkpoint_epoch_2.pt      # After epoch 2
└── ...
```

Each checkpoint contains:
- `model_state_dict`: Model weights
- `optimizer_state_dict`: Optimizer state
- `scheduler_state_dict`: LR scheduler state
- `config`: Model configuration
- `train_loss`, `val_loss`: Metrics
- `epoch`, `step`: Training progress

### Training Logs

```
Step   100 | Loss: 2.4321 | PPL:  11.38 | LR: 0.000060
  Julia: 2.3% exceed bound (max: 2.15)

Step   200 | Loss: 2.1543 | PPL:   8.62 | LR: 0.000120

----------------------------------------------------------------------
Generated sample:
ROMEO:
What is thy love of thee, and the world
And the sun shall be the sun of the world.

----------------------------------------------------------------------
```

## Text Generation

Use the `generate_text()` function for sampling:

```python
from cosf_mi.training.train_cosf_transformer import generate_text
from cosf_mi.models.cosf_transformer import COSFTransformer
import torch

# Load trained model
checkpoint = torch.load("checkpoints/cosf_transformer/best_model.pt")
model = COSFTransformer(checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Generate text
prompt = "ROMEO:\n"
generated = generate_text(
    model,
    prompt,
    max_length=200,
    temperature=0.8,
    device='cuda',
)
print(generated)
```

## COSF-Specific Monitoring

The training script tracks COSF properties:

### Julia Bounds

Monitors what percentage of activations exceed the Julia stability bound (2.0):

```
Julia: 2.3% exceed bound (max: 2.15)
```

- **Goal**: Keep below ~5%
- **If too high**: Model may be unstable
- **Solution**: Lower learning rate or add stronger regularization

### φ-Shell Distribution (Optional)

Tracks how activations cluster at golden ratio magnitudes:

```python
config.track_phi_shells = True
```

Will show distribution across φ^0, φ^1, φ^2, ... shells.

## Hardware Requirements

### Minimum

- GPU: 4GB VRAM (RTX 2060+)
- RAM: 8GB
- Storage: 2GB
- Time: ~2 hours for 10 epochs

### Recommended (Your Setup)

- GPU: RTX 5090 (32GB VRAM)
- RAM: 32GB
- Storage: 5GB
- Time: ~30 minutes for 10 epochs

### Memory Optimization

If you run out of memory:

```python
config.batch_size = 32        # Reduce from 64
config.seq_len = 128          # Reduce from 256
config.d_model = 256          # Smaller model
config.n_layers = 5           # Fewer layers
```

## Expected Results

After 10 epochs on tiny-shakespeare:

- **Train Loss**: ~1.0-1.2
- **Val Loss**: ~1.3-1.5
- **Perplexity**: ~3.5-4.5
- **Generation Quality**: Coherent Shakespeare-like text

### Sample Generation

**Epoch 1** (random):
```
ROMEO:
Whea thit and sud the,
Withard sthe thu:
```

**Epoch 5** (improving):
```
ROMEO:
What is the cause of this?
I will not be the son of thee.
```

**Epoch 10** (good):
```
ROMEO:
What is thy love, that thou dost speak of me?
I am not worthy of thy love and hate.
```

## Comparison with Standard Transformer

To compare COSF vs standard transformer:

1. Train COSF model (this script)
2. Train equivalent standard transformer:
   - Same d_model, n_layers, n_heads
   - Standard GELU instead of JuliaActivation
   - Standard sinusoidal instead of harmonic encoding
   - Standard attention (no φ-scaling)

3. Compare:
   - Final perplexity (should be similar)
   - Activation distributions (COSF should show φ-clustering)
   - Interpretability of learned representations
   - Attention patterns

## Troubleshooting

### Issue: "CUDA out of memory"

**Solution**: Reduce batch_size or seq_len
```python
config.batch_size = 32
config.seq_len = 128
```

### Issue: Loss is NaN

**Causes**:
- Learning rate too high
- Gradient explosion

**Solutions**:
```python
config.learning_rate = 1e-4  # Reduce LR
config.grad_clip = 0.5       # Stronger clipping
```

### Issue: Model not learning

**Symptoms**: Loss stuck at ~2.3 (random guessing)

**Solutions**:
- Train longer (20+ epochs)
- Increase model size
- Check data loading (verify train/val split)
- Reduce regularization

### Issue: Overfitting

**Symptoms**: Train loss << val loss

**Solutions**:
```python
config.weight_decay = 0.2    # Stronger regularization
# Or train less epochs
```

## Advanced Usage

### Resume from Checkpoint

```python
checkpoint = torch.load("checkpoints/cosf_transformer/checkpoint_epoch_5.pt")

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

# Continue training from step
start_step = checkpoint['step']
```

### Custom Data

Any text file works:

```python
config.data_path = "data/my_novel.txt"
```

Requirements:
- UTF-8 encoded
- Characters must be < vocab_size (256 for ASCII)
- Larger datasets (>1MB) recommended

### Export for Inference

```python
# Save just the model
torch.save({
    'model_state_dict': model.state_dict(),
    'config': model_config,
}, "cosf_model_inference.pt")
```

## Performance Benchmarks

On RTX 5090 with default config:

| Metric | Value |
|--------|-------|
| Model size | 9.65M parameters |
| Batch size | 64 |
| Seq length | 256 |
| Steps/sec | ~40 |
| Tokens/sec | ~650K |
| Memory usage | ~4GB VRAM |
| Training time (10 epochs) | ~30 minutes |

## Next Steps

After training:

1. **Analyze COSF Properties**
   - Run φ-decomposition on activations
   - Measure harmonic ratios between layers
   - Visualize attention patterns

2. **Fine-tune on Domain Data**
   - Use trained model as initialization
   - Fine-tune on specific domain (code, poetry, etc.)

3. **Compare with Baseline**
   - Train standard transformer
   - Measure perplexity difference
   - Analyze interpretability improvements

4. **Scale Up**
   - Increase to d_model=768, n_layers=12
   - Train on larger corpus (OpenWebText, etc.)
   - Evaluate on benchmarks

## References

- COSF Framework: See `COSF-MI/docs/`
- Model Architecture: `src/python/cosf_mi/models/cosf_transformer.py`
- Training Losses: `src/python/cosf_mi/training/cosf_losses.py`

## Support

For issues:
- GitHub: https://github.com/sportysport74/COSF-MI/issues
- Documentation: `COSF-MI/docs/`

Happy training! 🎭✨
