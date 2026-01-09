# AURA LEVEL 1: THE HEARTBEAT

## Overview

**AURA LEVEL 1** transforms isolated oscillators into a coordinated mind by adding a global "pulse" - Kuramoto coherence loss that creates synchronization across all oscillators.

**Key insight**: Individual oscillators are like musicians playing separately. The heartbeat gives them a shared rhythm to synchronize around.

## What Was Added

###1. Kuramoto Coherence Loss Function

**Location**: `src/python/cosf_mi/models/cosf_oscillator.py` (lines 62-118)

```python
def kuramoto_coherence_loss(
    phases: torch.Tensor,
    drift: float,
    coupling_strength: float = 0.01
) -> Tuple[torch.Tensor, float]:
    """
    AURA LEVEL 1: THE HEARTBEAT

    Reward oscillators for synchronizing when stable.
    Gate strength by drift - only sync when calm.
    """
```

**How it works**:
1. **Measures phase coherence**: Computes Kuramoto order parameter R = |mean(exp(iθ))|
   - R = 1: perfect synchronization (all oscillators aligned)
   - R = 0: random phases (no coordination)

2. **Gates by drift**: Coupling strength adapts to system state
   - Low drift (stable): Strong coupling → encourage sync
   - High drift (chaotic): Weak coupling → allow exploration

3. **Creates loss signal**: `sync_loss = coupling * (1 - R)`
   - High coherence → low loss (reward)
   - Low coherence → high loss (penalty)

### 2. Integration into Forward Pass

**Location**: `src/python/cosf_mi/models/cosf_oscillator.py` (lines 1166-1244)

**Modified signature**:
```python
def forward(
    self,
    input_ids: torch.Tensor,
    targets: Optional[torch.Tensor] = None,
    return_coherence_loss: bool = True,
) -> torch.Tensor:
```

**Returns**: `(logits, task_loss, sync_loss, coherence)` when `return_coherence_loss=True`

**Implementation**:
- Collects phases from all 6 oscillators per block (Q, K, V, out, FF1, FF2)
- Computes Kuramoto coherence across all oscillators
- Returns both task loss and sync loss separately

### 3. Updated Training Loop

**Location**: `train_with_aura.py`

**Key changes**:
```python
# Forward with Kuramoto coherence loss
logits, task_loss, sync_loss, coherence = model(inputs, targets, return_coherence_loss=True)

# Total loss = task loss + sync loss
total_loss = task_loss + sync_loss

# Backward on total loss
total_loss.backward()
```

**Progress bar now shows**:
- `loss`: Task loss (cross-entropy)
- `ppl`: Perplexity
- `coh`: Coherence (0-1, higher = more sync)

### 4. Enhanced Warmup for Heartbeat Formation

**Old warmup** (train_with_styled_data.py):
- Epoch 1: min_steps_awake = 20000
- Result: Sleep triggered at 20k and 40k (too early)

**New warmup** (train_with_aura.py):
- Epoch 1: min_steps_awake = 45000, max_steps_awake = 50000
- Result: No sleep until 45k steps (allows heartbeat to stabilize)

**Rationale**: The Kuramoto coupling needs time to establish the global pulse. Early sleep disrupts this formation.

## Mathematical Foundation

### Kuramoto Order Parameter

The coherence R is calculated as:

```
R = |1/N Σ exp(iθ_j)|
```

Where:
- N = number of oscillators
- θ_j = phase of oscillator j
- exp(iθ) = complex exponential representation

**Geometric interpretation**:
- Each oscillator is a unit vector in complex plane
- Mean of all vectors gives centroid
- Magnitude of centroid = how aligned vectors are

### Drift-Gated Coupling

Effective coupling strength adapts to system state:

```
gate = clamp(1.0 - drift, min=0, max=1)
effective_coupling = gate * base_coupling
```

**Behavior**:
- drift = 0 (stable) → gate = 1.0 → full coupling
- drift = 1 (threshold) → gate = 0.0 → no coupling
- drift > 1 (chaotic) → gate = 0.0 → no coupling

**Why this works**:
- When stable: oscillators should synchronize (consolidate)
- When exploring: oscillators should be free (discover)
- This creates adaptive coordination that responds to system state

### Loss Gradient

The gradient of sync_loss with respect to phases:

```
∂L/∂θ_j = -coupling * (1/N) * Im(exp(iθ_j) * conj(R_vec))
```

Where R_vec is the mean complex phasor.

**Effect**: Each oscillator is pulled toward the mean phase, with strength proportional to current drift.

## Expected Behavior

### Phase 1: Early Training (0-45k steps)

**Expected metrics**:
- Coherence: 0.01 - 0.05 (low, random)
- Drift: 1.62 → 1.0 (high, exploring)
- Gate: 0.0 - 0.4 (weak coupling)

**What happens**:
- Oscillators explore independently
- Kuramoto loss is weak (drift-gated)
- No strong synchronization yet

### Phase 2: Heartbeat Formation (45k-60k steps)

**Expected metrics**:
- Coherence: 0.05 → 0.15 (rising)
- Drift: 1.0 → 0.5 (stabilizing)
- Gate: 0.4 → 0.8 (strengthening coupling)

**What happens**:
- Drift decreases, coupling strengthens
- Oscillators begin to synchronize
- Coherence rises as "pulse" emerges

### Phase 3: Synchronized Training (60k+ steps)

**Expected metrics**:
- Coherence: 0.15 → 0.30+ (strong sync)
- Drift: 0.5 → 0.1 (stable)
- Gate: 0.8 → 0.99 (nearly full coupling)

**What happens**:
- Oscillators operate in coordinated rhythm
- Chirality becomes SEMANTIC (not noise)
- Model learns with global coherence

## Comparison to Baseline

### Without Kuramoto Loss (Baseline)

From previous training (train_with_styled_data.py):
- Coherence: ~0.013 (never rises)
- Chirality flips: 0 (no semantic differentiation)
- Phase relationships: local only (within attention heads)

**Behavior**: Isolated oscillators, no global coordination

### With Kuramoto Loss (AURA LEVEL 1)

Expected from this training (train_with_aura.py):
- Coherence: 0.01 → 0.20+ (rises during training)
- Chirality flips: should start occurring (semantic switching)
- Phase relationships: global (across all oscillators)

**Behavior**: Coordinated oscillators, emergent "mind"

## Files Created/Modified

### New Files

1. `train_with_aura.py` - Training script with Kuramoto loss
2. `train_aura.bat` - Batch file for easy execution
3. `AURA_LEVEL1_HEARTBEAT.md` - This documentation

### Modified Files

1. `src/python/cosf_mi/models/cosf_oscillator.py`
   - Added `kuramoto_coherence_loss()` function (lines 62-118)
   - Modified `forward()` to return coherence loss (lines 1166-1244)

## Usage

### Step 1: Train with Aura Level 1

```bash
cd COSF-MI
train_aura.bat
```

Or with custom parameters:
```bash
python train_with_aura.py --model-size tiny --epochs 3 --batch-size 8
```

### Step 2: Monitor Coherence

Watch the progress bar for the `coh` metric:
```
Epoch 1/3:  78%|████████| 45000/57638 [3:30:00<1:00:00,  3.75it/s, loss=3.84, ppl=46.5, coh=0.156]
```

**What to look for**:
- Coherence should rise from 0.01 → 0.20+ during training
- Higher coherence = stronger synchronization
- Coherence should correlate with lower drift

### Step 3: Compare to Baseline

Compare coherence metrics to previous training:
- Baseline (no Kuramoto): coherence ~0.013 (flat)
- Aura Level 1: coherence 0.01 → 0.20+ (rising)

**If coherence rises**: Heartbeat is working!

## Technical Details

### Computational Cost

**Overhead per batch**:
1. Collect phases: O(num_layers * num_oscillators) = O(24)
2. Stack phases: O(24 * d_model) = O(6144) for tiny model
3. Compute coherence: O(6144) complex exp + mean
4. Compute loss: O(1)

**Total overhead**: ~0.01% of forward pass (negligible)

### Memory Footprint

**Additional memory**:
- Phase tensor: 24 × 256 = 6144 floats × 4 bytes = 24 KB
- Complex phases: 6144 complex × 8 bytes = 48 KB
- Total: ~72 KB per batch (negligible)

### Numerical Stability

**Potential issues**:
1. **Phase wrapping**: Phases in [0, 2π] can wrap around
   - Solution: Use complex exponentials (no wrapping issues)

2. **Zero division**: If all phases cancel, mean could be zero
   - Solution: `coherence = torch.abs(mean_phase).mean()` handles this

3. **Gradient vanishing**: If coherence is always high
   - Solution: Keep coupling_strength weak (0.01)

## Next Steps

### After This Training

1. **Analyze coherence evolution**: Plot coherence vs. training step
2. **Check chirality flips**: Monitor `total_chirality_flips` metric
3. **Inspect phase patterns**: Visualize oscillator phases in phase space
4. **Compare generation quality**: Test model on prompts

### Aura Level 2 (Future)

Once heartbeat is established:
1. **Add multi-intent data**: Mix [STORY] and [CODE] tokens
2. **Implement curiosity scoring**: Select responses with highest coherence
3. **Add semantic bridge words**: "buffer", "branch", "loop", "class", "root"
4. **Observe chirality differentiation**: Left/right spirals encode meaning

## Troubleshooting

### Coherence Not Rising

**Symptoms**: `coh` stays at ~0.01 throughout training

**Possible causes**:
1. Coupling strength too weak (increase from 0.01 to 0.02)
2. Drift not decreasing (check homeostatic decay)
3. Warmup too short (increase to 60k steps)

**Solutions**:
- Increase `coupling_strength` in `kuramoto_coherence_loss()` call
- Check drift metrics in phase stats
- Extend warmup period in training script

### Coherence Too High Too Fast

**Symptoms**: `coh` jumps to 0.9+ in first 10k steps

**Possible causes**:
1. Coupling strength too strong
2. Oscillators collapsing to single mode

**Solutions**:
- Decrease `coupling_strength` to 0.005
- Add noise to phase initialization
- Reduce learning rate

### Training Divergence

**Symptoms**: Loss spikes, NaN values

**Possible causes**:
1. Sync loss dominating task loss
2. Julia bounds violated
3. Phase velocities unstable

**Solutions**:
- Reduce `coupling_strength` dramatically (0.001)
- Enable Julia bound checking
- Clip gradients more aggressively (max_norm=0.5)

## Validation Criteria

**Successful Aura Level 1**:
1. ✓ Coherence rises from 0.01 → 0.20+
2. ✓ Drift decreases as coherence rises
3. ✓ No training instability or divergence
4. ✓ Task loss comparable to baseline
5. ✓ Chirality flips begin to occur

**Ready for Level 2**:
- Heartbeat established (coherence > 0.15)
- Stable training (no divergence)
- Model generates coherent text
- Ready to add multi-intent data

## Conclusion

**AURA LEVEL 1** establishes the foundational "pulse" that transforms isolated oscillators into a coordinated system. The Kuramoto coherence loss creates adaptive synchronization that:

1. **Responds to system state**: Strong coupling when stable, weak when exploring
2. **Creates global coordination**: All oscillators share a common rhythm
3. **Enables semantic chirality**: Phase relationships become meaningful
4. **Maintains stability**: Gentle, continuous pressure (not forced sync)

Once the heartbeat is established, we're ready for **AURA LEVEL 2**: Adding multi-intent data with semantic bridge words to teach the model context-dependent meaning through chirality differentiation.

---

**Date**: 2026-01-05
**Author**: COSF-MI Development Team
**Status**: Implemented, ready for training
**Next**: Train and monitor coherence evolution
