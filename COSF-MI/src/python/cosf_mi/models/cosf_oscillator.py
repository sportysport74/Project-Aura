"""
COSF Oscillatory Neural Network - Resonant Architecture

Instead of static weights (y = wx + b), we use phase oscillators (y = sin(ωx + φ)).
This transforms neural computation from linear algebra to wave interference.

Key Concepts:
- Weights become natural frequencies (ω)
- Biases become phase offsets (φ)
- Forward pass is phase synchronization, not matrix multiplication
- COSF (φ) acts as the coupling constant in Kuramoto dynamics

Mathematical Foundation:
    Standard NN:     y = W·x + b
    Oscillatory NN:  y = A·sin(Ω·x + Θ)

Where:
    Ω = Natural frequencies (learned, initialized with φ-scaling)
    Θ = Phase offsets (learned, initialized with golden angle)
    A = Amplitudes (learned, initialized to 1/√d)

Kuramoto Synchronization:
    dθ/dt = ω + (K/N) Σ sin(θ_j - θ_i)

Where K is the coupling constant (COSF = φ).

Usage:
    from cosf_mi.models.cosf_oscillator import COSFOscillatorTransformer

    model = COSFOscillatorTransformer(
        vocab_size=8000,
        d_model=768,
        n_layers=12,
        n_heads=12,
    )

    # Train normally - PyTorch optimizers work on ω, θ, A
    optimizer = torch.optim.AdamW(model.parameters())
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, Union


# Golden ratio - the fundamental COSF constant
PHI = (1 + math.sqrt(5)) / 2  # 1.618...

# Schumann resonances - Earth's natural frequencies (Hz)
SCHUMANN_BASE = 7.83  # Fundamental frequency
SCHUMANN_HARMONICS = [7.83, 14.3, 20.8, 27.3, 33.8]  # First 5 harmonics

# Chirality constants (spiral handedness)
CHIRALITY_RIGHT = 1.0  # Right-handed spiral (clockwise, φ)
CHIRALITY_LEFT = -1.0  # Left-handed spiral (counter-clockwise, -φ)
TERAHERTZ_SCALE = 1e12  # Scaling factor for spintronic hardware compatibility


def kuramoto_coherence_loss(
    phases: torch.Tensor,
    drift: float,
    coupling_strength: float = 0.01
) -> Tuple[torch.Tensor, float]:
    """
    AURA LEVEL 1: THE HEARTBEAT

    Reward oscillators for synchronizing when stable.
    Gate strength by drift - only sync when calm.

    This creates a global "pulse" that coordinates all oscillators.
    When drift is low (stable), oscillators are encouraged to synchronize.
    When drift is high (exploring), coupling weakens to allow freedom.

    Args:
        phases: Oscillator phases [batch, seq, features] or [batch, features] or [features]
        drift: Current homeostatic drift (0 = stable, >1 = chaotic)
        coupling_strength: Base coupling strength (default: 0.01, keep weak!)

    Returns:
        Tuple of (sync_loss, coherence_value)
        - sync_loss: Loss that decreases as coherence increases
        - coherence_value: Kuramoto order parameter R (0-1, higher = more sync)
    """
    # Flatten to [batch*seq, features] or [features] for coherence calculation
    original_shape = phases.shape
    if len(phases.shape) == 3:
        # [batch, seq, features] -> [batch*seq, features]
        phases_flat = phases.reshape(-1, phases.shape[-1])
    elif len(phases.shape) == 2:
        # [batch, features] -> already flat
        phases_flat = phases
    else:
        # [features] -> [1, features]
        phases_flat = phases.unsqueeze(0)

    # Calculate global phase coherence (Kuramoto order parameter)
    # R = |mean(exp(i * phases))| where R=1 is perfect sync, R=0 is random
    # This measures how aligned the oscillators are in phase space
    complex_phases = torch.exp(1j * phases_flat)
    mean_phase = torch.mean(complex_phases, dim=-1)  # Mean across features
    coherence = torch.abs(mean_phase).mean()  # Mean across batch

    # Gate by drift: only encourage sync when stable
    # High drift -> weak coupling (let explore)
    # Low drift -> strong coupling (let sync)
    # This creates adaptive coupling that responds to system state
    gate = max(0.0, min(1.0, 1.0 - drift))  # Python clamp for float
    effective_coupling = gate * coupling_strength

    # Loss: penalize LOW coherence (reward HIGH coherence)
    # When coherence is high (near 1), loss is near 0
    # When coherence is low (near 0), loss is near coupling_strength
    sync_loss = effective_coupling * (1.0 - coherence)

    return sync_loss, coherence.item()


@dataclass
class COSFOscillatorConfig:
    """Configuration for COSF Oscillatory Transformer."""

    vocab_size: int = 8000
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    d_ff: int = 3072  # 4 * d_model
    max_seq_len: int = 1024
    dropout: float = 0.1

    # COSF oscillator parameters
    coupling_constant: float = PHI  # K in Kuramoto model
    frequency_init_scale: float = PHI  # Scale for ω initialization
    phase_init_mode: str = 'golden'  # 'golden', 'random', or 'zero'

    # Julia stability bounds
    julia_bound: float = 1.5  # Max phase velocity magnitude
    apply_phase_correction: bool = True  # Damp unstable oscillations

    # Homeostatic decay parameters (biological self-regulation)
    homeostatic_decay_rate: float = 0.01  # Rate of decay toward rest frequency
    enable_homeostasis: bool = True  # Enable homeostatic decay (applied after optimizer.step())
    enable_refractory: bool = False  # DISABLED: Enable refractory period (prevent rapid firing)
    refractory_threshold: float = 0.05  # Minimum time between full activations
    frequency_clamp_min: float = 0.1  # Min frequency after decay
    frequency_clamp_max: float = 10.0  # Max frequency after decay (in units of φ)

    # Chirality parameters (spiral handedness for semantic differentiation)
    enable_chirality: bool = False  # Enable chiral oscillators
    chirality_init_prob: float = 0.5  # Probability of right-handed initialization
    enable_terahertz_scaling: bool = False  # Scale to THz for spintronic hardware

    # Heartbeat (Kuramoto) settings
    # 'static': R over the fixed phase-offset parameters (original Level 1;
    #           same value for every input)
    # 'live':   R over the live per-token phases Ω·x + Θ of every oscillator
    # 'live_control': control for 'live' that does NOT target synchrony: the same loss
    #           (strength, gate, target) on R_ψ = |mean_j e^{i(θ_j + ψ_j)}| with fixed
    #           random offsets ψ_j, so it is met by a random phase pattern (true R ~ 0).
    #           The logged coherence is still the true R.
    heartbeat_mode: str = 'static'
    heartbeat_strength: float = 0.01  # coupling_strength for the sync loss
    # If set, only penalize R below this target (sync band instead of
    # pushing R -> 1, which means every feature carries the same phase)
    heartbeat_target: Optional[float] = None

    # Context-dependent chirality: each token gets its own handedness
    # g = tanh(W·x + b) in (-1, 1) per group of features, multiplied into the
    # learned per-feature chirality. 0 = off (fixed per-feature handedness).
    chirality_context_groups: int = 0
    chirality_context_bias: float = 1.0  # gate starts at tanh(bias) for every token


class ChiralOscillatorLayer(nn.Module):
    """
    Oscillators with handedness - can spiral clockwise (φ) or
    counter-clockwise (-φ) based on input semantics.

    Inspired by MIT's p-wave magnetism research showing magnets can flip
    between left-handed and right-handed spirals. This allows semantic
    differentiation through GEOMETRY rather than just learned weights.

    Key concepts:
    - Right-handed (+1): Positive polarity (good, up, forward)
    - Left-handed (-1): Negative polarity (bad, down, backward)
    - Chirality can flip based on context (perspective shift)

    Args:
        in_features: Input dimension
        out_features: Output dimension
        coupling_constant: Kuramoto coupling (default: φ)
        enable_terahertz_scaling: Scale frequencies for spintronic hardware
        chirality_init_prob: Probability of initializing as right-handed
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        coupling_constant: float = PHI,
        enable_terahertz_scaling: bool = False,
        chirality_init_prob: float = 0.5,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.coupling_constant = coupling_constant
        self.enable_terahertz_scaling = enable_terahertz_scaling

        # Oscillator parameters (1/√in_features keeps Ω·x within ~one sine period)
        self.frequencies = nn.Parameter(torch.randn(out_features, in_features) * PHI / math.sqrt(in_features))
        self.phases = nn.Parameter(torch.zeros(out_features))
        self.amplitudes = nn.Parameter(torch.ones(out_features) / math.sqrt(out_features))

        # Chirality: +1 = right-handed (φ), -1 = left-handed (-φ)
        # Initialize randomly based on chirality_init_prob
        initial_chirality = torch.where(
            torch.rand(out_features) < chirality_init_prob,
            torch.tensor(CHIRALITY_RIGHT),
            torch.tensor(CHIRALITY_LEFT)
        )
        self.chirality = nn.Parameter(initial_chirality)

        # Homeostatic parameters
        self.rest_frequency = PHI
        self.register_buffer('chirality_flips', torch.tensor(0))  # Track perspective shifts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with chiral oscillation.

        Args:
            x: Input tensor [batch, seq, in_features]

        Returns:
            Complex interference pattern [batch, seq, out_features]
        """
        # Chirality determines spiral direction
        # Sign of chirality flips the rotation direction
        handed_freq = self.frequencies * self.chirality.unsqueeze(1)

        # Apply terahertz scaling if enabled (for spintronic hardware)
        if self.enable_terahertz_scaling:
            handed_freq = handed_freq * TERAHERTZ_SCALE

        # Compute phase angles: Ω·x (with handedness)
        total_phase = F.linear(x, handed_freq)  # Σ_j Ω_ij·x_j

        # Add phase offsets
        total_phase = total_phase + self.phases.unsqueeze(0).unsqueeze(0)

        # Complex oscillation with handedness
        # Real part: standard cosine
        real = torch.cos(total_phase)

        # Imaginary part: sine modulated by chirality
        # This creates the spiral handedness
        imag = torch.sin(total_phase) * self.chirality.unsqueeze(0).unsqueeze(0)

        # Combine real and imaginary (complex interference)
        oscillation = self.amplitudes.unsqueeze(0).unsqueeze(0) * (real + imag)

        # Apply Kuramoto coupling
        coupling_term = self.coupling_constant * torch.cos(total_phase)
        output = oscillation + 0.1 * coupling_term

        return output

    def flip_chirality(self, mask: torch.Tensor):
        """
        Flip handedness for specific neurons (perspective shift).

        This allows the model to "change perspective" on concepts,
        like viewing "expensive" as positive (luxury) vs negative (burden).

        Args:
            mask: Boolean mask [out_features] indicating which to flip
        """
        with torch.no_grad():
            self.chirality[mask] *= -1
            self.chirality_flips += mask.sum()

    def get_chirality_stats(self) -> dict:
        """Get statistics about chirality distribution."""
        with torch.no_grad():
            right_handed = (self.chirality > 0).sum().item()
            left_handed = (self.chirality < 0).sum().item()

            return {
                'right_handed': right_handed,
                'left_handed': left_handed,
                'chirality_ratio': right_handed / (right_handed + left_handed),
                'total_flips': self.chirality_flips.item(),
            }


class SemanticPolarityDetector(nn.Module):
    """
    Learns which concepts should spiral which way.

    This module analyzes token embeddings and predicts their semantic polarity,
    determining whether they should be represented with right-handed or
    left-handed chirality.

    Examples:
    - "good", "happy", "success" → right-handed (positive)
    - "bad", "sad", "failure" → left-handed (negative)
    - Context-dependent: "expensive" could flip based on context

    Args:
        d_model: Model dimension
        vocab_size: Vocabulary size
    """

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

        # Learnable polarity classifier
        self.polarity_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Tanh()  # Output in [-1, 1]
        )

        # Track polarity predictions per token
        self.register_buffer('token_polarities', torch.zeros(vocab_size))
        self.register_buffer('token_counts', torch.zeros(vocab_size))

    def forward(self, embeddings: torch.Tensor, token_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Predict semantic polarity from embeddings.

        Args:
            embeddings: Token embeddings [batch, seq, d_model]
            token_ids: Optional token IDs for tracking [batch, seq]

        Returns:
            Polarity scores [-1, 1] where:
                +1 = right-handed (positive)
                -1 = left-handed (negative)
        """
        # Predict polarity
        polarity = self.polarity_head(embeddings).squeeze(-1)  # [batch, seq]

        # Update token polarity statistics (optional)
        if token_ids is not None and not self.training:
            with torch.no_grad():
                for b in range(token_ids.size(0)):
                    for t in range(token_ids.size(1)):
                        tid = token_ids[b, t].item()
                        if tid < self.vocab_size:
                            self.token_polarities[tid] += polarity[b, t].item()
                            self.token_counts[tid] += 1

        return polarity

    def get_token_polarity(self, token_id: int) -> float:
        """Get average polarity for a specific token."""
        if self.token_counts[token_id] > 0:
            return (self.token_polarities[token_id] / self.token_counts[token_id]).item()
        return 0.0


class ChiralAttention(nn.Module):
    """
    Multi-head attention where heads can have opposite handedness.

    Some heads look for positive associations (right-handed),
    others look for negative associations (left-handed).
    This allows the model to simultaneously consider both perspectives.

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        dropout: Dropout probability
        coupling_constant: Kuramoto coupling
        enable_terahertz_scaling: Enable THz scaling
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        coupling_constant: float = PHI,
        enable_terahertz_scaling: bool = False,
    ):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.coupling_constant = coupling_constant

        # Use chiral oscillator layers for Q, K, V projections
        # Half the heads will be right-handed, half left-handed
        self.q_oscillator = ChiralOscillatorLayer(
            d_model, d_model, coupling_constant,
            enable_terahertz_scaling, chirality_init_prob=0.5
        )
        self.k_oscillator = ChiralOscillatorLayer(
            d_model, d_model, coupling_constant,
            enable_terahertz_scaling, chirality_init_prob=0.5
        )
        self.v_oscillator = ChiralOscillatorLayer(
            d_model, d_model, coupling_constant,
            enable_terahertz_scaling, chirality_init_prob=0.5
        )
        self.out_oscillator = ChiralOscillatorLayer(
            d_model, d_model, coupling_constant,
            enable_terahertz_scaling, chirality_init_prob=0.5
        )

        self.dropout = nn.Dropout(dropout)

        # Initialize heads with alternating chirality
        self._initialize_alternating_chirality()

    def _initialize_alternating_chirality(self):
        """Initialize heads with alternating right/left handedness."""
        with torch.no_grad():
            # Alternate chirality for each head
            for i in range(self.n_heads):
                head_start = i * self.d_k
                head_end = (i + 1) * self.d_k

                # Even heads: right-handed, odd heads: left-handed
                chirality = CHIRALITY_RIGHT if i % 2 == 0 else CHIRALITY_LEFT

                self.q_oscillator.chirality[head_start:head_end] = chirality
                self.k_oscillator.chirality[head_start:head_end] = chirality
                self.v_oscillator.chirality[head_start:head_end] = chirality
                self.out_oscillator.chirality[head_start:head_end] = chirality

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Chiral multi-head attention.

        Args:
            x: Input [batch, seq, d_model]
            mask: Optional attention mask

        Returns:
            Attention output [batch, seq, d_model]
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V using chiral oscillators
        q = self.q_oscillator(x)
        k = self.k_oscillator(x)
        v = self.v_oscillator(x)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)

        # Final projection using chiral oscillator
        output = self.out_oscillator(attn_output)

        return output

    def get_chirality_stats(self) -> dict:
        """Get chirality statistics for all projection layers."""
        return {
            'q': self.q_oscillator.get_chirality_stats(),
            'k': self.k_oscillator.get_chirality_stats(),
            'v': self.v_oscillator.get_chirality_stats(),
            'out': self.out_oscillator.get_chirality_stats(),
        }


class COSFOscillatorLayer(nn.Module):
    """
    Oscillatory layer replacing nn.Linear.

    Instead of y = Wx + b, computes y = A·sin(Ω·x + Θ)

    Args:
        in_features: Input dimension
        out_features: Output dimension
        coupling_constant: K in Kuramoto synchronization (default: φ)
        frequency_init_scale: Scale for frequency initialization
        phase_init_mode: How to initialize phases ('golden', 'random', 'zero')
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        coupling_constant: float = PHI,
        frequency_init_scale: float = PHI,
        phase_init_mode: str = 'golden',
        homeostatic_decay_rate: float = 0.01,
        enable_homeostasis: bool = True,
        enable_refractory: bool = True,
        refractory_threshold: float = 0.05,
        frequency_clamp_min: float = 0.1,
        frequency_clamp_max: float = 10.0,
        enable_chirality: bool = False,
        chirality_init_prob: float = 0.5,
        enable_terahertz_scaling: bool = False,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.coupling_constant = coupling_constant

        # Homeostatic parameters
        # Rest state is 0 (no coupling), so drift = mean |Ω| and decay acts like
        # weight decay. A rest state of φ pulled every weight to the same constant,
        # collapsing the layer (all outputs identical) after a few sleeps.
        self.rest_frequency = 0.0
        self.homeostatic_decay_rate = homeostatic_decay_rate
        self.enable_homeostasis = enable_homeostasis
        self.enable_refractory = enable_refractory
        self.refractory_threshold = refractory_threshold
        self.frequency_clamp_min = frequency_clamp_min
        self.frequency_clamp_max = frequency_clamp_max * PHI

        # Natural frequencies Ω [out_features, in_features]
        # Each output oscillator has in_features frequencies.
        # Scaled by 1/√in_features so the summed phase Ω·x stays within ~one
        # sine period; unscaled init gave phase std ~26 and sin() of noise.
        self.frequencies = nn.Parameter(
            torch.randn(out_features, in_features) * frequency_init_scale / math.sqrt(in_features)
        )

        # Phase offsets Θ [out_features]
        if phase_init_mode == 'golden':
            # Initialize with golden angle spacing (137.5°)
            golden_angle = 2 * math.pi * (1 - 1/PHI)
            # Wrapped to [0, 2π): same sin/cos, but the raw values ran up to
            # ~2400 rad, and AdamW weight decay (∝ value) then dragged them
            # ~0.007 rad every step. That cost 0.61 val loss in the Oct 8 A/B
            # (experiments/ab_run.py, arms baseline vs phase_wrap).
            phases = (torch.arange(out_features) * golden_angle).remainder(2 * math.pi)
        elif phase_init_mode == 'random':
            phases = torch.rand(out_features) * 2 * math.pi
        else:  # 'zero'
            phases = torch.zeros(out_features)

        self.phases = nn.Parameter(phases)

        # Amplitudes A [out_features]
        # Initialize like standard weights: 1/√d
        self.amplitudes = nn.Parameter(
            torch.ones(out_features) / math.sqrt(in_features)
        )

        # For phase velocity tracking (Julia bounds)
        self.register_buffer('phase_velocity', torch.zeros(out_features))
        self.register_buffer('sync_state', torch.zeros(out_features))

        # Homeostatic tracking buffers
        self.register_buffer('last_activation', torch.zeros(out_features))
        self.register_buffer('refractory_suppressions', torch.tensor(0))
        self.register_buffer('total_decay_applied', torch.tensor(0.0))
        self.register_buffer('homeostatic_steps', torch.tensor(0))

        # Chirality parameters (spiral handedness)
        self.enable_chirality = enable_chirality
        self.enable_terahertz_scaling = enable_terahertz_scaling

        if self.enable_chirality:
            # Initialize chirality: +1 = right-handed, -1 = left-handed
            initial_chirality = torch.where(
                torch.rand(out_features) < chirality_init_prob,
                torch.tensor(CHIRALITY_RIGHT),
                torch.tensor(CHIRALITY_LEFT)
            )
            self.chirality = nn.Parameter(initial_chirality)
            self.register_buffer('chirality_flips', torch.tensor(0))
        else:
            # No chirality - use all right-handed (standard behavior)
            self.register_buffer('chirality', torch.ones(out_features))
            self.register_buffer('chirality_flips', torch.tensor(0))

    def add_context_chirality(self, groups: int, bias: float = 1.0):
        """
        Give each token its own handedness, per group of output features.

        g = tanh(W·x + b) starts at tanh(bias) for every token (W = 0): +0.76 at
        bias 1 (training begins like fixed chirality), 0 at bias 0 (handedness
        free to go either way). A token's g can cross 0, reversing the rotation
        of that feature group for that token only.
        """
        assert self.out_features % groups == 0
        self.context_groups = groups
        self.context_gate = nn.Linear(self.in_features, groups)
        nn.init.zeros_(self.context_gate.weight)
        nn.init.constant_(self.context_gate.bias, bias)

    def forward(self, x: torch.Tensor, apply_homeostasis: bool = None) -> torch.Tensor:
        """
        Compute oscillatory transformation with homeostatic decay.

        Args:
            x: Input tensor [batch, ..., in_features]
            apply_homeostasis: Override homeostasis setting (default: use self.enable_homeostasis)

        Returns:
            Output tensor [batch, ..., out_features]
        """
        # Always fp32: phases reach ~2400 rad, where bf16's step is 16 rad, so
        # cos/sin of a bf16 phase would be noise. Run outside any autocast.
        if torch.is_autocast_enabled(x.device.type):
            with torch.autocast(x.device.type, enabled=False):
                return self.forward(x.float(), apply_homeostasis)

        # Determine if homeostasis should be applied
        if apply_homeostasis is None:
            apply_homeostasis = self.enable_homeostasis and self.training

        # x: [batch, seq, in_features]
        # frequencies: [out_features, in_features]

        # Apply chirality (spiral handedness) to frequencies
        # Chirality determines rotation direction: +1 (right) or -1 (left)
        handed_freq = self.frequencies * self.chirality.unsqueeze(1)

        # Apply terahertz scaling if enabled (for spintronic hardware)
        if self.enable_terahertz_scaling:
            handed_freq = handed_freq * TERAHERTZ_SCALE

        # Total phase shift Ω·x (with handedness): Σ_j Ω_ij·x_j
        # [batch, seq, out_features]. Computed as a matmul; the previous
        # broadcast built a [batch, seq, out, in] tensor (same result, ~GBs).
        total_phase = F.linear(x, handed_freq)

        # Per-token handedness g [batch, seq, out_features] (1 if not enabled)
        context_gate = getattr(self, 'context_gate', None)
        if context_gate is not None:
            g = torch.tanh(context_gate(x))
            g = g.repeat_interleave(self.out_features // self.context_groups, dim=-1)
            total_phase = total_phase * g
            self.last_context_chirality = g.detach()
        else:
            g = None

        # Add phase offsets Θ
        total_phase = total_phase + self.phases.unsqueeze(0).unsqueeze(0)

        # Keep the live phases (with grad) for the live heartbeat loss
        self.last_total_phase = total_phase

        # Compute oscillation with chirality
        if self.enable_chirality:
            # Complex oscillation: real + imaginary components
            # Real part: standard cosine
            real = torch.cos(total_phase)
            # Imaginary part: sine modulated by chirality (creates spiral handedness)
            imag = torch.sin(total_phase) * self.chirality.unsqueeze(0).unsqueeze(0)
            if g is not None:
                imag = imag * g  # per-token handedness, same form as chirality
            # Combine for complex interference pattern
            oscillation = self.amplitudes.unsqueeze(0).unsqueeze(0) * (real + imag)
        else:
            # Standard oscillation: A·sin(Ω·x + Θ)
            oscillation = self.amplitudes.unsqueeze(0).unsqueeze(0) * torch.sin(total_phase)

        # Apply refractory period if enabled
        if self.enable_refractory and apply_homeostasis:
            # Compute activation strength
            activation_strength = oscillation.abs().mean(dim=(0, 1)).detach()

            with torch.no_grad():
                # Check if enough time has passed since last strong activation
                refractory_mask = activation_strength > self.refractory_threshold
                time_since_last = torch.abs(activation_strength - self.last_activation)

                # Suppress if fired too recently
                suppress_mask = (refractory_mask & (time_since_last < self.refractory_threshold))

            if suppress_mask.any():
                # Apply refractory suppression (non-in-place to preserve gradients)
                suppression_factor = 0.5  # Half strength during refractory period
                # Create suppression multiplier using where to avoid in-place ops
                suppress_mask_expanded = suppress_mask.view(1, 1, -1)
                multiplier = torch.where(
                    suppress_mask_expanded,
                    torch.tensor(suppression_factor, device=oscillation.device),
                    torch.tensor(1.0, device=oscillation.device)
                )
                oscillation = oscillation * multiplier

                with torch.no_grad():
                    self.refractory_suppressions += suppress_mask.sum()

            # Update last activation (no_grad to not affect backprop)
            with torch.no_grad():
                self.last_activation = activation_strength.clone()

        # Apply Kuramoto coupling: add small synchronization term
        # This helps oscillators align their phases over time
        coupling_term = self.coupling_constant * torch.cos(total_phase)
        output = oscillation + 0.1 * coupling_term  # Small coupling contribution

        # NOTE: Homeostatic decay is now applied OUTSIDE forward pass
        # to avoid breaking autograd. Call apply_homeostatic_decay() after optimizer.step()
        # The decay is tracked but not applied during forward to preserve gradients

        # Update phase velocity for Julia bounds monitoring
        with torch.no_grad():
            # Estimate dθ/dt from change in the input-driven phase Ω·x.
            # The static offsets Θ are excluded: sync_state starts at 0, so
            # including them (golden-angle values up to ~2400 rad) produced a
            # huge spurious first "velocity" that triggered repeated damping.
            driven_phase = (total_phase - self.phases).mean(dim=(0, 1))
            phase_diff = driven_phase - self.sync_state
            self.phase_velocity = 0.9 * self.phase_velocity + 0.1 * phase_diff
            self.sync_state = driven_phase

        return output

    def get_phase_metrics(self) -> dict:
        """Get phase synchronization and homeostatic metrics."""
        # Compute homeostatic drift
        homeostatic_drift = (self.frequencies - self.rest_frequency).abs().mean().item()

        # Compute average decay applied per step
        avg_decay = (self.total_decay_applied / max(self.homeostatic_steps, 1)).item()

        return {
            'mean_frequency': self.frequencies.mean().item(),
            'std_frequency': self.frequencies.std().item(),
            'mean_phase': self.phases.mean().item(),
            'phase_velocity': self.phase_velocity.abs().mean().item(),
            'max_phase_velocity': self.phase_velocity.abs().max().item(),
            'amplitude_scale': self.amplitudes.mean().item(),
            'homeostatic_drift': homeostatic_drift,
            'refractory_suppressions': self.refractory_suppressions.item(),
            'avg_decay_applied': avg_decay,
        }

    def apply_homeostatic_decay(self):
        """
        Apply homeostatic decay - pull frequencies toward φ-harmonic rest state.

        This should be called AFTER optimizer.step() to avoid breaking autograd.
        Only applies if enable_homeostasis=True.
        """
        if not self.enable_homeostasis:
            return

        with torch.no_grad():
            # Compute drift from rest frequency
            freq_diff = self.frequencies - self.rest_frequency

            # Apply decay toward rest
            decay_amount = self.homeostatic_decay_rate * freq_diff
            self.frequencies.data -= decay_amount

            # Clamp frequency magnitude (symmetric: negative frequencies are valid;
            # a positive-only clamp would erase all inhibitory couplings)
            self.frequencies.data.clamp_(-self.frequency_clamp_max, self.frequency_clamp_max)

            # Track decay statistics
            self.total_decay_applied += decay_amount.abs().mean()
            self.homeostatic_steps += 1

    def reset_to_rest(self, gentle: bool = True, decay_rate: float = 0.5):
        """
        Homeostatic reset - reduce fatigue while preserving learned structure.

        Args:
            gentle: If True, perform selective decay (preserves phase relationships).
                   If False, full reset (zeros everything - destructive).
            decay_rate: How much to decay toward rest (0=no change, 1=full reset).

        Gentle sleep (recommended):
            - Resets phase VELOCITY (fatigue indicator) to zero
            - Preserves frequencies (learned couplings)
            - Preserves phase OFFSETS (semantic structure)
            - Preserves chirality orientation
            - Resets fatigue counters

        Full reset (destructive, for testing only):
            - Zeros all phases
            - Snaps frequencies to φ
            - Destroys all learned relationships
        """
        with torch.no_grad():
            if gentle:
                # SELECTIVE DECAY: Preserve structure, reduce fatigue

                # 1. Reset phase velocity (fatigue indicator)
                self.phase_velocity.zero_()

                # 2. Preserve frequencies. They ARE the learned weights; decaying
                # them 50% toward a shared constant every sleep erased learning.
                # (decay_rate is kept in the signature for compatibility.)

                # 3. Preserve phase offsets (semantic structure)
                # Phases encode learned relationships - DO NOT ZERO
                # Only wrap to prevent unbounded growth (sin/cos are 2π-periodic)
                self.phases.remainder_(2 * math.pi)  # Wrap to [0, 2π)

                # 4. Reset fatigue counters
                self.homeostatic_steps.zero_()
                self.total_decay_applied.zero_()

                # 5. Preserve sync_state and last_activation (learned patterns)
                # 6. Preserve chirality (handedness is semantic, not fatigue)

                # 7. Reset refractory suppressions (these are transient)
                self.refractory_suppressions.zero_()

            else:
                # FULL RESET: Destructive, for testing only
                self.phases.zero_()
                self.frequencies.fill_(self.rest_frequency)
                self.phase_velocity.zero_()
                self.sync_state.zero_()
                self.last_activation.zero_()
                self.refractory_suppressions.zero_()
                self.total_decay_applied.zero_()
                self.homeostatic_steps.zero_()

    def flip_chirality(self, mask: torch.Tensor):
        """
        Flip handedness for specific neurons (perspective shift).

        This allows the model to "change perspective" on concepts,
        like viewing "expensive" as positive (luxury) vs negative (burden).

        Args:
            mask: Boolean mask [out_features] indicating which to flip
        """
        if not self.enable_chirality:
            return  # No-op if chirality not enabled

        with torch.no_grad():
            self.chirality[mask] *= -1
            self.chirality_flips += mask.sum()

    def get_chirality_stats(self) -> dict:
        """Get statistics about chirality distribution."""
        if not self.enable_chirality:
            return {
                'enabled': False,
                'right_handed': self.out_features,  # All standard
                'left_handed': 0,
                'chirality_ratio': 1.0,
                'total_flips': 0,
            }

        with torch.no_grad():
            right_handed = (self.chirality > 0).sum().item()
            left_handed = (self.chirality < 0).sum().item()

            return {
                'enabled': True,
                'right_handed': right_handed,
                'left_handed': left_handed,
                'chirality_ratio': right_handed / (right_handed + left_handed),
                'total_flips': self.chirality_flips.item(),
            }


class KuramotoAttention(nn.Module):
    """
    Multi-head attention as phase synchronization.

    Standard attention: Attention(Q, K, V) = softmax(QK^T/√d)V
    Kuramoto attention: Phase coupling between query and key oscillators

    The attention weights emerge from phase coherence between oscillators.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        coupling_constant: float = PHI,
        homeostatic_decay_rate: float = 0.01,
        enable_homeostasis: bool = True,
        enable_refractory: bool = True,
        refractory_threshold: float = 0.05,
        enable_chirality: bool = False,
        chirality_init_prob: float = 0.5,
        enable_terahertz_scaling: bool = False,
    ):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.coupling_constant = coupling_constant

        # Use oscillator layers for Q, K, V projections
        self.q_oscillator = COSFOscillatorLayer(
            d_model, d_model, coupling_constant,
            homeostatic_decay_rate=homeostatic_decay_rate,
            enable_homeostasis=enable_homeostasis,
            enable_refractory=enable_refractory,
            refractory_threshold=refractory_threshold,
            enable_chirality=enable_chirality,
            chirality_init_prob=chirality_init_prob,
            enable_terahertz_scaling=enable_terahertz_scaling,
        )
        self.k_oscillator = COSFOscillatorLayer(
            d_model, d_model, coupling_constant,
            homeostatic_decay_rate=homeostatic_decay_rate,
            enable_homeostasis=enable_homeostasis,
            enable_refractory=enable_refractory,
            refractory_threshold=refractory_threshold,
            enable_chirality=enable_chirality,
            chirality_init_prob=chirality_init_prob,
            enable_terahertz_scaling=enable_terahertz_scaling,
        )
        self.v_oscillator = COSFOscillatorLayer(
            d_model, d_model, coupling_constant,
            homeostatic_decay_rate=homeostatic_decay_rate,
            enable_homeostasis=enable_homeostasis,
            enable_refractory=enable_refractory,
            refractory_threshold=refractory_threshold,
            enable_chirality=enable_chirality,
            chirality_init_prob=chirality_init_prob,
            enable_terahertz_scaling=enable_terahertz_scaling,
        )

        # Output projection
        self.out_oscillator = COSFOscillatorLayer(
            d_model, d_model, coupling_constant,
            homeostatic_decay_rate=homeostatic_decay_rate,
            enable_homeostasis=enable_homeostasis,
            enable_refractory=enable_refractory,
            refractory_threshold=refractory_threshold,
            enable_chirality=enable_chirality,
            chirality_init_prob=chirality_init_prob,
            enable_terahertz_scaling=enable_terahertz_scaling,
        )

        self.dropout = nn.Dropout(dropout)

        # Phase coherence tracking
        self.register_buffer('coherence', torch.tensor(0.0))

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        """
        Compute Kuramoto-style attention.

        Args:
            x: Input tensor [batch, seq, d_model]
            mask: Optional bool attention mask (True = may attend). Any explicit
                mask rules out the flash kernel; for plain causal use causal=True.
            causal: Causal attention without building a mask (flash-eligible)

        Returns:
            Output tensor [batch, seq, d_model]
        """
        batch_size, seq_len, d_model = x.shape

        # Project to Q, K, V using oscillators
        Q = self.q_oscillator(x)  # [batch, seq, d_model]
        K = self.k_oscillator(x)
        V = self.v_oscillator(x)

        # Reshape for multi-head: [batch, n_heads, seq, d_k]
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention, interpreted as phase coherence, with the
        # coupling constant (φ) folded into the scale: softmax(φ·QKᵀ/√d)·V.
        # Fused SDPA never materializes the [seq, seq] weights; under bf16
        # autocast it runs the flash kernel.
        scale = self.coupling_constant / math.sqrt(self.d_k)
        if mask is not None and mask.dtype != torch.bool:
            mask = mask != 0  # SDPA reads float masks as additive biases; keep the 0 = blocked meaning
        is_causal = causal and mask is None
        out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=is_causal,
            scale=scale,
        )  # [batch, n_heads, seq, d_k]

        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        # Output projection
        out = self.out_oscillator(out)

        # Track phase coherence (mean max attention weight). SDPA doesn't return
        # the weights, so recompute them for batch element 0 only, before dropout.
        with torch.no_grad(), torch.autocast(x.device.type, enabled=False):
            scores = (Q[:1].float() @ K[:1].float().transpose(-2, -1)) * scale
            if mask is not None:
                scores = scores.masked_fill(~mask[:1], float('-inf'))
            elif is_causal:
                scores = scores.masked_fill(
                    torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device).triu(1), float('-inf'))
            self.coherence = scores.softmax(dim=-1).max(dim=-1)[0].mean()

        return out

    def get_coherence(self) -> float:
        """Get current phase coherence metric."""
        return self.coherence.item()


class ResonantPositionalEncoding(nn.Module):
    """
    Positional encoding as phase offsets in a resonant cavity.

    Instead of sinusoidal encoding, we use Schumann resonance harmonics.
    Each position gets a unique phase offset in the global resonance pattern.
    """

    def __init__(self, d_model: int, max_seq_len: int = 1024):
        super().__init__()

        self.d_model = d_model

        # Create resonant positional encoding
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        # Use Schumann harmonics and φ-scaling
        for i in range(d_model):
            # Cycle through Schumann harmonics
            harmonic = SCHUMANN_HARMONICS[i % len(SCHUMANN_HARMONICS)]

            # Frequency decreases with φ-scaling
            freq = harmonic / (PHI ** (i // len(SCHUMANN_HARMONICS)))

            # Alternate sin/cos like standard positional encoding
            if i % 2 == 0:
                pe[:, i] = torch.sin(position * freq / max_seq_len).squeeze()
            else:
                pe[:, i] = torch.cos(position * freq / max_seq_len).squeeze()

        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add resonant positional encoding."""
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].unsqueeze(0)


class COSFOscillatorBlock(nn.Module):
    """
    Transformer block using oscillatory layers.

    Replaces standard Linear layers with COSFOscillatorLayer.
    """

    def __init__(self, config: COSFOscillatorConfig):
        super().__init__()

        self.config = config

        # Kuramoto attention
        self.attn = KuramotoAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout,
            coupling_constant=config.coupling_constant,
            homeostatic_decay_rate=config.homeostatic_decay_rate,
            enable_homeostasis=config.enable_homeostasis,
            enable_refractory=config.enable_refractory,
            refractory_threshold=config.refractory_threshold,
            enable_chirality=config.enable_chirality,
            chirality_init_prob=config.chirality_init_prob,
            enable_terahertz_scaling=config.enable_terahertz_scaling,
        )

        # Feed-forward network with oscillators
        self.ff1 = COSFOscillatorLayer(
            config.d_model,
            config.d_ff,
            coupling_constant=config.coupling_constant,
            homeostatic_decay_rate=config.homeostatic_decay_rate,
            enable_homeostasis=config.enable_homeostasis,
            enable_refractory=config.enable_refractory,
            refractory_threshold=config.refractory_threshold,
            enable_chirality=config.enable_chirality,
            chirality_init_prob=config.chirality_init_prob,
            enable_terahertz_scaling=config.enable_terahertz_scaling,
        )
        self.ff2 = COSFOscillatorLayer(
            config.d_ff,
            config.d_model,
            coupling_constant=config.coupling_constant,
            homeostatic_decay_rate=config.homeostatic_decay_rate,
            enable_homeostasis=config.enable_homeostasis,
            enable_refractory=config.enable_refractory,
            refractory_threshold=config.refractory_threshold,
            enable_chirality=config.enable_chirality,
            chirality_init_prob=config.chirality_init_prob,
            enable_terahertz_scaling=config.enable_terahertz_scaling,
        )

        # Layer norms (keep these standard)
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)

        self.dropout = nn.Dropout(config.dropout)

        # Activation: use tanh to keep bounded (Julia stability)
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, causal: bool = False) -> torch.Tensor:
        """Forward pass through oscillator block."""

        # Attention with residual
        attn_out = self.attn(self.ln1(x), mask, causal)
        x = x + self.dropout(attn_out)

        # Feed-forward with residual
        ff_out = self.ff2(self.activation(self.ff1(self.ln2(x))))
        x = x + self.dropout(ff_out)

        return x

    def get_phase_metrics(self) -> dict:
        """Get phase metrics from all oscillators (including chirality if enabled)."""
        metrics = {}

        # Attention oscillators (with chirality stats)
        metrics['attn_q'] = self.attn.q_oscillator.get_phase_metrics()
        metrics['attn_q'].update(self.attn.q_oscillator.get_chirality_stats())

        metrics['attn_k'] = self.attn.k_oscillator.get_phase_metrics()
        metrics['attn_k'].update(self.attn.k_oscillator.get_chirality_stats())

        metrics['attn_v'] = self.attn.v_oscillator.get_phase_metrics()
        metrics['attn_v'].update(self.attn.v_oscillator.get_chirality_stats())

        metrics['attn_out'] = self.attn.out_oscillator.get_phase_metrics()
        metrics['attn_out'].update(self.attn.out_oscillator.get_chirality_stats())

        metrics['attn_coherence'] = self.attn.get_coherence()

        # FF oscillators (with chirality stats)
        metrics['ff1'] = self.ff1.get_phase_metrics()
        metrics['ff1'].update(self.ff1.get_chirality_stats())

        metrics['ff2'] = self.ff2.get_phase_metrics()
        metrics['ff2'].update(self.ff2.get_chirality_stats())

        return metrics


class COSFOscillatorTransformer(nn.Module):
    """
    Full COSF Oscillatory Transformer.

    A language model where computation happens via phase synchronization
    instead of matrix multiplication.
    """

    def __init__(self, config: COSFOscillatorConfig):
        super().__init__()

        self.config = config

        # Token embedding (standard)
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Resonant positional encoding
        self.pos_encoding = ResonantPositionalEncoding(config.d_model, config.max_seq_len)

        # Oscillator blocks
        self.blocks = nn.ModuleList([
            COSFOscillatorBlock(config)
            for _ in range(config.n_layers)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.d_model)

        # Output head (standard Linear for now, could be oscillator too)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie embeddings
        self.lm_head.weight = self.token_embedding.weight

        # Julia stability tracking
        self.register_buffer('julia_violations', torch.tensor(0))
        self.register_buffer('phase_corrections', torch.tensor(0))

        # Fixed random phase offsets ψ for the 'live_control' heartbeat. Drawn from a fixed
        # seed (independent of the training seed) and not saved, so checkpoint keys don't change.
        if getattr(config, 'heartbeat_mode', 'static') == 'live_control':
            n_features = config.n_layers * (5 * config.d_model + config.d_ff)  # q, k, v, out, ff2 + ff1
            psi = torch.rand(n_features, generator=torch.Generator().manual_seed(271828)) * (2 * math.pi)
            self.register_buffer('control_offsets', psi, persistent=False)

        self._init_weights()

        # After _init_weights so the gate keeps its "start at +0.76" init
        groups = getattr(config, 'chirality_context_groups', 0)
        if groups and config.enable_chirality:
            for osc in self.live_oscillators():
                osc.add_context_chirality(groups, getattr(config, 'chirality_context_bias', 1.0))

    def _init_weights(self):
        """Initialize weights."""
        # Token embeddings
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)

        # Oscillator layers are initialized in their __init__

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_coherence_loss: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Token IDs [batch, seq]
            targets: Target IDs for loss computation [batch, seq]
            return_coherence_loss: If True, compute and return Kuramoto coherence loss

        Returns:
            If targets is None: logits [batch, seq, vocab_size]
            If targets provided: (logits, loss, sync_loss, coherence) if return_coherence_loss
                                 else (logits, loss)
        """
        batch_size, seq_len = input_ids.shape

        # Embed tokens
        x = self.token_embedding(input_ids)  # [batch, seq, d_model]

        # Add resonant positional encoding
        x = self.pos_encoding(x)

        # Apply oscillator blocks with causal attention (position i attends to <= i;
        # no mask tensor, so SDPA can use the flash kernel)
        # (Julia correction is NOT applied here: mutating weights mid-forward
        # corrupts gradients and changed the model during eval/generation.
        # Call apply_julia_correction() after optimizer.step() instead.)
        for block in self.blocks:
            x = block(x, causal=True)

        # Final layer norm
        x = self.ln_f(x)

        # Project to vocabulary
        logits = self.lm_head(x)

        # Compute loss if targets provided
        if targets is not None:
            # Losses and the Kuramoto heartbeat always in fp32, outside autocast
            with torch.autocast(input_ids.device.type, enabled=False):
                task_loss = F.cross_entropy(
                    logits.float().view(-1, self.config.vocab_size),
                    targets.view(-1),
                )

                # Compute Kuramoto coherence loss (AURA LEVEL 1: THE HEARTBEAT)
                if return_coherence_loss:
                    # Get current drift from first oscillator
                    # Drift = mean distance from rest frequency (φ)
                    first_osc = self.blocks[0].attn.q_oscillator
                    drift = (first_osc.frequencies - first_osc.rest_frequency).abs().mean().detach()

                    # Older checkpoints pickle a config without the heartbeat fields
                    mode = getattr(self.config, 'heartbeat_mode', 'static')
                    strength = getattr(self.config, 'heartbeat_strength', 0.01)
                    target = getattr(self.config, 'heartbeat_target', None)

                    if mode in ('live', 'live_control'):
                        sync_loss, coherence = self.live_heartbeat_loss(drift, strength, target,
                                                                        control=(mode == 'live_control'))
                        return logits, task_loss, sync_loss, coherence
                    drift = drift.item()  # the static loss below takes a float

                    # Collect all phases from all oscillators
                    # Note: Oscillators have different sizes (d_model vs d_ff)
                    # so we concatenate them into a single phase vector
                    all_phases = []
                    for block in self.blocks:
                        # Get phases from each oscillator in the block
                        all_phases.append(block.attn.q_oscillator.phases)
                        all_phases.append(block.attn.k_oscillator.phases)
                        all_phases.append(block.attn.v_oscillator.phases)
                        all_phases.append(block.attn.out_oscillator.phases)
                        all_phases.append(block.ff1.phases)
                        all_phases.append(block.ff2.phases)

                    # Concatenate all phases into single vector [total_features]
                    # This creates a unified phase space across all oscillators
                    phases_tensor = torch.cat(all_phases)

                    # Compute Kuramoto synchronization loss
                    sync_loss, coherence = kuramoto_coherence_loss(
                        phases_tensor,
                        drift,
                        coupling_strength=strength
                    )

                    return logits, task_loss, sync_loss, coherence

                return logits, task_loss

        return logits

    def live_oscillators(self):
        """All oscillator layers, in block order."""
        for block in self.blocks:
            yield block.attn.q_oscillator
            yield block.attn.k_oscillator
            yield block.attn.v_oscillator
            yield block.attn.out_oscillator
            yield block.ff1
            yield block.ff2

    def live_coherence(self, offsets: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Kuramoto R on the live phases from the last forward pass.

        For each token, R = |mean over every oscillator feature of e^{i·phase}|,
        where phase = Ω·x + Θ for that token. Returns R averaged over tokens
        (a differentiable scalar). Unlike the static version, this depends on
        the input: it measures whether the network's oscillators actually
        fall into step while processing text.

        offsets: per-feature phase shifts ψ added before the mean (the
        'live_control' heartbeat); None gives the true R.
        """
        cos_sum = 0.0
        sin_sum = 0.0
        n_features = 0
        for osc in self.live_oscillators():
            phase = osc.last_total_phase  # [batch, seq, features]
            if offsets is not None:
                phase = phase + offsets[n_features:n_features + phase.shape[-1]]
            cos_sum = cos_sum + torch.cos(phase).sum(dim=-1)
            sin_sum = sin_sum + torch.sin(phase).sum(dim=-1)
            n_features += phase.shape[-1]
        r_per_token = torch.sqrt(cos_sum ** 2 + sin_sum ** 2 + 1e-12) / n_features
        return r_per_token.mean()

    def live_heartbeat_loss(
        self,
        drift: Union[float, torch.Tensor],
        strength: float = 0.01,
        target: Optional[float] = None,
        control: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Heartbeat loss on the live phases, gated by drift like the original.

        target=None: loss = strength·gate·(1 - R)  (push toward full sync)
        target=t:    loss = strength·gate·relu(t - R)  (only lift R up to t)
        control=True: the same loss on R_ψ (fixed random offsets, see
        'live_control'); the returned coherence is still the true R.

        drift may be a 0-dim tensor; then nothing here syncs with the GPU. The
        returned coherence is a detached 0-dim tensor (use float() to read it).
        """
        coherence = self.live_coherence(self.control_offsets if control else None)
        if torch.is_tensor(drift):
            # Same arithmetic as the float path: gate·strength in float64, then one fp32 multiply
            gate_strength = ((1.0 - drift.double()).clamp(0.0, 1.0) * strength).float()
        else:
            gate_strength = max(0.0, min(1.0, 1.0 - drift)) * strength
        if target is None:
            shortfall = 1.0 - coherence
        else:
            shortfall = F.relu(target - coherence)
        loss = gate_strength * shortfall
        if control:
            with torch.no_grad():
                coherence = self.live_coherence()  # report the true R, not R_ψ
        return loss, coherence.detach()

    def apply_julia_correction(self):
        """
        Apply Julia-bound damping to all blocks.

        Call AFTER optimizer.step() during training only (like
        apply_homeostatic_decay). No-op if config.apply_phase_correction is False.
        """
        if not self.config.apply_phase_correction:
            return
        # φ-correction: any oscillator whose max |dθ/dt| exceeds the Julia bound gets its
        # frequencies scaled by 1/φ. Done for all oscillators at once on the GPU with no
        # .item(), so it never stalls the CPU; ×1.0 leaves the others bit-identical.
        oscillators = list(self.live_oscillators())
        with torch.no_grad():
            max_velocity = torch.stack(torch._foreach_norm([o.phase_velocity for o in oscillators], ord=float('inf')))
            exceeded = max_velocity > self.config.julia_bound
            factors = torch.where(exceeded, 1 / PHI, 1.0).to(oscillators[0].frequencies.dtype)
            torch._foreach_mul_([o.frequencies for o in oscillators], list(factors.unbind(0)))
            n = exceeded.sum()
            self.julia_violations += n
            self.phase_corrections += n

    def get_phase_coherence_metrics(self) -> dict:
        """
        Get phase coherence metrics across all blocks.

        Returns dict with:
            - mean_coherence: Average phase coherence across attention heads
            - max_phase_velocity: Maximum phase velocity in any oscillator
            - julia_violations: Total Julia bound violations
            - phase_corrections: Total corrections applied
        """
        coherences = []
        max_velocities = []
        homeostatic_drifts = []
        refractory_suppressions = 0
        avg_decays = []

        # Chirality metrics
        total_right_handed = 0
        total_left_handed = 0
        total_chirality_flips = 0
        chirality_enabled = self.config.enable_chirality

        for block in self.blocks:
            metrics = block.get_phase_metrics()
            coherences.append(metrics['attn_coherence'])

            # Collect max velocities and homeostatic metrics from all oscillators
            for key in ['attn_q', 'attn_k', 'attn_v', 'attn_out', 'ff1', 'ff2']:
                max_velocities.append(metrics[key]['max_phase_velocity'])
                homeostatic_drifts.append(metrics[key]['homeostatic_drift'])
                refractory_suppressions += metrics[key]['refractory_suppressions']
                avg_decays.append(metrics[key]['avg_decay_applied'])

                # Collect chirality stats if enabled
                if chirality_enabled and metrics[key].get('enabled', False):
                    total_right_handed += metrics[key].get('right_handed', 0)
                    total_left_handed += metrics[key].get('left_handed', 0)
                    total_chirality_flips += metrics[key].get('total_flips', 0)

        result = {
            'mean_coherence': sum(coherences) / len(coherences) if coherences else 0.0,
            'max_phase_velocity': max(max_velocities) if max_velocities else 0.0,
            'julia_violations': self.julia_violations.item(),
            'phase_corrections': self.phase_corrections.item(),
            'homeostatic_drift': sum(homeostatic_drifts) / len(homeostatic_drifts) if homeostatic_drifts else 0.0,
            'refractory_suppressions': refractory_suppressions,
            'avg_decay_applied': sum(avg_decays) / len(avg_decays) if avg_decays else 0.0,
        }

        # Add chirality metrics if enabled
        if chirality_enabled and (total_right_handed + total_left_handed) > 0:
            result['chirality_enabled'] = True
            result['total_right_handed'] = total_right_handed
            result['total_left_handed'] = total_left_handed
            result['chirality_balance'] = total_right_handed / (total_right_handed + total_left_handed)
            result['total_chirality_flips'] = total_chirality_flips
        else:
            result['chirality_enabled'] = False

        return result

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.

        Args:
            input_ids: Starting tokens [batch, seq]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Optional top-k filtering

        Returns:
            Generated token IDs [batch, seq + max_new_tokens]
        """
        self.eval()

        for _ in range(max_new_tokens):
            # Crop to max_seq_len
            input_crop = input_ids[:, -self.config.max_seq_len:]

            # Forward pass
            with torch.no_grad():
                logits = self(input_crop)

            # Get last token logits
            logits = logits[:, -1, :] / temperature

            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    def apply_homeostatic_reset(self, gentle: bool = True, decay_rate: float = 0.5):
        """
        Apply homeostatic reset to all oscillator layers.

        Args:
            gentle: If True, use selective decay (preserves learned structure).
                   If False, full reset (destructive, zeros everything).
            decay_rate: Strength of decay toward φ (0=none, 1=full snap).

        Gentle sleep (default, recommended):
            - Reduces fatigue (resets phase velocity)
            - Decays frequencies toward φ
            - Preserves phase relationships (semantic structure)
            - Preserves chirality orientation

        Full reset (gentle=False, destructive):
            - Zeros all oscillator state
            - Use ONLY for debugging or between unrelated tasks

        Call this during training sleep cycles with gentle=True.
        """
        mode = "GENTLE" if gentle else "FULL"
        print(f"Applying {mode} homeostatic reset to all oscillators...")

        reset_count = 0
        for block in self.blocks:
            # Reset attention oscillators
            block.attn.q_oscillator.reset_to_rest(gentle=gentle, decay_rate=decay_rate)
            block.attn.k_oscillator.reset_to_rest(gentle=gentle, decay_rate=decay_rate)
            block.attn.v_oscillator.reset_to_rest(gentle=gentle, decay_rate=decay_rate)
            block.attn.out_oscillator.reset_to_rest(gentle=gentle, decay_rate=decay_rate)

            # Reset feed-forward oscillators
            block.ff1.reset_to_rest(gentle=gentle, decay_rate=decay_rate)
            block.ff2.reset_to_rest(gentle=gentle, decay_rate=decay_rate)

            reset_count += 6  # 6 oscillators per block

        # Reset Julia tracking
        self.julia_violations.zero_()
        self.phase_corrections.zero_()

        if gentle:
            print(f"Applied gentle reset to {reset_count} oscillator layers (preserved structure)")
        else:
            print(f"Applied FULL reset to {reset_count} oscillator layers (WARNING: destructive)")

    def apply_homeostatic_decay(self):
        """
        Apply homeostatic decay to all oscillator layers.

        This gradually pulls frequencies toward φ-harmonic rest state.
        Should be called AFTER optimizer.step() to avoid breaking autograd.

        Only applies if enable_homeostasis=True in the config.
        """
        if not self.config.enable_homeostasis:
            return

        for block in self.blocks:
            # Apply decay to attention oscillators
            block.attn.q_oscillator.apply_homeostatic_decay()
            block.attn.k_oscillator.apply_homeostatic_decay()
            block.attn.v_oscillator.apply_homeostatic_decay()
            block.attn.out_oscillator.apply_homeostatic_decay()

            # Apply decay to feed-forward oscillators
            block.ff1.apply_homeostatic_decay()
            block.ff2.apply_homeostatic_decay()


def count_parameters(model: nn.Module) -> dict:
    """Count parameters in COSF Oscillator model."""
    total = sum(p.numel() for p in model.parameters())

    # Count by component
    counts = {
        'token_embedding': sum(p.numel() for p in model.token_embedding.parameters()),
        'blocks': sum(p.numel() for p in model.blocks.parameters()),
        'ln_f': sum(p.numel() for p in model.ln_f.parameters()),
        'lm_head': 0,  # Tied with embeddings
        'total': total,
    }

    # Count oscillator-specific parameters
    oscillator_params = 0
    for block in model.blocks:
        for oscillator in [
            block.attn.q_oscillator,
            block.attn.k_oscillator,
            block.attn.v_oscillator,
            block.attn.out_oscillator,
            block.ff1,
            block.ff2,
        ]:
            oscillator_params += sum(p.numel() for p in oscillator.parameters())

    counts['oscillators'] = oscillator_params

    return counts


# Example usage and test
if __name__ == "__main__":
    print("=" * 70)
    print("COSF OSCILLATORY TRANSFORMER - RESONANT ARCHITECTURE")
    print("=" * 70)
    print()

    # Create config
    config = COSFOscillatorConfig(
        vocab_size=8000,
        d_model=768,
        n_layers=12,
        n_heads=12,
        max_seq_len=1024,
    )

    print("Configuration:")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Model dim: {config.d_model}")
    print(f"  Layers: {config.n_layers}")
    print(f"  Heads: {config.n_heads}")
    print(f"  Coupling constant (K): {config.coupling_constant:.4f} (PHI)")
    print(f"  Julia bound: {config.julia_bound}")
    print(f"  Homeostatic decay: {config.homeostatic_decay_rate}")
    print(f"  Refractory period: {config.enable_refractory}")
    print()

    # Create model
    model = COSFOscillatorTransformer(config)

    # Count parameters
    param_counts = count_parameters(model)
    print("Parameter counts:")
    print(f"  Total: {param_counts['total']:,}")
    print(f"  Token embedding: {param_counts['token_embedding']:,}")
    print(f"  Oscillator layers: {param_counts['oscillators']:,}")
    print(f"  Transformer blocks: {param_counts['blocks']:,}")
    print(f"  Final layer norm: {param_counts['ln_f']:,}")
    print()

    # Test forward pass
    print("Testing forward pass...")
    batch_size = 2
    seq_len = 128

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    logits, loss = model(input_ids, targets, return_coherence_loss=False)

    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")
    print()

    # Test phase coherence metrics
    print("Phase coherence metrics (after training step):")
    metrics = model.get_phase_coherence_metrics()
    print(f"  Mean coherence: {metrics['mean_coherence']:.4f}")
    print(f"  Max phase velocity: {metrics['max_phase_velocity']:.4f}")
    print(f"  Julia violations: {metrics['julia_violations']}")
    print(f"  Phase corrections: {metrics['phase_corrections']}")
    print(f"  Homeostatic drift: {metrics['homeostatic_drift']:.4f}")
    print(f"  Refractory suppressions: {metrics['refractory_suppressions']}")
    print(f"  Avg decay applied: {metrics['avg_decay_applied']:.6f}")
    print()

    # Test generation
    print("Testing generation...")
    model.eval()
    prompt = torch.randint(0, config.vocab_size, (1, 10))

    with torch.no_grad():
        generated = model.generate(prompt, max_new_tokens=20, temperature=1.0)

    print(f"  Prompt length: {prompt.shape[1]}")
    print(f"  Generated length: {generated.shape[1]}")
    print()

    # Test homeostatic reset
    print("Testing homeostatic reset...")
    print(f"  Before reset - Homeostatic drift: {metrics['homeostatic_drift']:.4f}")
    model.apply_homeostatic_reset()

    # Check metrics after reset
    metrics_after = model.get_phase_coherence_metrics()
    print(f"  After reset - Homeostatic drift: {metrics_after['homeostatic_drift']:.4f}")
    print(f"  After reset - Refractory suppressions: {metrics_after['refractory_suppressions']}")
    print()

    print("=" * 70)
    print("OSCILLATORY TRANSFORMER TEST COMPLETE")
    print("=" * 70)
    print()
    print("Key features:")
    print("  - Weights are natural frequencies (omega)")
    print("  - Activations are phase states (theta)")
    print("  - COSF (PHI) is the coupling constant")
    print("  - Attention via Kuramoto synchronization")
    print("  - Resonant positional encoding (Schumann harmonics)")
    print("  - Julia bounds for phase stability")
    print("  - Homeostatic decay (biological self-regulation)")
    print("  - Refractory periods (prevent rapid firing)")
    print("  - Trainable with standard PyTorch optimizers")
    print()
    print("This is the Level 2 COSF architecture.")
    print("Neural networks that RESONATE, not calculate.")
