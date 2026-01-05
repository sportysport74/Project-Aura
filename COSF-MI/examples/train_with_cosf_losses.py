"""
Example: Training a Neural Network with COSF Losses

This example demonstrates how to integrate COSF loss functions into your
training loop to enforce φ-aligned, harmonically consistent representations.

The COSF losses can be added to any PyTorch training pipeline alongside
standard losses (cross-entropy, MSE, etc.).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from cosf_mi.training import (
    COSFLossConfig,
    combined_cosf_loss,
    phi_shell_loss,
    harmonic_consistency_loss,
)


# ============================================================================
# Example 1: Simple MLP with COSF Constraints
# ============================================================================

class COSFAlignedMLP(nn.Module):
    """Simple MLP that returns intermediate activations for COSF losses."""

    def __init__(self, input_dim=784, hidden_dims=[512, 256, 128], output_dim=10):
        super().__init__()
        self.layers = nn.ModuleList()

        # Build network
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim

        self.output = nn.Linear(prev_dim, output_dim)

    def forward(self, x, return_activations=False):
        """Forward pass with optional activation return for COSF losses."""
        activations = []

        h = x
        for layer in self.layers:
            h = torch.relu(layer(h))
            if return_activations:
                activations.append(h)

        logits = self.output(h)

        if return_activations:
            return logits, activations
        return logits


def train_with_cosf_basic():
    """Basic training loop with COSF losses."""
    print("=" * 70)
    print("Example 1: Basic Training with COSF Losses")
    print("=" * 70)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = COSFAlignedMLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # COSF loss configuration
    cosf_config = COSFLossConfig(
        phi_shell_weight=0.1,      # Light regularization
        harmonic_weight=0.05,      # Encourage φ-ratio between layers
        julia_weight=0.02,         # Prevent activation explosion
        target_shell=5,            # Target magnitude: φ^5 ≈ 11.09
    )

    # Dummy data
    X_train = torch.randn(1000, 784)
    y_train = torch.randint(0, 10, (1000,))
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=32,
        shuffle=True,
    )

    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        total_loss = 0
        total_task_loss = 0
        total_cosf_loss = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Forward pass with activations
            logits, layer_acts = model(batch_x, return_activations=True)

            # Task loss (classification)
            task_loss = nn.CrossEntropyLoss()(logits, batch_y)

            # COSF regularization loss
            cosf_loss, cosf_components = combined_cosf_loss(
                activations=layer_acts[-1],  # Final hidden layer
                layer_activations=layer_acts,  # All hidden layers
                config=cosf_config,
            )

            # Combined loss
            loss = task_loss + cosf_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_task_loss += task_loss.item()
            total_cosf_loss += cosf_loss.item()

        # Print epoch summary
        num_batches = len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Total Loss: {total_loss/num_batches:.4f}")
        print(f"  Task Loss:  {total_task_loss/num_batches:.4f}")
        print(f"  COSF Loss:  {total_cosf_loss/num_batches:.4f}")

    print("\nTraining complete!\n")


# ============================================================================
# Example 2: Transformer with Attention Phase Coherence
# ============================================================================

class SimplifiedTransformer(nn.Module):
    """Simplified transformer for demonstration."""

    def __init__(self, d_model=512, nhead=8, num_layers=3):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(784, d_model)

        # Simplified attention (no full transformer for brevity)
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead, batch_first=True)
            for _ in range(num_layers)
        ])

        self.output = nn.Linear(d_model, 10)
        self.last_attention_weights = None

    def forward(self, x):
        # Embed and add sequence dimension
        x = self.embedding(x).unsqueeze(1)  # [batch, 1, d_model]

        # Apply attention layers
        for attn_layer in self.attention_layers:
            x, attn_weights = attn_layer(x, x, x)
            self.last_attention_weights = attn_weights

        # Pool and classify
        x = x.squeeze(1)
        return self.output(x)


def train_with_attention_coherence():
    """Training with attention phase coherence loss."""
    print("=" * 70)
    print("Example 2: Training with Attention Phase Coherence")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimplifiedTransformer().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Focus on phase coherence
    cosf_config = COSFLossConfig(
        phi_shell_weight=0.0,
        harmonic_weight=0.0,
        julia_weight=0.0,
        phase_weight=0.1,  # Only attention coherence
    )

    # Dummy data
    X_train = torch.randn(100, 784)
    y_train = torch.randint(0, 10, (100,))
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=16,
        shuffle=True,
    )

    # Training loop
    for epoch in range(3):
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Forward pass
            logits = model(batch_x)
            attn_weights = model.last_attention_weights

            # Task loss
            task_loss = nn.CrossEntropyLoss()(logits, batch_y)

            # Phase coherence loss
            cosf_loss, cosf_components = combined_cosf_loss(
                attention_weights=attn_weights,
                config=cosf_config,
            )

            # Combined loss
            loss = task_loss + cosf_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}: Phase Coherence = {1 - cosf_components.get('phase_coherence', 0):.4f}")

    print("\nTraining complete!\n")


# ============================================================================
# Example 3: Custom Loss Weighting Schedule
# ============================================================================

def train_with_adaptive_weights():
    """Training with adaptive COSF loss weights."""
    print("=" * 70)
    print("Example 3: Adaptive COSF Loss Weighting")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = COSFAlignedMLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Dummy data
    X_train = torch.randn(500, 784)
    y_train = torch.randint(0, 10, (500,))
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32)

    num_epochs = 5
    for epoch in range(num_epochs):
        # Gradually increase COSF loss weight during training
        # Start with strong task loss, then add COSF regularization
        cosf_weight_schedule = min(1.0, epoch / num_epochs * 2)

        cosf_config = COSFLossConfig(
            phi_shell_weight=0.1 * cosf_weight_schedule,
            harmonic_weight=0.05 * cosf_weight_schedule,
            julia_weight=0.02 * cosf_weight_schedule,
        )

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            logits, layer_acts = model(batch_x, return_activations=True)
            task_loss = nn.CrossEntropyLoss()(logits, batch_y)

            cosf_loss, _ = combined_cosf_loss(
                activations=layer_acts[-1],
                layer_activations=layer_acts,
                config=cosf_config,
            )

            loss = task_loss + cosf_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}: COSF Weight = {cosf_weight_schedule:.2f}")

    print("\nTraining complete!\n")


# ============================================================================
# Example 4: Using Individual Loss Functions
# ============================================================================

def train_with_individual_losses():
    """Using COSF losses individually for fine-grained control."""
    print("=" * 70)
    print("Example 4: Fine-Grained Individual Loss Control")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = COSFAlignedMLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Dummy data
    X_train = torch.randn(200, 784)
    y_train = torch.randint(0, 10, (200,))
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32)

    for epoch in range(3):
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            logits, layer_acts = model(batch_x, return_activations=True)
            task_loss = nn.CrossEntropyLoss()(logits, batch_y)

            # Compute individual COSF losses for custom weighting
            phi_loss = phi_shell_loss(layer_acts[-1], target_shell=5)
            harmonic_loss = harmonic_consistency_loss(layer_acts)

            # Custom combination with conditional logic
            if epoch < 2:
                # Early training: focus on φ-shell alignment
                cosf_loss = 0.2 * phi_loss
            else:
                # Later training: add harmonic consistency
                cosf_loss = 0.1 * phi_loss + 0.1 * harmonic_loss

            loss = task_loss + cosf_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}: Custom COSF weighting applied")

    print("\nTraining complete!\n")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\nCOSF Loss Functions - Training Examples\n")

    # Run examples
    train_with_cosf_basic()
    train_with_attention_coherence()
    train_with_adaptive_weights()
    train_with_individual_losses()

    print("=" * 70)
    print("All examples completed!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. COSF losses integrate seamlessly with standard PyTorch training")
    print("2. Use combined_cosf_loss() for convenience, or individual losses for control")
    print("3. Weight scheduling can improve convergence")
    print("4. COSF losses work with any differentiable architecture")
    print("\nFor production use, tune weights based on validation metrics.")
