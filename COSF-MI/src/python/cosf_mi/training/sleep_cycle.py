"""
Sleep Cycle Hook for COSF Oscillatory Training

Implements automatic homeostatic reset ("neural sleep") when the model
becomes "fatigued" (high homeostatic drift).

Biological Analogy:
    - Neurons accumulate metabolic waste during prolonged activity
    - Sleep allows neurons to reset and consolidate memories
    - After sleep, performance often improves (memory consolidation)

COSF Analogy:
    - Oscillators drift from φ during prolonged training
    - Homeostatic reset returns them to φ-harmonic rest state
    - After reset, loss may spike briefly then converge to lower minimum

Usage:
    from cosf_mi.training.sleep_cycle import SleepCycleHook

    hook = SleepCycleHook(
        drift_threshold=2.0,      # Trigger sleep when drift > 2.0
        min_steps_awake=500,      # Min steps between sleeps
        measure_consolidation=True # Track loss before/after sleep
    )

    for step in range(num_steps):
        logits, loss = model(inputs, targets)
        loss.backward()
        optimizer.step()

        # Check if model needs sleep
        hook.check_and_sleep(model, step, loss.item())
"""

import torch
import time
from typing import Optional, Dict, List
from dataclasses import dataclass, field


@dataclass
class SleepEvent:
    """Record of a single sleep cycle."""
    step: int
    drift_before: float
    drift_after: float
    loss_before: float
    loss_after_1: Optional[float] = None  # Loss 1 step after wake
    loss_after_10: Optional[float] = None  # Loss 10 steps after wake
    loss_after_50: Optional[float] = None  # Loss 50 steps after wake
    min_loss_after: Optional[float] = None  # Minimum loss within 50 steps
    consolidation_benefit: Optional[float] = None  # loss_before - min_loss_after


@dataclass
class SleepCycleHook:
    """
    Automatic sleep cycle hook for COSF Oscillatory training.

    Monitors homeostatic drift and triggers reset when model is "fatigued".

    Args:
        drift_threshold: Trigger sleep when drift exceeds this (default: 2.0)
        min_steps_awake: Minimum steps between sleep cycles (default: 500)
        max_steps_awake: Force sleep after this many steps (default: 5000)
        measure_consolidation: Track loss evolution after sleep (default: True)
        verbose: Print sleep events (default: True)
    """

    drift_threshold: float = 2.0
    min_steps_awake: int = 500
    max_steps_awake: int = 5000
    measure_consolidation: bool = True
    verbose: bool = True

    # Internal state
    _last_sleep_step: int = field(default=0, init=False)
    _steps_since_sleep: int = field(default=0, init=False)
    _sleep_events: List[SleepEvent] = field(default_factory=list, init=False)
    _consolidation_window: Dict[int, float] = field(default_factory=dict, init=False)

    def check_and_sleep(
        self,
        model,
        step: int,
        loss: float,
    ) -> bool:
        """
        Check if model needs sleep and trigger reset if needed.

        Args:
            model: COSFOscillatorTransformer model
            step: Current training step
            loss: Current training loss

        Returns:
            True if sleep was triggered, False otherwise
        """
        self._steps_since_sleep = step - self._last_sleep_step

        # Update consolidation tracking for previous sleep events
        if self.measure_consolidation:
            self._update_consolidation_tracking(step, loss)

        # Check if sleep is needed
        should_sleep, reason = self._should_sleep(model, step)

        if should_sleep:
            self._trigger_sleep(model, step, loss, reason)
            return True

        return False

    def _should_sleep(self, model, step: int) -> tuple[bool, str]:
        """Determine if model should sleep."""

        # Don't sleep too early
        if self._steps_since_sleep < self.min_steps_awake:
            return False, ""

        # Get current drift
        metrics = model.get_phase_coherence_metrics()
        drift = metrics['homeostatic_drift']

        # Force sleep if too long awake
        if self._steps_since_sleep >= self.max_steps_awake:
            return True, f"max_steps_awake ({self.max_steps_awake})"

        # Sleep if drift too high (fatigued)
        if drift > self.drift_threshold:
            return True, f"drift {drift:.4f} > {self.drift_threshold}"

        return False, ""

    def _trigger_sleep(
        self,
        model,
        step: int,
        loss: float,
        reason: str,
    ):
        """Trigger sleep cycle."""

        # Get metrics before sleep
        metrics_before = model.get_phase_coherence_metrics()
        drift_before = metrics_before['homeostatic_drift']

        if self.verbose:
            print()
            print("=" * 70)
            print(f"💤 NEURAL SLEEP TRIGGERED (Step {step})")
            print("=" * 70)
            print(f"  Reason: {reason}")
            print(f"  Steps since last sleep: {self._steps_since_sleep}")
            print(f"  Homeostatic drift: {drift_before:.4f}")
            print(f"  Loss before sleep: {loss:.4f}")
            print()

        # Apply homeostatic reset
        model.apply_homeostatic_reset()

        # Get metrics after sleep
        metrics_after = model.get_phase_coherence_metrics()
        drift_after = metrics_after['homeostatic_drift']

        if self.verbose:
            print(f"  Homeostatic drift after: {drift_after:.4f}")
            print(f"  Drift reduction: {drift_before - drift_after:.4f}")
            print("=" * 70)
            print()

        # Record sleep event
        sleep_event = SleepEvent(
            step=step,
            drift_before=drift_before,
            drift_after=drift_after,
            loss_before=loss,
        )
        self._sleep_events.append(sleep_event)

        # Setup consolidation tracking
        if self.measure_consolidation:
            self._consolidation_window[step] = {
                'event_idx': len(self._sleep_events) - 1,
                'steps_after': 0,
                'min_loss': float('inf'),
            }

        # Update last sleep step
        self._last_sleep_step = step
        self._steps_since_sleep = 0

    def _update_consolidation_tracking(self, step: int, loss: float):
        """Track loss evolution after sleep for consolidation analysis."""

        # Find active consolidation windows
        completed_windows = []

        for sleep_step, window in self._consolidation_window.items():
            steps_after = step - sleep_step
            window['steps_after'] = steps_after

            # Update min loss
            if loss < window['min_loss']:
                window['min_loss'] = loss

            # Get the sleep event
            event_idx = window['event_idx']
            event = self._sleep_events[event_idx]

            # Record loss at specific checkpoints
            if steps_after == 1 and event.loss_after_1 is None:
                event.loss_after_1 = loss
            elif steps_after == 10 and event.loss_after_10 is None:
                event.loss_after_10 = loss
            elif steps_after == 50 and event.loss_after_50 is None:
                event.loss_after_50 = loss
                event.min_loss_after = window['min_loss']
                event.consolidation_benefit = event.loss_before - window['min_loss']

                # Report consolidation
                if self.verbose:
                    self._report_consolidation(event, sleep_step)

                # Mark for removal
                completed_windows.append(sleep_step)

        # Remove completed windows
        for sleep_step in completed_windows:
            del self._consolidation_window[sleep_step]

    def _report_consolidation(self, event: SleepEvent, sleep_step: int):
        """Report memory consolidation results."""
        print()
        print("[CONSOLIDATION] MEMORY CONSOLIDATION ANALYSIS")
        print(f"   Sleep event at step {sleep_step}")
        print(f"   Loss before sleep:  {event.loss_before:.4f}")

        # Only print if we have the data
        if event.loss_after_1 is not None:
            print(f"   Loss after 1 step:  {event.loss_after_1:.4f} ({event.loss_after_1 - event.loss_before:+.4f})")
        if event.loss_after_10 is not None:
            print(f"   Loss after 10 steps: {event.loss_after_10:.4f} ({event.loss_after_10 - event.loss_before:+.4f})")
        print(f"   Loss after 50 steps: {event.loss_after_50:.4f} ({event.loss_after_50 - event.loss_before:+.4f})")
        print(f"   Min loss (0-50):     {event.min_loss_after:.4f}")

        if event.consolidation_benefit > 0:
            print(f"   [OK] CONSOLIDATION: {event.consolidation_benefit:.4f} improvement!")
        else:
            print(f"   [X] NO CONSOLIDATION: {abs(event.consolidation_benefit):.4f} worse")

        print()

    def get_statistics(self) -> Dict:
        """Get sleep cycle statistics."""

        if not self._sleep_events:
            return {
                'total_sleeps': 0,
                'avg_steps_between_sleeps': 0,
                'avg_drift_before': 0,
                'avg_drift_reduction': 0,
                'consolidation_success_rate': 0,
                'avg_consolidation_benefit': 0,
            }

        # Calculate statistics
        total_sleeps = len(self._sleep_events)
        steps_between = [
            self._sleep_events[i+1].step - self._sleep_events[i].step
            for i in range(len(self._sleep_events) - 1)
        ]
        avg_steps_between = sum(steps_between) / len(steps_between) if steps_between else 0

        drift_before = [e.drift_before for e in self._sleep_events]
        drift_reduction = [e.drift_before - e.drift_after for e in self._sleep_events]

        # Consolidation stats (only for completed windows)
        consolidation_events = [e for e in self._sleep_events if e.consolidation_benefit is not None]
        successful_consolidations = [e for e in consolidation_events if e.consolidation_benefit > 0]

        return {
            'total_sleeps': total_sleeps,
            'avg_steps_between_sleeps': avg_steps_between,
            'avg_drift_before': sum(drift_before) / len(drift_before),
            'avg_drift_reduction': sum(drift_reduction) / len(drift_reduction),
            'consolidation_success_rate': len(successful_consolidations) / len(consolidation_events) if consolidation_events else 0,
            'avg_consolidation_benefit': sum(e.consolidation_benefit for e in successful_consolidations) / len(successful_consolidations) if successful_consolidations else 0,
        }

    def print_summary(self):
        """Print summary of sleep cycles."""
        stats = self.get_statistics()

        print()
        print("=" * 70)
        print("SLEEP CYCLE SUMMARY")
        print("=" * 70)
        print(f"Total sleep cycles: {stats['total_sleeps']}")
        print(f"Avg steps between sleeps: {stats['avg_steps_between_sleeps']:.0f}")
        print(f"Avg drift before sleep: {stats['avg_drift_before']:.4f}")
        print(f"Avg drift reduction: {stats['avg_drift_reduction']:.4f}")
        print()

        if stats['consolidation_success_rate'] > 0:
            print("MEMORY CONSOLIDATION:")
            print(f"  Success rate: {stats['consolidation_success_rate']:.1%}")
            print(f"  Avg benefit: {stats['avg_consolidation_benefit']:.4f}")
            print()

        if self._sleep_events:
            print("Recent sleep events:")
            for event in self._sleep_events[-5:]:  # Last 5
                print(f"  Step {event.step}: drift {event.drift_before:.4f} → {event.drift_after:.4f}")
                if event.consolidation_benefit is not None:
                    sign = "[OK]" if event.consolidation_benefit > 0 else "[X]"
                    print(f"    {sign} Consolidation: {event.consolidation_benefit:+.4f}")

        print("=" * 70)
        print()


def run_fatigue_test(
    model,
    train_loader,
    optimizer,
    num_steps: int = 1000,
    device: str = 'cuda',
):
    """
    Run fatigue test: train without sleep, then sleep and observe consolidation.

    This validates that "neural sleep" actually helps learning.

    Args:
        model: COSFOscillatorTransformer
        train_loader: Training data loader
        optimizer: Optimizer
        num_steps: Number of steps to train (default: 1000)
        device: Device to use

    Returns:
        Dict with test results
    """
    print()
    print("=" * 70)
    print("FATIGUE TEST - NEURAL SLEEP VALIDATION")
    print("=" * 70)
    print()
    print("Protocol:")
    print("  1. Train for 1,000 steps WITHOUT sleep")
    print("  2. Measure homeostatic drift")
    print("  3. Trigger sleep (homeostatic reset)")
    print("  4. Observe loss for 50 steps after wake")
    print("  5. Validate memory consolidation hypothesis")
    print()
    print("Hypothesis:")
    print("  If loss improves after sleep beyond pre-sleep minimum,")
    print("  then neural sleep provides memory consolidation benefit.")
    print()
    print("=" * 70)
    print()

    model.train()
    losses_before_sleep = []
    losses_after_sleep = []

    # Phase 1: Train without sleep
    print("Phase 1: Training without sleep (1000 steps)...")
    data_iter = iter(train_loader)

    for step in range(num_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        if isinstance(batch, dict):
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
        else:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

        logits, loss = model(inputs, targets)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # NOTE: Homeostatic decay is NOT applied during regular training
        # It only happens during sleep (homeostatic reset)
        # Applying decay every step causes gradient version conflicts

        losses_before_sleep.append(loss.item())

        if (step + 1) % 100 == 0:
            metrics = model.get_phase_coherence_metrics()
            print(f"  Step {step+1:4d}: Loss={loss.item():.4f}, Drift={metrics['homeostatic_drift']:.4f}")

    # Get final metrics before sleep
    metrics_before = model.get_phase_coherence_metrics()
    loss_before_sleep = losses_before_sleep[-1]
    min_loss_before = min(losses_before_sleep)

    print()
    print("Phase 1 Complete:")
    print(f"  Final loss: {loss_before_sleep:.4f}")
    print(f"  Min loss: {min_loss_before:.4f}")
    print(f"  Homeostatic drift: {metrics_before['homeostatic_drift']:.4f}")
    print()

    # Phase 2: Sleep
    print("Phase 2: Triggering neural sleep...")
    print()
    model.apply_homeostatic_reset()

    metrics_after = model.get_phase_coherence_metrics()
    print(f"  Drift after sleep: {metrics_after['homeostatic_drift']:.4f}")
    print(f"  Drift reduction: {metrics_before['homeostatic_drift'] - metrics_after['homeostatic_drift']:.4f}")
    print()

    # Phase 3: Train after sleep
    print("Phase 3: Training after wake (50 steps)...")

    for step in range(50):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        if isinstance(batch, dict):
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
        else:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

        logits, loss = model(inputs, targets)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # NOTE: Homeostatic decay is NOT applied during regular training
        # It only happens during sleep (homeostatic reset)
        # Applying decay every step causes gradient version conflicts

        losses_after_sleep.append(loss.item())

        if (step + 1) % 10 == 0:
            print(f"  Step {step+1:2d}: Loss={loss.item():.4f}")

    # Phase 4: Analysis
    print()
    print("=" * 70)
    print("FATIGUE TEST RESULTS")
    print("=" * 70)
    print()

    loss_after_1 = losses_after_sleep[0] if losses_after_sleep else 0
    loss_after_10 = losses_after_sleep[9] if len(losses_after_sleep) > 9 else 0
    loss_after_50 = losses_after_sleep[-1] if losses_after_sleep else 0
    min_loss_after = min(losses_after_sleep) if losses_after_sleep else 0

    print("Loss trajectory:")
    print(f"  Before sleep: {loss_before_sleep:.4f}")
    print(f"  After 1 step: {loss_after_1:.4f} ({loss_after_1 - loss_before_sleep:+.4f})")
    print(f"  After 10 steps: {loss_after_10:.4f} ({loss_after_10 - loss_before_sleep:+.4f})")
    print(f"  After 50 steps: {loss_after_50:.4f} ({loss_after_50 - loss_before_sleep:+.4f})")
    print()

    print("Consolidation analysis:")
    print(f"  Min loss before sleep: {min_loss_before:.4f}")
    print(f"  Min loss after sleep:  {min_loss_after:.4f}")

    consolidation_benefit = min_loss_before - min_loss_after

    if consolidation_benefit > 0:
        print(f"  [OK] CONSOLIDATION CONFIRMED: {consolidation_benefit:.4f} improvement!")
        print()
        print("  Neural sleep provides memory consolidation benefit.")
        print("  The model learned better after resting.")
        result = "success"
    else:
        print(f"  [X] NO CONSOLIDATION: {abs(consolidation_benefit):.4f} worse")
        print()
        print("  Neural sleep did not improve learning in this test.")
        print("  May need more steps or different threshold.")
        result = "no_benefit"

    print()
    print("=" * 70)
    print()

    return {
        'result': result,
        'loss_before_sleep': loss_before_sleep,
        'min_loss_before': min_loss_before,
        'loss_after_1': loss_after_1,
        'loss_after_10': loss_after_10,
        'loss_after_50': loss_after_50,
        'min_loss_after': min_loss_after,
        'consolidation_benefit': consolidation_benefit,
        'drift_before': metrics_before['homeostatic_drift'],
        'drift_after': metrics_after['homeostatic_drift'],
    }


# Example usage
if __name__ == "__main__":
    print("""
COSF Sleep Cycle Hook - Example Usage

1. Basic Usage:
    hook = SleepCycleHook(drift_threshold=2.0)

    for step in range(10000):
        logits, loss = model(inputs, targets)
        loss.backward()
        optimizer.step()

        # Auto sleep when needed
        hook.check_and_sleep(model, step, loss.item())

2. Custom Thresholds:
    hook = SleepCycleHook(
        drift_threshold=1.5,      # Sleep when drift > 1.5
        min_steps_awake=1000,     # At least 1000 steps between sleeps
        max_steps_awake=10000,    # Force sleep after 10k steps
    )

3. Run Fatigue Test:
    results = run_fatigue_test(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        num_steps=1000,
    )

    if results['result'] == 'success':
        print("Neural sleep validated!")
    """)
