"""
COSF Jarvis Interactive CLI

A command-line interface to interact with the COSF-powered Jarvis agent.
Integrates Intent Routing, Memory System, and Stability Monitoring.

Usage:
    python jarvis_cli.py

Commands:
    /exit, /quit - Exit the CLI
    /memory      - Show recent memories
    /status      - Show system status
    /clear       - Clear memory (with confirmation)
"""

import os
import sys
from pathlib import Path

# Add COSF-MI to path
cosf_path = Path(__file__).parent / "COSF-MI"
sys.path.insert(0, str(cosf_path))

# Also add the src/python path for the checkpoint to find cosf_mi module
src_python_path = Path(__file__).parent / "COSF-MI" / "src" / "python"
sys.path.insert(0, str(src_python_path))

import torch
import torch.nn.functional as F
from src.python.cosf_mi.agent.intent_router import SimpleIntentRouter, Intent
from src.python.cosf_mi.agent.memory import COSFMemory
from src.python.cosf_mi.agent.stability_governor import JuliaGovernor
from src.python.cosf_mi.models.cosf_transformer import COSFConfig, COSFTransformer
from src.python.cosf_mi.data.tokenizer import Tokenizer, BPETokenizer


class JarvisCLI:
    """Interactive Jarvis CLI with COSF integration."""

    def __init__(
        self,
        model_path: str = None,
        memory_path: str = "memory/jarvis",
    ):
        """Initialize Jarvis CLI.

        Args:
            model_path: Path to trained model checkpoint
            memory_path: Path to memory database
        """
        print("=" * 70)
        print("INITIALIZING COSF JARVIS")
        print("=" * 70)
        print()

        # Load model first to get config
        print("[1/5] Loading COSF transformer model...")
        self.model, self.config = self._load_model(model_path)
        print(f"      Model loaded: {self.config.n_layers} layers, {self.config.d_model} dim, {self.config.vocab_size} vocab")

        # Initialize tokenizer based on vocab size
        print("[2/5] Loading tokenizer...")
        if self.config.vocab_size > 256:
            # BPE tokenizer for TinyStories model
            tokenizer_path = "COSF-MI/models/tokenizer_8k.model"
            if not Path(tokenizer_path).exists():
                tokenizer_path = "models/tokenizer_8k.model"
            self.tokenizer = BPETokenizer(tokenizer_path)
            print(f"      BPE tokenizer ready (vocab_size={self.tokenizer.vocab_size})")
        else:
            # Character-level for Shakespeare model
            self.tokenizer = Tokenizer.char(vocab_size=256)
            print("      Character-level tokenizer ready (vocab_size=256)")

        # Initialize Intent Router
        print("[3/5] Initializing Intent Router...")
        self.router = SimpleIntentRouter()
        print("      Intent router ready (4 intent types)")

        # Initialize Memory System
        print("[4/5] Initializing Memory System...")
        self.memory = COSFMemory(db_path=memory_path)
        stats = self.memory.stats()
        print(f"      Memory system ready ({stats['total_memories']} memories loaded)")

        # Initialize Stability Governor
        print("[5/5] Initializing Stability Governor...")
        # Bounds: 1.5 for well-trained models, 4.0 for experimental/Shakespeare models
        julia_bounds = 1.5 if self.config.vocab_size > 256 else 4.0
        self.governor = JuliaGovernor(self.model, bounds=julia_bounds)
        print(f"      Stability governor ready (Julia bounds = {julia_bounds})")

        print()
        print("=" * 70)
        print("JARVIS IS ONLINE")
        print("=" * 70)
        print()

        # Conversation state
        self.conversation_history = []
        self.turn_count = 0

    def _load_model(self, model_path: str = None):
        """Load trained COSF model from checkpoint.

        Args:
            model_path: Path to checkpoint file

        Returns:
            Tuple of (model, config)
        """
        # Find checkpoint
        if model_path is None:
            # Priority order:
            # 1. TinyStories trained model (cosf_large)
            # 2. Old Shakespeare model (cosf_transformer)

            # Check for TinyStories model first
            checkpoint_dirs = [
                Path("COSF-MI/checkpoints/cosf_large"),
                Path("checkpoints/cosf_large"),
                Path("COSF-MI/checkpoints/cosf_transformer"),
                Path("checkpoints/cosf_transformer"),
            ]

            checkpoint_dir = None
            for dir_path in checkpoint_dirs:
                if dir_path.exists():
                    checkpoint_dir = dir_path
                    break

            if checkpoint_dir is None:
                raise FileNotFoundError(
                    "No checkpoint directory found. Train a model first:\n"
                    "  train_tinystories_small.bat"
                )

            if (checkpoint_dir / "best_model.pt").exists():
                model_path = checkpoint_dir / "best_model.pt"
            else:
                # Find latest checkpoint
                checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
                if checkpoints:
                    # Sort by epoch number
                    checkpoints.sort(key=lambda p: int(p.stem.split('_')[-1]))
                    model_path = checkpoints[-1]
                else:
                    raise FileNotFoundError(
                        f"No model checkpoints found in {checkpoint_dir}"
                    )

        print(f"      Loading from: {model_path}")

        # Load checkpoint (weights_only=False for custom classes)
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

        # Extract config from checkpoint
        if 'config' in checkpoint:
            config_obj = checkpoint['config']
            # Check if it's already a COSFConfig object or a dict
            if isinstance(config_obj, COSFConfig):
                config = config_obj
            elif isinstance(config_obj, dict):
                config = COSFConfig(**config_obj)
            else:
                # Try to convert to dict
                config = COSFConfig(**vars(config_obj))
        else:
            # Use default config
            config = COSFConfig(vocab_size=256)

        # Create model
        model = COSFTransformer(config)

        # Load weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.eval()

        return model, config

    def _format_intent(self, intent: Intent) -> str:
        """Format intent for display."""
        intent_colors = {
            Intent.CONVERSATION: "CONVERSATION",
            Intent.SYSTEM_TASK: "SYSTEM_TASK",
            Intent.KNOWLEDGE_SEARCH: "KNOWLEDGE_SEARCH",
            Intent.MATH_REASONING: "MATH_REASONING",
        }
        return intent_colors.get(intent, str(intent))

    def _generate_response(self, user_input: str, max_length: int = 50, temperature: float = 0.5) -> str:
        """Generate response using COSF model.

        Args:
            user_input: User's input text
            max_length: Maximum generation length (default: 50, reduced for coherence)
            temperature: Sampling temperature (default: 0.5, lower = more focused)

        Returns:
            Generated response text
        """
        # Encode input
        input_ids = self.tokenizer.encode(user_input)
        input_tensor = torch.tensor([input_ids], dtype=torch.long)

        # Generate
        self.model.eval()
        with torch.no_grad():
            # Get initial output
            output = self.model(input_tensor)

            # Monitor stability
            stability = self.governor.monitor_inference(output)

            # Simple temperature-based sampling
            generated_ids = input_ids.copy()

            for _ in range(max_length):
                # Get logits for next token
                current_tensor = torch.tensor([generated_ids], dtype=torch.long)
                logits = self.model(current_tensor)

                # Monitor stability at each step
                stability = self.governor.monitor_inference(logits)

                # Apply temperature and get next token
                next_token_logits = logits[0, -1, :] / temperature

                # Use temperature sampling instead of greedy for better quality
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()

                # Stop on newline or period (simple stopping)
                if next_token in [ord('\n'), ord('.')] and len(generated_ids) > len(input_ids) + 15:
                    generated_ids.append(next_token)
                    break

                generated_ids.append(next_token)

                # Stop if too long
                if len(generated_ids) >= len(input_ids) + max_length:
                    break

        # Decode response (skip the input part)
        full_text = self.tokenizer.decode(generated_ids)
        response = full_text[len(user_input):].strip()

        return response, stability

    def _inject_memory_context(self, user_input: str, retrieved_memories: list) -> str:
        """Add retrieved memories as context to input.

        Args:
            user_input: Original user input
            retrieved_memories: List of retrieved memory dicts

        Returns:
            Input with memory context prepended
        """
        if not retrieved_memories:
            return user_input

        # Format memory context
        context_parts = ["[Memory Context]"]
        for i, mem in enumerate(retrieved_memories[:2], 1):  # Top 2 memories
            context_parts.append(f"{i}. {mem['text'][:100]}")

        context_parts.append("[User Input]")
        context_parts.append(user_input)

        return "\n".join(context_parts)

    def process_turn(self, user_input: str):
        """Process a single conversational turn.

        Args:
            user_input: User's input text
        """
        self.turn_count += 1

        print()
        print("-" * 70)
        print(f"Turn {self.turn_count}")
        print("-" * 70)

        # Step 1: Intent Classification
        intent = self.router.route(user_input)
        print(f"Intent: {self._format_intent(intent)}")

        # Step 2: Memory Retrieval
        print("Retrieving memories...", end=" ")
        retrieved = self.memory.retrieve(user_input, top_k=3, phi_weighted=True)
        print(f"({len(retrieved)} found)")

        if retrieved:
            print("  Relevant memories:")
            for i, mem in enumerate(retrieved[:2], 1):
                snippet = mem['text'][:60] + "..." if len(mem['text']) > 60 else mem['text']
                print(f"    {i}. {snippet} (distance: {mem['distance']:.3f})")

        # Step 3: Generate Response
        print("Generating response...")

        # Add memory context to input
        contextualized_input = self._inject_memory_context(user_input, retrieved)

        # Generate (reduced max_length to 50 for more coherent output)
        response, gen_stability = self._generate_response(contextualized_input, max_length=50)

        # Step 4: Get overall stability status
        stability = self.governor.get_stability_metrics()

        # Step 5: Apply fallback if unstable
        if stability['status'] == 'unstable' or stability.get('avg_exceed_pct', 0) > 0.8:
            # Determine model type for better error message
            model_type = "conversational" if self.config.vocab_size > 256 else "Shakespeare"
            fallback_response = (
                "I'm experiencing instability in my response generation. "
                f"This might indicate the model needs more training. "
                f"(Current model: {model_type}, stability: {stability['status']})"
            )
            # Show both the attempted response and fallback
            print()
            print(f"Jarvis (raw): {response[:100]}...")
            print(f"Jarvis (fallback): {fallback_response}")
            # Use fallback for storage
            final_response = fallback_response
        else:
            final_response = response
            print()
            print(f"Jarvis: {final_response}")

        # Step 6: Store in Memory
        self.memory.store(
            f"User: {user_input}\nJarvis: {final_response}",
            metadata={'turn': self.turn_count},
            source='conversation'
        )

        # Step 7: Display Stability Status
        status_symbol = {
            'stable': '[OK]',
            'warning': '[WARNING]',
            'unstable': '[UNSTABLE]',
            'no_data': '[NO_DATA]',
        }.get(stability['status'], '[UNKNOWN]')

        print()
        print(f"Stability: {status_symbol} {stability['status'].upper()}")
        if stability['status'] != 'no_data':
            print(f"  Avg exceed: {stability['avg_exceed_pct']:.1%}")
            print(f"  Corrections: {stability['corrections_applied']}")
            if stability['avg_exceed_pct'] > 0.8:
                print(f"  Note: High instability - using fallback responses")

        # Store in conversation history
        self.conversation_history.append({
            'turn': self.turn_count,
            'user': user_input,
            'intent': intent,
            'response': final_response,
            'stability': stability,
        })

    def show_memory(self):
        """Display recent memories."""
        print()
        print("=" * 70)
        print("RECENT MEMORIES")
        print("=" * 70)

        stats = self.memory.stats()
        print(f"Total memories: {stats['total_memories']}")
        print()

        # Get recent conversation memories
        recent = self.memory.retrieve("conversation", top_k=5, source_filter='conversation')

        if recent:
            for i, mem in enumerate(recent, 1):
                print(f"{i}. {mem['text'][:200]}...")
                print(f"   Source: {mem['metadata']['source']}")
                print()
        else:
            print("No memories found.")

        print("=" * 70)

    def show_status(self):
        """Display system status."""
        print()
        print("=" * 70)
        print("JARVIS SYSTEM STATUS")
        print("=" * 70)

        # Model info
        print(f"Model: COSF Transformer")
        print(f"  Layers: {self.config.n_layers}")
        print(f"  Dimension: {self.config.d_model}")
        print(f"  Vocab: {self.config.vocab_size}")
        print()

        # Memory stats
        mem_stats = self.memory.stats()
        print(f"Memory System:")
        print(f"  Total memories: {mem_stats['total_memories']}")
        print(f"  Database: {mem_stats['db_path']}")
        print()

        # Stability metrics
        stability = self.governor.get_stability_metrics()
        print(f"Stability Governor:")
        print(f"  Status: {stability['status'].upper()}")
        if stability['status'] != 'no_data':
            print(f"  Samples: {stability['sample_count']}")
            print(f"  Avg exceed: {stability['avg_exceed_pct']:.1%}")
            print(f"  Corrections: {stability['corrections_applied']}")
        print()

        # Session info
        print(f"Session:")
        print(f"  Turns: {self.turn_count}")
        print(f"  Conversation history: {len(self.conversation_history)} entries")

        print("=" * 70)

    def clear_memory(self):
        """Clear all memories with confirmation."""
        print()
        confirm = input("Are you sure you want to clear all memories? (yes/no): ")
        if confirm.lower() == 'yes':
            self.memory.clear()
            print("Memory cleared.")
        else:
            print("Cancelled.")

    def run(self):
        """Run the interactive CLI loop."""
        print("Type your message and press Enter. Use /help for commands.")
        print()

        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.lower() in ['/exit', '/quit']:
                    print()
                    print("Shutting down Jarvis...")
                    print("Goodbye!")
                    break

                elif user_input.lower() == '/memory':
                    self.show_memory()
                    continue

                elif user_input.lower() == '/status':
                    self.show_status()
                    continue

                elif user_input.lower() == '/clear':
                    self.clear_memory()
                    continue

                elif user_input.lower() == '/help':
                    print()
                    print("Commands:")
                    print("  /exit, /quit - Exit the CLI")
                    print("  /memory      - Show recent memories")
                    print("  /status      - Show system status")
                    print("  /clear       - Clear memory (with confirmation)")
                    print("  /help        - Show this help")
                    print()
                    continue

                # Process normal input
                self.process_turn(user_input)

            except KeyboardInterrupt:
                print()
                print()
                print("Interrupted. Use /exit to quit.")
                print()
            except Exception as e:
                print()
                print(f"Error: {e}")
                print()
                import traceback
                traceback.print_exc()


def main():
    """Main entry point."""
    try:
        # Initialize Jarvis
        jarvis = JarvisCLI()

        # Run interactive loop
        jarvis.run()

    except Exception as e:
        print()
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
