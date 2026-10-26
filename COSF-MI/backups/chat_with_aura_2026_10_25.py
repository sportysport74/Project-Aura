"""
Chat with AURA - Interactive Conversation with Your Trained Model

This lets you have a back-and-forth conversation with your trained COSF model.

Two modes, picked from the checkpoint:
  - chat model (made by finetune_chat.py): real conversation, AURA remembers
    the last few turns ("You: ... Aura: ...")
  - story model: AURA continues whatever you type as a story

Usage:
    python chat_with_aura.py [checkpoint]
"""

import sys
import torch
from pathlib import Path

# Add COSF-MI to path
sys.path.insert(0, str(Path(__file__).parent / "src" / "python"))

from cosf_mi.models.cosf_oscillator import COSFOscillatorTransformer
from cosf_mi.data.tokenizer import BPETokenizer


DEFAULT_CHECKPOINTS = [
    "checkpoints/aura_chat_smart10/best_model.pt",  # chat model on the 10-epoch mix base (chat val 1.879)
    "checkpoints/aura_chat_smart/best_model.pt",  # chat model on the big mix-pretrained base (wiki/web/textbook/code/math)
    "checkpoints/aura_chat_mid2/best_model.pt",   # mid chat model, malformed [INST] turns removed
    "checkpoints/aura_chat_mid/best_model.pt",    # first mid chat model (leaks [/INST] sometimes)
    "checkpoints/aura_chat_tiny/best_model.pt",   # chat model on the 10-epoch tiny base
    "checkpoints/fixed_run_100ep/best_model.pt",  # story model
]


def load_model(checkpoint_path: str = None):
    """Load trained model from checkpoint."""
    if checkpoint_path is None:
        checkpoint_path = next((c for c in DEFAULT_CHECKPOINTS if Path(c).exists()), DEFAULT_CHECKPOINTS[-1])
    print(f"Loading model: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    config = checkpoint['config']
    model = COSFOscillatorTransformer(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"  Epoch: {checkpoint['epoch'] + 1}")
    print(f"  Val loss: {checkpoint['val_loss']:.4f}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model, config, checkpoint.get('chat_format')


def generate_chat_reply(model, tokenizer, history, user_input, chat_format,
                        max_tokens=80, temperature=0.7, top_k=40, device='cuda'):
    """
    Aura's next reply. The prompt is built piece by piece exactly as
    finetune_chat.py built its training data, and generation stops when the
    model writes the user marker ("You:"), which ends Aura's turn.
    """
    user_ids = tokenizer.encode(chat_format['user'])
    aura_ids = tokenizer.encode(chat_format['aura'])
    ids = []
    for u, a in history:
        ids += user_ids + tokenizer.encode(u) + aura_ids + tokenizer.encode(a)
    ids += user_ids + tokenizer.encode(user_input) + aura_ids

    model.eval()
    model = model.to(device)
    input_ids = torch.tensor([ids[-256:]], dtype=torch.long, device=device)
    reply = []
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(input_ids[:, -256:])[0, -1, :] / temperature
            if top_k:
                kth = torch.topk(logits, top_k).values[-1]
                logits[logits < kth] = -float('inf')
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), 1)
            reply.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            if reply[-len(user_ids):] == user_ids:
                reply = reply[:-len(user_ids)]
                break
    return tokenizer.decode(reply).strip()


def generate_response(model, tokenizer, prompt, max_tokens=150, temperature=0.7, device='cuda'):
    """Generate text response from prompt."""
    model.eval()
    model = model.to(device)

    # Encode prompt
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)

    generated = []
    generated_ids = []

    with torch.no_grad():
        for _ in range(max_tokens):
            # Forward pass (no targets = inference mode, returns only logits)
            logits = model(input_ids[:, -256:])  # Use last 256 tokens

            # Sample next token
            logits = logits[0, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Decode token
            token_text = tokenizer.decode([next_token.item()])
            generated.append(token_text)
            generated_ids.append(next_token.item())

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

            # Stop at end of sentence (rough heuristic)
            if token_text.strip() in ['.', '!', '?'] and len(generated) > 20:
                # Generate a bit more to complete the thought
                if len(generated) > 30:
                    break

    # Decode all at once - per-token decode drops the word-boundary spaces
    return tokenizer.decode(generated_ids)


def main():
    print()
    print("=" * 70)
    print("CHAT WITH AURA")
    print("=" * 70)
    print()
    print("Interactive conversation with your trained COSF model")
    print()

    # Load model
    model, config, chat_format = load_model(*sys.argv[1:2])

    # Load tokenizer
    print()
    print("Loading tokenizer...")
    tokenizer = BPETokenizer("models/tokenizer_mix16k.model" if config.vocab_size == 16000
                             else "models/tokenizer_8k.model")
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    # Get model stats
    print()
    print("=" * 70)
    print("MODEL STATISTICS")
    print("=" * 70)

    metrics = model.get_phase_coherence_metrics()
    print()
    print("Phase Metrics:")
    print(f"  Mean coherence: {metrics['mean_coherence']:.4f}")
    print(f"  Max phase velocity: {metrics['max_phase_velocity']:.4f}")
    print(f"  Homeostatic drift: {metrics['homeostatic_drift']:.4f}")

    if metrics.get('chirality_enabled', False):
        print()
        print("Chirality Metrics:")
        print(f"  Right-handed: {metrics['total_right_handed']}")
        print(f"  Left-handed: {metrics['total_left_handed']}")
        print(f"  Balance: {metrics['chirality_balance']:.3f}")
        print(f"  Total flips: {metrics['total_chirality_flips']}")

    # Chat settings
    print()
    print("=" * 70)
    print("CHAT SETTINGS")
    print("=" * 70)
    print()
    if chat_format:
        print("  Mode: CHAT (AURA remembers the last 4 turns)")
        print("  Temperature: 0.7, top-k 40")
        print()
        print("  Tips:")
        print("    - Talk to it like a friend, in simple sentences")
        print("    - Type 'reset' to start a new conversation")
    else:
        print("  Mode: STORY (AURA continues what you type)")
        print("  Max tokens per response: 150")
        print("  Temperature: 0.7 (balanced)")
        print("  Style token: [STORY]")
        print()
        print("  Tips:")
        print("    - Start prompts with '[STORY]' for story-style text")
        print("    - Use complete sentences for better responses")
    print("    - Type 'quit', 'exit', or 'bye' to end chat")
    print("    - Type 'stats' to see current model metrics")
    print("    - Type 'temp X' to change temperature (0.1-1.5)")
    print()

    # Chat loop
    print("=" * 70)
    print("START CHATTING")
    print("=" * 70)
    print()

    conversation_history = []  # (you, aura) turns, chat mode only
    temperature = 0.7

    while True:
        # Get user input
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue

        # Check for commands
        if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
            print()
            print("Goodbye! Thanks for chatting with AURA!")
            break

        if user_input.lower() == 'stats':
            metrics = model.get_phase_coherence_metrics()
            print()
            print("Current Model Stats:")
            print(f"  Coherence: {metrics['mean_coherence']:.4f}")
            print(f"  Drift: {metrics['homeostatic_drift']:.4f}")
            if metrics.get('chirality_enabled', False):
                print(f"  Chirality balance: {metrics['chirality_balance']:.3f}")
                print(f"  Chirality flips: {metrics['total_chirality_flips']}")
            print()
            continue

        if user_input.lower() == 'reset':
            conversation_history = []
            print("(New conversation)")
            print()
            continue

        if chat_format and not user_input.lower().startswith('temp '):
            reply = generate_chat_reply(model, tokenizer, conversation_history[-4:], user_input,
                                        chat_format, temperature=temperature, device=device)
            print(f"AURA: {reply}")
            print()
            conversation_history.append((user_input, reply))
            continue

        if user_input.lower().startswith('temp '):
            try:
                new_temp = float(user_input.split()[1])
                if 0.1 <= new_temp <= 1.5:
                    temperature = new_temp
                    print(f"Temperature set to {temperature}")
                else:
                    print("Temperature must be between 0.1 and 1.5")
            except (ValueError, IndexError):
                print("Usage: temp 0.7")
            print()
            continue

        # Prepare prompt (add [STORY] if not present)
        if not user_input.startswith('['):
            prompt = f"[STORY] {user_input}"
        else:
            prompt = user_input

        # Generate response
        print()
        print("AURA: ", end='', flush=True)

        try:
            response = generate_response(
                model, tokenizer, prompt,
                max_tokens=150,
                temperature=temperature,
                device=device
            )
            print(response)
        except Exception as e:
            print(f"\n[Error generating response: {e}]")

        print()

    print()
    print("=" * 70)
    print("CHAT SESSION ENDED")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
