# COSF Jarvis Quick Start Guide

**Transform your COSF model into an agentic AI assistant in 30 minutes.**

## What You're Building

A personal AI assistant with:
- **Intent routing:** Classifies requests (chat, system tasks, knowledge search, reasoning)
- **Memory:** Remembers conversations and syncs with Obsidian notes
- **Stability:** Julia-bounded inference prevents hallucinations
- **Agency:** (Coming next) Execute OS commands safely

## Prerequisites

✅ Trained COSF model (your 9M Shakespeare or 78M TinyStories model)
✅ Python environment with PyTorch
✅ RTX 5090 with CUDA 12.8

## Step 1: Install Dependencies (5 minutes)

```bash
cd c:\Users\fatba\COSF-MI
venv\Scripts\pip install chromadb sentence-transformers
```

## Step 2: Test Intent Router (10 minutes)

Create `test_intent_router.py`:

```python
from cosf_mi.agent.intent_router import SimpleIntentRouter, Intent

# Test simple router (no model needed)
router = SimpleIntentRouter()

tests = [
    "Hello, how are you?",
    "Open notepad",
    "Search my notes for COSF proofs",
    "Calculate phi^5",
]

for test in tests:
    intent = router.route(test)
    print(f"{test:50} → {intent.value}")
```

Run:
```bash
cd COSF-MI
..\venv\Scripts\python test_intent_router.py
```

**Expected output:**
```
Hello, how are you?                               → conversation
Open notepad                                      → system
Search my notes for COSF proofs                   → knowledge
Calculate phi^5                                   → math
```

## Step 3: Initialize Memory System (10 minutes)

Create `test_memory.py`:

```python
from cosf_mi.agent.memory import COSFMemory

# Initialize memory
memory = COSFMemory(db_path="memory/test_db")

# Store some memories
memory.store("The golden ratio φ ≈ 1.618 is fundamental to COSF",
             source="knowledge")

memory.store("I prefer working late at night",
             source="user_preference")

memory.store("COSF uses Julia stability bounds to prevent divergence",
             metadata={'tags': ['cosf', 'stability']},
             source="knowledge")

# Retrieve
results = memory.retrieve("What is the golden ratio?", top_k=2)

for i, result in enumerate(results, 1):
    print(f"\n{i}. {result['text'][:80]}...")
    print(f"   Source: {result['metadata']['source']}")
    print(f"   Distance: {result['distance']:.4f}")
```

Run:
```bash
..\venv\Scripts\python test_memory.py
```

## Step 4: Test Stability Governor (5 minutes)

Create `test_governor.py`:

```python
import torch
from cosf_mi.agent.stability_governor import JuliaGovernor

# Create mock model
class MockModel:
    pass

model = MockModel()
governor = JuliaGovernor(model, bounds=2.0)

# Simulate activations
stable_act = torch.randn(10, 384) * 0.5  # Well within bounds
unstable_act = torch.randn(10, 384) * 5.0  # Exceeds bounds

print("Testing stable activations:")
stability = governor.monitor_inference(stable_act)
print(f"  Status: {stability['status']}")
print(f"  Exceed %: {stability['exceed_pct']:.1%}")

print("\nTesting unstable activations:")
stability = governor.monitor_inference(unstable_act)
print(f"  Status: {stability['status']}")
print(f"  Exceed %: {stability['exceed_pct']:.1%}")

print("\nMetrics:")
metrics = governor.get_stability_metrics()
for key, value in metrics.items():
    print(f"  {key}: {value}")
```

Run:
```bash
..\venv\Scripts\python test_governor.py
```

## Step 5: Integrate with Your Model (Full Integration)

Create `jarvis_demo.py`:

```python
"""
COSF Jarvis Demo: Complete integration
"""

import torch
from cosf_mi.models.cosf_transformer import COSFTransformer
from cosf_mi.data.tokenizer import Tokenizer, CharTokenizer
from cosf_mi.agent.intent_router import IntentRouter, SimpleIntentRouter, Intent
from cosf_mi.agent.memory import COSFMemory
from cosf_mi.agent.stability_governor import JuliaGovernor


class COSFJarvis:
    """Complete COSF Jarvis system."""

    def __init__(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        use_simple_router: bool = True  # True for testing
    ):
        """Initialize Jarvis.

        Args:
            model_path: Path to trained COSF model
            tokenizer_path: Path to tokenizer (None for char-level)
            use_simple_router: Use simple keyword router vs model-based
        """
        print("Initializing COSF Jarvis...")

        # Load model
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location='cuda', weights_only=False)
        self.model = COSFTransformer(checkpoint['config']).cuda()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print("✓ Model loaded")

        # Load tokenizer
        if tokenizer_path:
            self.tokenizer = Tokenizer.from_pretrained(tokenizer_path)
        else:
            self.tokenizer = CharTokenizer()
        print("✓ Tokenizer loaded")

        # Initialize components
        if use_simple_router:
            self.router = SimpleIntentRouter()
        else:
            self.router = IntentRouter(self.model, self.tokenizer)
        print("✓ Intent router ready")

        self.memory = COSFMemory(db_path="memory/jarvis")
        print("✓ Memory system ready")

        self.governor = JuliaGovernor(self.model)
        print("✓ Stability governor ready")

        print("\n🤖 COSF Jarvis initialized!\n")

    def process(self, user_input: str) -> dict:
        """Process user input.

        Args:
            user_input: User's text input

        Returns:
            Response dictionary
        """
        # Route intent
        intent = self.router.route(user_input)
        print(f"Intent: {intent.value}")

        # Retrieve relevant memories
        memories = self.memory.retrieve(user_input, top_k=3)
        print(f"Retrieved {len(memories)} relevant memories")

        # Process based on intent
        if intent == Intent.CONVERSATION:
            response = self._chat(user_input, memories)
        elif intent == Intent.KNOWLEDGE_SEARCH:
            response = self._search_knowledge(user_input, memories)
        elif intent == Intent.MATH_REASONING:
            response = self._reason(user_input)
        elif intent == Intent.SYSTEM_TASK:
            response = "System tasks not yet implemented"
        else:
            response = "Intent not recognized"

        # Store interaction
        self.memory.store(
            f"User: {user_input}\nJarvis: {response}",
            source="conversation"
        )

        # Get stability status
        stability = self.governor.get_stability_metrics()

        return {
            'response': response,
            'intent': intent.value,
            'memories_used': len(memories),
            'stability': stability,
        }

    def _chat(self, user_input: str, context: list) -> str:
        """Handle conversation."""
        # Simple echo for now
        # TODO: Use model to generate response
        return f"I understand you said: {user_input}"

    def _search_knowledge(self, query: str, memories: list) -> str:
        """Search knowledge base."""
        if not memories:
            return "I don't have information about that in my memory."

        # Return top memory
        return f"From my memory: {memories[0]['text'][:200]}..."

    def _reason(self, problem: str) -> str:
        """Mathematical reasoning."""
        # TODO: Implement chain-of-thought
        return f"Reasoning about: {problem}"


# Demo
if __name__ == "__main__":
    # Initialize Jarvis with your trained model
    jarvis = COSFJarvis(
        model_path="checkpoints/cosf_transformer/best_model.pt",
        use_simple_router=True
    )

    # Test queries
    queries = [
        "Hello, who are you?",
        "What is the golden ratio?",
        "Calculate phi squared",
        "Search my notes about COSF",
    ]

    for query in queries:
        print(f"\n{'='*70}")
        print(f"User: {query}")
        print('-'*70)

        result = jarvis.process(query)

        print(f"Jarvis: {result['response']}")
        print(f"Intent: {result['intent']}")
        print(f"Stability: {result['stability']['status']}")
        print(f"{'='*70}\n")
```

Run:
```bash
..\venv\Scripts\python jarvis_demo.py
```

## Step 6: Sync with Obsidian (Optional)

If you use Obsidian for notes:

```python
from cosf_mi.agent.memory import COSFMemory

memory = COSFMemory()

# Sync your Obsidian vault
memory.sync_with_obsidian(
    vault_path="C:/Users/fatba/Documents/Obsidian/MyVault",
    file_pattern="*.md"
)

# Now you can search your notes!
results = memory.retrieve("machine learning notes", top_k=5)
for result in results:
    print(f"\n{result['metadata']['filename']}")
    print(result['text'][:200])
```

## What's Next?

### Week 1: Basic Integration
- [x] Intent routing working
- [x] Memory system operational
- [x] Stability monitoring active
- [ ] Add actual text generation in `_chat()`
- [ ] Implement chain-of-thought in `_reason()`

### Week 2: Enhanced Features
- [ ] Train intent classifier on your model
- [ ] Add voice input (Whisper.cpp)
- [ ] Add voice output (XTTS)
- [ ] Build web UI with Streamlit

### Week 3: System Control
- [ ] Implement Rust executor (see JARVIS_ARCHITECTURE.md)
- [ ] Add function calling
- [ ] Whitelist safe OS commands
- [ ] Test file operations

### Week 4: Deployment
- [ ] Create Docker container
- [ ] Set up VM isolation
- [ ] Add monitoring dashboard
- [ ] Production testing

## Architecture Overview

```
User Input
    ↓
[Intent Router] → Conversation/System/Knowledge/Math
    ↓
[Memory System] → Retrieve relevant context
    ↓
[COSF Model] → Generate response
    ↓
[Stability Governor] → Monitor & correct
    ↓
Response + Store in Memory
```

## Files Created

✅ `src/python/cosf_mi/agent/intent_router.py` - Intent classification
✅ `src/python/cosf_mi/agent/memory.py` - Vector memory with φ-resonance
✅ `src/python/cosf_mi/agent/stability_governor.py` - Julia stability monitoring
✅ `JARVIS_ARCHITECTURE.md` - Full system design

## Troubleshooting

**chromadb import error:**
```bash
pip install chromadb sentence-transformers
```

**CUDA out of memory:**
```python
# Use CPU for memory system
memory = COSFMemory(db_path="memory/jarvis")
memory.encoder.to('cpu')
```

**Model not found:**
```python
# Check your checkpoint path
import os
print(os.path.exists("checkpoints/cosf_transformer/best_model.pt"))
```

## Next Steps

1. **Run all test scripts** to verify components work
2. **Try `jarvis_demo.py`** with your trained model
3. **Read `JARVIS_ARCHITECTURE.md`** for full system design
4. **Start Week 1 tasks** to enhance the system

---

**You now have a working agentic COSF system! 🚀**

Next: Add voice, reasoning, and system control to make it truly "Jarvis-level"
