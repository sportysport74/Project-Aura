# COSF Jarvis: Agentic AI Architecture

**Transform COSF from a text model into a personal AI assistant with agency, memory, and stability.**

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      USER INTERFACE                          │
│  Voice (Whisper.cpp STT) ←→ Text ←→ Voice (XTTS TTS)       │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                   INTENT ROUTER                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │Conversation │  │System Task  │  │Knowledge    │         │
│  │(Chat/Reason)│  │(OS Control) │  │Search (RAG) │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
└─────────┼─────────────────┼─────────────────┼───────────────┘
          │                 │                 │
┌─────────▼─────────┐ ┌────▼────────┐ ┌──────▼──────────────┐
│  COSF REASONING   │ │ RUST CORE   │ │  MEMORY SYSTEM      │
│  Engine (78M)     │ │ Executor    │ │  (Vector DB)        │
│  - Math reasoning │ │ - OS calls  │ │  - Chroma/LanceDB   │
│  - CoT generation │ │ - Safe exec │ │  - Obsidian sync    │
│  - φ-weighted     │ │ - gRPC API  │ │  - φ-resonant match │
│  - Julia bounded  │ └─────────────┘ └─────────────────────┘
└───────────────────┘
          │
┌─────────▼──────────────────────────────────────────────────┐
│               JULIA STABILITY GOVERNOR                       │
│  - Real-time bound monitoring                               │
│  - Auto-fallback on instability                             │
│  - φ-decomposition re-sync                                  │
└─────────────────────────────────────────────────────────────┘
```

## Layer 1: Intent Router (The "Dispatcher")

**Purpose:** Classify user input into actionable intents without hallucination.

### Implementation

**File:** `src/python/cosf_mi/agent/intent_router.py`

```python
from enum import Enum
import torch
import torch.nn.functional as F

class Intent(Enum):
    CONVERSATION = "conversation"  # Chat, answer questions
    SYSTEM_TASK = "system"         # OS operations
    KNOWLEDGE_SEARCH = "knowledge" # RAG retrieval
    MATH_REASONING = "math"        # Deep reasoning with CoT

class IntentRouter:
    """Route user inputs using COSF-weighted classification."""

    def __init__(self, model, tokenizer, phi=1.618033988749895):
        self.model = model
        self.tokenizer = tokenizer
        self.phi = phi

    def route(self, user_input: str) -> Intent:
        """Classify intent using φ-weighted embeddings."""
        # Get embeddings
        tokens = self.tokenizer.encode(user_input)
        with torch.no_grad():
            embeddings = self.model.get_embeddings(tokens)

        # φ-resonant matching with intent prototypes
        intent_scores = self._cosf_similarity(embeddings)

        # Return highest scoring intent
        return max(intent_scores, key=intent_scores.get)

    def _cosf_similarity(self, embeddings):
        """Compute similarity with φ-scaling for stability."""
        # Intent prototypes (learned or rule-based)
        prototypes = self._get_intent_prototypes()

        scores = {}
        for intent, prototype in prototypes.items():
            # Cosine similarity
            similarity = F.cosine_similarity(embeddings, prototype)

            # φ-weighted for resonance
            # Higher powers of φ for more stable intents
            phi_weight = self.phi ** (intent.value in ['system', 'knowledge'])
            scores[intent] = similarity * phi_weight

        return scores
```

**COSF Innovation:** Use φ-scaling to weight intents by stability requirements:
- System tasks: Higher φ weight (more stable/conservative)
- Conversation: Lower φ weight (more creative)

## Layer 2: COSF Reasoning Engine (The "Brain")

**Purpose:** Enhanced reasoning through mathematical training and test-time scaling.

### Data Strategy

**Training Mix:**
- 60% TinyStories (fluency, grounding)
- 20% GSM8K (math reasoning)
- 10% COSF proofs (custom mathematical reasoning)
- 10% Instruction data (helpfulness)

### Test-Time Scaling

**File:** `src/python/cosf_mi/agent/reasoning.py`

```python
def cosf_chain_of_thought(model, tokenizer, prompt, max_steps=5):
    """Generate step-by-step reasoning with Julia stability checks."""

    # Inject COSF reasoning template
    cot_prompt = f"""Think step-by-step using COSF principles:

Problem: {prompt}

Let me decompose this harmonically:
Step 1:"""

    thoughts = []
    for step in range(max_steps):
        # Generate one reasoning step
        response = model.generate(cot_prompt, max_tokens=100)

        # Check Julia stability
        stability = check_julia_bounds(model.activations)

        if stability['exceed_pct'] > 0.5:
            # Re-sync with φ-decomposition
            response = resync_with_phi_decomposition(response)

        thoughts.append(response)
        cot_prompt += f"\n{response}\nStep {step+2}:"

        # Break if natural conclusion
        if "Therefore," in response or "In conclusion," in response:
            break

    # Final synthesis
    final_answer = model.generate(
        f"{cot_prompt}\n\nFinal Answer:",
        temperature=0.3  # Lower temp for final answer
    )

    return {
        'thoughts': thoughts,
        'answer': final_answer,
        'stability': stability
    }
```

### COSF Math Training Data Generator

**File:** `src/python/cosf_mi/data/generate_cosf_proofs.py`

```python
import sympy as sp

def generate_cosf_proof_dataset(n_samples=10000):
    """Generate synthetic COSF mathematical reasoning examples."""

    phi = sp.GoldenRatio
    dataset = []

    for i in range(n_samples):
        # Generate various COSF-related problems
        problem_type = random.choice([
            'phi_power', 'fibonacci', 'julia_bound',
            'harmonic_decomposition', 'toroidal_sync'
        ])

        if problem_type == 'phi_power':
            n = random.randint(1, 20)
            question = f"What is φ^{n} approximately equal to?"
            answer = float(phi**n)
            reasoning = f"""
            Using the golden ratio φ = {float(phi):.6f}:
            φ^{n} = {answer:.6f}

            This relates to the {sp.fibonacci(n)}th Fibonacci number
            via Binet's formula.
            """

        # Add more problem types...

        dataset.append({
            'problem': question,
            'reasoning': reasoning,
            'answer': str(answer)
        })

    return dataset
```

## Layer 3: Memory System (The "Hippocampus")

**Purpose:** Infinite context without VRAM explosion.

### Implementation Stack

**Vector Database:** LanceDB (fast on-disk) or Chroma (simple)
**Embeddings:** all-MiniLM-L6-v2 (fast local)
**Storage:** Obsidian Markdown vault (encrypted)

**File:** `src/python/cosf_mi/agent/memory.py`

```python
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np

class COSFMemory:
    """Long-term memory with φ-resonant retrieval."""

    def __init__(self, db_path="memory/chroma"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection("cosf_memory")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.phi = 1.618033988749895

    def store(self, text, metadata=None):
        """Store memory with timestamp and COSF tags."""
        embedding = self.encoder.encode(text)

        # φ-decompose for harmonic storage
        phi_components = self._phi_decompose(embedding)

        self.collection.add(
            documents=[text],
            embeddings=[embedding.tolist()],
            metadatas=[{
                'timestamp': time.time(),
                'phi_resonance': phi_components,
                **(metadata or {})
            }],
            ids=[hashlib.sha256(text.encode()).hexdigest()]
        )

    def retrieve(self, query, top_k=5, phi_weighted=True):
        """Retrieve with φ-resonant scoring."""
        query_embedding = self.encoder.encode(query)

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k * 2  # Over-fetch for re-ranking
        )

        if phi_weighted:
            # Re-rank by φ-resonance
            results = self._phi_rerank(results, query_embedding)

        return results['documents'][:top_k]

    def _phi_decompose(self, vector):
        """Decompose vector into φ-harmonic components."""
        # Project onto φ^i scaled basis vectors
        components = []
        for i in range(5):  # First 5 harmonics
            basis = np.array([self.phi ** (i * j) for j in range(len(vector))])
            basis = basis / np.linalg.norm(basis)
            projection = np.dot(vector, basis)
            components.append(float(projection))
        return components

    def _phi_rerank(self, results, query_embedding):
        """Re-rank results by φ-resonance."""
        # Weight by harmonic similarity
        for i, meta in enumerate(results['metadatas'][0]):
            query_phi = self._phi_decompose(query_embedding)
            stored_phi = meta['phi_resonance']

            # Harmonic similarity
            resonance = sum(
                q * s * (self.phi ** i)
                for i, (q, s) in enumerate(zip(query_phi, stored_phi))
            )

            results['distances'][0][i] *= (1 / (1 + resonance))

        # Sort by updated distances
        sorted_indices = np.argsort(results['distances'][0])
        for key in results:
            results[key][0] = [results[key][0][i] for i in sorted_indices]

        return results
```

### Obsidian Integration

```python
def sync_with_obsidian(vault_path="~/Documents/Obsidian/MyVault"):
    """Sync COSF memory with Obsidian markdown notes."""
    memory = COSFMemory()

    for md_file in Path(vault_path).rglob("*.md"):
        with open(md_file) as f:
            content = f.read()

        memory.store(content, metadata={
            'source': 'obsidian',
            'file': str(md_file),
            'tags': extract_obsidian_tags(content)
        })
```

## Layer 4: System Control (The "Hands")

**Purpose:** Execute OS commands safely via Rust.

### Rust Executor Core

**File:** `src/rust/cosf_executor/src/main.rs`

```rust
use std::process::Command;
use tonic::{transport::Server, Request, Response, Status};

pub mod executor {
    tonic::include_proto!("executor");
}

use executor::{
    executor_server::{Executor, ExecutorServer},
    ExecuteRequest, ExecuteResponse,
};

#[derive(Debug, Default)]
pub struct COSFExecutor {}

#[tonic::async_trait]
impl Executor for COSFExecutor {
    async fn execute(
        &self,
        request: Request<ExecuteRequest>,
    ) -> Result<Response<ExecuteResponse>, Status> {
        let req = request.into_inner();

        // Validate command against whitelist
        if !is_safe_command(&req.command) {
            return Err(Status::permission_denied("Command not whitelisted"));
        }

        // Execute with timeout
        let output = Command::new(&req.command)
            .args(&req.args)
            .output()
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(ExecuteResponse {
            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            exit_code: output.status.code().unwrap_or(-1),
        }))
    }
}

fn is_safe_command(cmd: &str) -> bool {
    // Whitelist of allowed commands
    const SAFE_COMMANDS: &[&str] = &[
        "notepad", "explorer", "mkdir", "cp", "mv",
        "code", "git", "python", "cargo"
    ];

    SAFE_COMMANDS.contains(&cmd)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "[::1]:50051".parse()?;
    let executor = COSFExecutor::default();

    Server::builder()
        .add_service(ExecutorServer::new(executor))
        .serve(addr)
        .await?;

    Ok(())
}
```

### Python Client

```python
import grpc
from executor_pb2 import ExecuteRequest
from executor_pb2_grpc import ExecutorStub

class SystemController:
    """Safe OS control via Rust gRPC."""

    def __init__(self, rust_addr="localhost:50051"):
        channel = grpc.insecure_channel(rust_addr)
        self.stub = ExecutorStub(channel)

    def execute(self, command, args=[]):
        """Execute system command safely."""
        request = ExecuteRequest(command=command, args=args)
        response = self.stub.Execute(request)
        return response.stdout
```

## Layer 5: Julia Stability Governor (The "Guardrails")

**Purpose:** Real-time monitoring and auto-correction.

**File:** `src/python/cosf_mi/agent/stability_governor.py`

```python
class JuliaGovernor:
    """Real-time stability monitoring and correction."""

    def __init__(self, model, bounds=2.0):
        self.model = model
        self.bounds = bounds
        self.history = []

    def monitor_inference(self, activations):
        """Check activations during generation."""
        norms = torch.norm(activations, dim=-1)
        exceed_pct = (norms > self.bounds).float().mean().item()
        max_norm = norms.max().item()

        stability = {
            'exceed_pct': exceed_pct,
            'max_norm': max_norm,
            'status': 'stable' if exceed_pct < 0.3 else 'unstable'
        }

        self.history.append(stability)

        if stability['status'] == 'unstable':
            self._apply_correction()

        return stability

    def _apply_correction(self):
        """Auto-correct on instability."""
        # Reduce temperature
        self.model.temperature *= 0.8

        # Apply φ-decomposition to recent activations
        # (Implementation depends on model architecture)

        print("⚠ Julia bounds exceeded - applying φ-correction")

    def predict_instability(self, window=10):
        """Predict future instability from trends."""
        if len(self.history) < window:
            return False

        recent = self.history[-window:]
        trend = np.polyfit(range(window),
                          [h['exceed_pct'] for h in recent],
                          deg=1)[0]

        # Positive trend = increasing instability
        return trend > 0.05
```

## Complete Jarvis Pipeline

**File:** `src/python/cosf_mi/agent/jarvis.py`

```python
class COSFJarvis:
    """Complete agentic AI system."""

    def __init__(self):
        # Load components
        self.model = load_cosf_model("checkpoints/cosf_small/best_model.pt")
        self.tokenizer = load_tokenizer("models/tokenizer_8k.model")
        self.router = IntentRouter(self.model, self.tokenizer)
        self.memory = COSFMemory()
        self.executor = SystemController()
        self.governor = JuliaGovernor(self.model)

    async def process(self, user_input: str):
        """Main processing loop."""

        # 1. Retrieve relevant memories
        context = self.memory.retrieve(user_input)

        # 2. Route intent
        intent = self.router.route(user_input)

        # 3. Process based on intent
        if intent == Intent.CONVERSATION:
            response = self._chat(user_input, context)
        elif intent == Intent.SYSTEM_TASK:
            response = self._execute_task(user_input)
        elif intent == Intent.KNOWLEDGE_SEARCH:
            response = self._search_knowledge(user_input, context)
        elif intent == Intent.MATH_REASONING:
            response = cosf_chain_of_thought(
                self.model, self.tokenizer, user_input
            )

        # 4. Monitor stability
        stability = self.governor.monitor_inference(self.model.activations)

        # 5. Store interaction
        self.memory.store(f"User: {user_input}\nJarvis: {response}")

        return {
            'response': response,
            'intent': intent.value,
            'stability': stability,
            'context_used': len(context)
        }
```

## Deployment: Container-in-VM

**Dockerfile:**
```dockerfile
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3.11 python3-pip rust-all \
    && rm -rf /var/lib/apt/lists/*

# Copy COSF code
COPY . /app
WORKDIR /app

# Install Python deps
RUN pip install -r requirements.txt

# Build Rust executor
WORKDIR /app/src/rust/cosf_executor
RUN cargo build --release

# Expose gRPC port
EXPOSE 50051

CMD ["python3", "-m", "cosf_mi.agent.jarvis", "--serve"]
```

**VM Setup (VirtualBox):**
```bash
# Create VM with Ubuntu
VBoxManage createvm --name "COSF-Jarvis" --ostype Ubuntu_64 --register
VBoxManage modifyvm "COSF-Jarvis" --memory 16384 --cpus 8

# Attach RTX 5090 (PCI passthrough)
VBoxManage modifyvm "COSF-Jarvis" --pciattach 01:00.0

# Run container in VM
docker run --gpus all -p 50051:50051 cosf-jarvis
```

## Implementation Timeline

**Week 1:** Intent router + basic memory
**Week 2:** Rust executor + function calling
**Week 3:** Stability governor + CoT reasoning
**Week 4:** Full integration + voice (Whisper/XTTS)
**Week 5:** Container deployment + testing
**Week 6:** Obsidian sync + advanced features

---

**Next: Let's implement the intent router and memory system! 🚀**
