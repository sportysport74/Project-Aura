"""
Intent Router: Classify user inputs into actionable intents

Uses φ-weighted similarity for stable routing.
"""

from enum import Enum
from typing import Dict, List
import torch
import torch.nn.functional as F
import numpy as np

from ..core.constants import PHI


class Intent(Enum):
    """Intent categories for routing."""
    CONVERSATION = "conversation"   # Chat, general Q&A
    SYSTEM_TASK = "system"         # OS operations, file management
    KNOWLEDGE_SEARCH = "knowledge" # RAG retrieval from memory
    MATH_REASONING = "math"        # Deep reasoning with CoT


class IntentRouter:
    """Route user inputs using COSF-weighted classification.

    Uses φ-harmonic similarity to weight intents by stability requirements.
    """

    def __init__(self, model, tokenizer):
        """Initialize router.

        Args:
            model: COSF transformer model
            tokenizer: Tokenizer instance
        """
        self.model = model
        self.tokenizer = tokenizer
        self.phi = PHI

        # Intent prototypes (can be learned or rule-based)
        self.prototypes = self._init_prototypes()

    def _init_prototypes(self) -> Dict[Intent, torch.Tensor]:
        """Initialize intent prototypes from example phrases."""

        examples = {
            Intent.CONVERSATION: [
                "hello", "how are you", "tell me about",
                "what do you think", "explain", "describe"
            ],
            Intent.SYSTEM_TASK: [
                "open notepad", "create folder", "run command",
                "delete file", "list files", "execute"
            ],
            Intent.KNOWLEDGE_SEARCH: [
                "search my notes", "find in obsidian", "recall",
                "what did i write about", "look up", "retrieve"
            ],
            Intent.MATH_REASONING: [
                "calculate", "prove", "derive", "what is phi",
                "solve for", "compute", "reason about"
            ],
        }

        prototypes = {}
        self.model.eval()

        with torch.no_grad():
            for intent, phrases in examples.items():
                # Encode all examples
                embeddings = []
                for phrase in phrases:
                    tokens = self.tokenizer.encode(phrase)
                    tokens_tensor = torch.tensor([tokens], dtype=torch.long)

                    # Get embeddings from model
                    # (This assumes your model has an embedding layer accessible)
                    emb = self._get_embedding(tokens_tensor)
                    embeddings.append(emb)

                # Average to get prototype
                prototype = torch.stack(embeddings).mean(dim=0)
                prototypes[intent] = prototype

        return prototypes

    def _get_embedding(self, tokens: torch.Tensor) -> torch.Tensor:
        """Get embedding vector from tokens.

        Args:
            tokens: Token tensor [batch, seq_len]

        Returns:
            Embedding vector [hidden_dim]
        """
        # Use model's embedding layer
        if hasattr(self.model, 'token_embedding'):
            emb = self.model.token_embedding(tokens)
        elif hasattr(self.model, 'embedding'):
            emb = self.model.embedding(tokens)
        else:
            # Fallback: run through model and take last hidden state
            with torch.no_grad():
                outputs = self.model(tokens)
                # For transformer, take mean of sequence
                emb = outputs.mean(dim=1)

        # Return mean pooling of sequence
        return emb.squeeze().mean(dim=0) if emb.dim() > 1 else emb.squeeze()

    def route(self, user_input: str) -> Intent:
        """Classify intent using φ-weighted embeddings.

        Args:
            user_input: User's input text

        Returns:
            Detected intent
        """
        # Encode input
        tokens = self.tokenizer.encode(user_input)
        tokens_tensor = torch.tensor([tokens], dtype=torch.long)

        # Get embedding
        self.model.eval()
        with torch.no_grad():
            input_embedding = self._get_embedding(tokens_tensor)

        # Compute φ-weighted similarity
        intent_scores = self._cosf_similarity(input_embedding)

        # Return highest scoring intent
        return max(intent_scores, key=intent_scores.get)

    def _cosf_similarity(self, embedding: torch.Tensor) -> Dict[Intent, float]:
        """Compute similarity with φ-scaling for stability.

        Args:
            embedding: Input embedding vector

        Returns:
            Dictionary of intent scores
        """
        scores = {}

        for intent, prototype in self.prototypes.items():
            # Ensure same device
            prototype = prototype.to(embedding.device)

            # Cosine similarity
            similarity = F.cosine_similarity(
                embedding.unsqueeze(0),
                prototype.unsqueeze(0),
                dim=1
            ).item()

            # φ-weighted for resonance
            # Higher powers of φ for more stable/critical intents
            if intent == Intent.SYSTEM_TASK:
                # System tasks need highest stability
                phi_weight = self.phi ** 2
            elif intent == Intent.KNOWLEDGE_SEARCH:
                # Knowledge search needs moderate stability
                phi_weight = self.phi ** 1
            elif intent == Intent.MATH_REASONING:
                # Math reasoning needs high precision
                phi_weight = self.phi ** 1.5
            else:
                # Conversation can be more creative
                phi_weight = 1.0

            scores[intent] = similarity * phi_weight

        return scores

    def route_with_confidence(self, user_input: str) -> tuple[Intent, float]:
        """Route and return confidence score.

        Args:
            user_input: User's input text

        Returns:
            Tuple of (intent, confidence)
        """
        tokens = self.tokenizer.encode(user_input)
        tokens_tensor = torch.tensor([tokens], dtype=torch.long)

        self.model.eval()
        with torch.no_grad():
            input_embedding = self._get_embedding(tokens_tensor)

        scores = self._cosf_similarity(input_embedding)

        # Get top intent
        top_intent = max(scores, key=scores.get)
        top_score = scores[top_intent]

        # Compute confidence as softmax of scores
        score_values = np.array(list(scores.values()))
        softmax_scores = np.exp(score_values) / np.exp(score_values).sum()
        confidence = softmax_scores.max()

        return top_intent, confidence


# Rule-based fallback for simpler cases
class SimpleIntentRouter:
    """Simple keyword-based intent router (fallback)."""

    def __init__(self):
        self.keywords = {
            Intent.SYSTEM_TASK: [
                'open', 'close', 'start', 'stop', 'run', 'execute',
                'create', 'delete', 'remove', 'copy', 'move', 'list'
            ],
            Intent.KNOWLEDGE_SEARCH: [
                'search', 'find', 'lookup', 'recall', 'retrieve',
                'my notes', 'obsidian', 'remember'
            ],
            Intent.MATH_REASONING: [
                'calculate', 'compute', 'solve', 'prove', 'derive',
                'phi', 'fibonacci', 'what is', 'how much'
            ],
        }

    def route(self, user_input: str) -> Intent:
        """Route based on keywords.

        Args:
            user_input: User's input text

        Returns:
            Detected intent
        """
        user_lower = user_input.lower()

        # Check for keywords
        for intent, keywords in self.keywords.items():
            if any(keyword in user_lower for keyword in keywords):
                return intent

        # Default to conversation
        return Intent.CONVERSATION
