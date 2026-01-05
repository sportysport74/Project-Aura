"""
COSF Agent: Agentic AI System

Transform COSF from text generation to a full AI agent with:
- Intent routing
- Memory (RAG)
- OS control
- Stability monitoring
"""

from .intent_router import IntentRouter, Intent
from .memory import COSFMemory
from .stability_governor import JuliaGovernor

__all__ = [
    'IntentRouter',
    'Intent',
    'COSFMemory',
    'JuliaGovernor',
]
